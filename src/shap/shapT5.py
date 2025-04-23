import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.exceptions import ConvergenceWarning
import warnings

from ..autoencoder.model import Model
from ..autoencoder.config import Config

warnings.filterwarnings("ignore", category=ConvergenceWarning)
torch.set_grad_enabled(False)


class ShapleyExplainer:
	"""
	Explainer class implementing SHAP (SHapley Additive exPlanations) values computation
	for text generation models to interpret token influence.
	"""

	def __init__(self, init_tokens=2):
		"""
		Initialize the SHAP explainer with model and tokenizer.

		Args:
			init_tokens: Number of initial tokens to use before generation
		"""
		self.init_tokens = init_tokens
		self.tokenizer = AutoTokenizer.from_pretrained(Config.model)
		self.model = self._load_model()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# State variables for generation and explanation
		self.memory = None
		self.input_ids = None
		self.correct_token_id = None

	@staticmethod
	def _load_model():
		"""Load and prepare the model with saved checkpoint."""
		model = Model(Config.model, Config.bias, Config.temperature)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		checkpoint = torch.load("checkpoint.pth", map_location=device, weights_only=True)
		model.eval()
		model.load_state_dict(checkpoint["model"])

		return model.to(device)

	def explain_generation(self, sentence, steps, nsamples=300, show_progress=False):
		"""
		Generate tokens and compute SHAP values to explain token influence.

		Args:
			sentence: Input text to continue from
			steps: Number of generation steps to explain
			nsamples: Number of coalition samples for SHAP estimation
			show_progress: Whether to show progress bars and intermediate outputs

		Returns:
			tuple: (shap_values, generated_tokens, input_tokens)
		"""
		# Encode and prepare the input sentence
		tokenized = self.tokenizer.encode(sentence)
		memory_ids = torch.tensor(tokenized).unsqueeze(0).to(self.device)

		# Initialize memory state
		self.memory = self._encode_memory(memory_ids)
		self.input_ids = memory_ids[:, :self.init_tokens]

		# Generate tokens
		tokens = self._generate_tokens(steps, memory_ids)
		decoded_tokens = [self.tokenizer.decode([token], skip_special_tokens=False) for token in tokens[0]]

		if show_progress:
			print(tokens)
			print(decoded_tokens)

		# Compute SHAP values for each generated token
		values_array = []
		for i in range(self.init_tokens, steps + self.init_tokens):
			self.input_ids = tokens[:, :i]
			self.correct_token_id = tokens[:, i]

			data, total_inds = self._prepare_data()
			shap_values = self._compute_shap_values(
				total_inds,
				nsamples=nsamples,
				show_progress=show_progress,
			).reshape(1, -1, 1)

			values_array.append(shap_values)

		# Get tokens that influence the generation (tokens with gate_mask=0)
		decoded_memory = [
			self.tokenizer.decode([token], skip_special_tokens=False)
			for token in memory_ids[0][self.memory.gate_mask[0] == 0]
		]

		return values_array, decoded_tokens, decoded_memory

	def _encode_memory(self, memory_ids):
		"""Encode input tokens into model memory state."""
		return self.model.encode(
			tokens=memory_ids,
			eos_mask=torch.stack((
				torch.arange(1, device=self.device),
				torch.tensor([memory_ids.shape[1] - 1], device=self.device, dtype=torch.long)
			)),
			pad_mask=torch.ones((1, memory_ids.shape[1]), dtype=torch.bool).to(self.device),
			attn_mask=None,
			attn_scores=False,
		)

	def _generate_tokens(self, num_steps, memory_ids):
		"""Generate a sequence of tokens from the model."""
		tokens = self.input_ids.clone()

		for _ in range(num_steps):
			result = self.model.forward(
				memory_tokens=memory_ids,
				memory_pad_mask=torch.ones((1, memory_ids.shape[1]), dtype=torch.bool).to(self.device),
				memory_attn_mask=None,
				memory_eos_mask=torch.stack((
					torch.arange(1, device=self.device),
					torch.tensor([memory_ids.shape[1] - 1], device=self.device, dtype=torch.long)
				)),
				input_tokens=tokens,
				input_pad_mask=torch.ones((1, tokens.shape[1]), dtype=torch.bool).to(self.device),
				input_attn_mask=None,
				memory_attn_scores=False,
			)

			logits = result[1][:, -1, :]
			prediction = torch.argmax(logits, dim=-1).unsqueeze(0)
			tokens = torch.cat((tokens, prediction), dim=-1)

		return tokens

	def _prepare_data(self):
		"""Prepare data for SHAP computation."""
		original_gate_mask = self.memory.gate_mask
		valid_indices = (original_gate_mask == 0).squeeze()
		total_inds = valid_indices.sum()
		num_valid_tokens = total_inds.item()
		data = torch.ones((1, num_valid_tokens))

		return data.cpu().detach().numpy(), total_inds

	def _compute_shap_values(self, total_inds, nsamples=300, show_progress=False):
		"""
		Compute SHAP values by sampling coalitions and measuring marginal contributions.

		This implements the Monte Carlo approximation of Shapley values.
		"""
		total_inds_int = total_inds.item() if isinstance(total_inds, torch.Tensor) else total_inds
		shap_values = np.zeros(total_inds_int)

		# Generate random coalitions (0: include feature, 1: exclude feature)
		coalitions = np.random.randint(0, 2, size=(nsamples, total_inds_int))

		# Iterate through each feature to compute its SHAP value
		iterator = tqdm(range(total_inds_int), desc="Computing SHAP values") if show_progress else range(total_inds_int)

		for feature_idx in iterator:
			# Create copies with and without the feature
			coalitions_with_feature = coalitions.copy()
			coalitions_without_feature = coalitions.copy()

			coalitions_with_feature[:, feature_idx] = 0  # Include the feature
			coalitions_without_feature[:, feature_idx] = 1  # Exclude the feature

			preds_with_feature = self._predict_with_coalition(coalitions_with_feature)
			preds_without_feature = self._predict_with_coalition(coalitions_without_feature)

			marginal_contributions = preds_with_feature - preds_without_feature

			# Weight by coalition size
			coalition_sizes = np.sum(coalitions, axis=1)
			weights = np.array([
				self._compute_coalition_weight(size, total_inds_int)
				for size in coalition_sizes
			])

			weighted_contributions = weights * marginal_contributions.squeeze()
			feature_contribution = np.sum(weighted_contributions)

			# Normalize and store the SHAP value
			shap_values[feature_idx] = feature_contribution / np.sum(weights)

		return shap_values

	def _compute_coalition_weight(self, coalition_size, total_features):
		"""
		Compute the weight for a coalition of a given size.

		These weights ensure proper Shapley value calculation.
		"""
		if coalition_size == 0 or coalition_size == total_features:
			return 1.0 / total_features

		n = total_features
		s = coalition_size

		binom_coeff = math.comb(n - 1, s)
		return 1.0 / (n * binom_coeff)

	def _predict_with_coalition(self, masked_indices):
		"""
		Make predictions using the model with the given feature coalition.

		Args:
			masked_indices: Boolean mask where True means the feature is excluded

		Returns:
			Probability of the correct token
		"""
		masked_indices = torch.tensor(masked_indices, dtype=torch.bool, device=self.device)
		if masked_indices.ndim == 1:
			masked_indices = masked_indices.unsqueeze(0)

		batch_size = masked_indices.shape[0]

		# Identify valid tokens in memory
		valid_indices = (self.memory.gate_mask == 0)
		vocab_size = len(self.tokenizer.vocab)

		# Create full mask for all tokens in batch
		full_mask = torch.zeros_like(self.memory.gate_mask, dtype=torch.bool, device=self.device)
		full_mask = full_mask.repeat(batch_size, 1)
		full_mask[valid_indices.repeat(batch_size, 1)] = masked_indices.flatten()

		# Modify gate mask to exclude masked tokens
		modified_gate_mask = self.memory.gate_mask.clone().repeat(batch_size, 1)
		modified_gate_mask[full_mask] = -torch.inf

		# Run model with modified memory
		result = self.model.decode(
			memory=Model.Memory(
				pad_mask=self.memory.pad_mask.repeat(batch_size, 1),
				gate_masks=modified_gate_mask.unsqueeze(0),
				embeds=self.memory.embeds.repeat(batch_size, 1, 1),
				kv_dim=64,
				attn_scores=None,
			),
			tokens=self.input_ids.repeat(batch_size, 1),
			pad_mask=torch.ones((batch_size, self.input_ids.shape[1]), dtype=torch.bool).to(self.device),
			attn_mask=None
		)

		logits = result[:, -1, :].unsqueeze(1)
		probs = torch.softmax(logits, dim=-1)
		correct_token_prob = probs[:, :, self.correct_token_id].cpu().detach().numpy()

		# Handle NaN values by replacing with uniform probability
		if np.isnan(correct_token_prob).any():
			uniform_prob = 1.0 / vocab_size
			correct_token_prob[np.isnan(correct_token_prob)] = uniform_prob

		return correct_token_prob


def visualize_influence(sentence, influence, key_words, skip_tokens=0, save_path=None):
	"""
	Visualize the influence of input tokens on generated tokens using a heatmap.

	Args:
		sentence: List of generated tokens to display on x-axis
		influence: Array of shape (len(key_words), len(sentence)-skip_tokens) with SHAP values
		key_words: List of input tokens to display on y-axis
		skip_tokens: Number of initial tokens to skip in visualization
		save_path: Path to save the figure, if provided
	"""
	assert influence.shape == (len(key_words), len(sentence) - skip_tokens), \
		f"Influence shape must be ({len(key_words)}, {len(sentence) - skip_tokens}), but got {influence.shape}"

	n_total = len(sentence)
	k = len(key_words)

	full_influence = np.full((k, n_total), np.nan)
	full_influence[:, skip_tokens:] = influence

	# Adjust figure size based on content
	fig_width = max(8, 0.7 * n_total)
	fig_height = max(2.5, 0.6 * k)

	plt.figure(figsize=(fig_width, fig_height))
	ax = sns.heatmap(
		full_influence,
		annot=[[f"{val:.2f}" if not np.isnan(val) else "" for val in row] for row in full_influence],
		xticklabels=sentence,
		yticklabels=key_words,
		cmap="RdBu_r",
		center=0,
		linewidths=0.5,
		cbar_kws={'label': 'Influence Score'},
		fmt=""
	)

	# Gray out initial tokens that were not generated
	for x in range(skip_tokens):
		ax.add_patch(plt.Rectangle((x, -0.5), 1, k, fill=True, color='lightgray', alpha=0.3, lw=0))

	plt.title("Influence of Key Words on Sentence Tokens")
	plt.xlabel("Generated Tokens")
	plt.ylabel("Input Tokens")
	plt.tight_layout()

	if save_path:
		plt.savefig(save_path, bbox_inches='tight')
	else:
		plt.show()


def explain_text_generation(start_text, num_steps, nsamples=300, init_tokens=2, show_progress=False):
	"""
	Main function to explain a text generation model using SHAP values.

	Args:
		start_text: The input text to continue from
		num_steps: Number of generation steps to explain
		nsamples: Number of coalition samples for SHAP estimation
		init_tokens: Number of initial tokens to use before generation
		show_progress: Whether to show progress bars and intermediate outputs

	Returns:
		None (displays visualization)
	"""

	explainer = ShapleyExplainer(init_tokens=init_tokens)
	results, decoded_tokens, input_tokens = explainer.explain_generation(
		start_text,
		steps=num_steps,
		nsamples=nsamples,
		show_progress=show_progress
	)

	accumulated_values = np.squeeze(np.concatenate(results, axis=2), axis=0)

	visualize_influence(decoded_tokens, accumulated_values, input_tokens, skip_tokens=init_tokens)


if __name__ == "__main__":
	example_text = "This cafe is a hidden gem! The cozy atmosphere and excellent coffee make it the perfect spot to relax and unwind. The staff are friendly and attentive, always ensuring that your cup is full."
	explain_text_generation(example_text, 25, show_progress=True)
