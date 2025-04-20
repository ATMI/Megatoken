import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import prepare
from model import Model
from transformers import AutoTokenizer
from config import Config
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
import seaborn as sns
import math

warnings.filterwarnings("ignore", category=ConvergenceWarning)
torch.set_grad_enabled(False)
INIT_TOKENS = 2
show_progress = True

def plot_seaborn(sentence, influence, key_words, skip_tokens=0, save_path=None):
    assert influence.shape == (len(key_words), len(sentence) - skip_tokens), \
        f"Influence shape must be ({len(key_words)}, {len(sentence) - skip_tokens}), but got {influence.shape}"

    n_total = len(sentence)
    k = len(key_words)

    full_influence = np.full((k, n_total), np.nan)
    full_influence[:, skip_tokens:] = influence

    fig_width = max(8, 0.7 * n_total)
    fig_height = max(2.5, 0.6 * k)

    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(
        full_influence,
        annot=[[f"{val:.2f}" if not np.isnan(val) else "" for val in row] for row in full_influence],
        xticklabels=sentence,
        yticklabels=key_words,
        cmap="RdBu",
        center=0,
        linewidths=0.5,
        cbar_kws={'label': 'Influence Score'},
        fmt=""
    )

    for x in range(skip_tokens):
        ax.add_patch(plt.Rectangle((x, -0.5), 1, k, fill=True, color='lightgray', alpha=0.3, lw=0))

    plt.title("Influence of Key Words on Sentence Tokens")
    plt.xlabel("Decoded Sentence")
    plt.ylabel("Key Words")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

class CustomSHAPexplainer():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.model)
        self.model = prepare.model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load("checkpoint.pth", map_location=self.device, weights_only=True)
        self.model.eval()
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.to(self.device)

        # buffer
        self.correct_token_id = None
        self.memory = None
        self.input_ids = None

    def shap_eval(self, sentence, steps, nsamples=300):
        tokenized = self.tokenizer.encode(sentence)
        memory_ids = torch.tensor(tokenized).unsqueeze(0).to(self.device)
        self.memory = self.model.encode(
            tokens=memory_ids,
            eos_mask=torch.stack((
                torch.arange(1, device=self.device),
                torch.tensor([memory_ids.shape[1] - 1], device=self.device, dtype=torch.long)
            )),

            pad_mask=torch.ones((1, memory_ids.shape[1]), dtype=torch.bool).to(self.device),
            attn_mask=None,
        )
        self.input_ids = memory_ids[:, :INIT_TOKENS]

        tokens = self.generate_tokens(steps, memory_ids)
        decoded_tokens = [self.tokenizer.decode([token], skip_special_tokens=False) for token in tokens[0]]
        if show_progress:
            print(tokens)
            print(decoded_tokens)

        values_array = []
        for i in range(INIT_TOKENS, steps + INIT_TOKENS):
            self.input_ids = tokens[:, :i]
            self.correct_token_id = tokens[:, i]
            data, total_inds = self.prepare_data()

            shap_values = self.compute_shap_values(data, total_inds, nsamples=nsamples).reshape(1, -1, 1)
            values_array.append([shap_values, self.tokenizer.decode(self.correct_token_id)])

        decoded_memory = [self.tokenizer.decode([token], skip_special_tokens=False) for token in memory_ids[self.memory.gate_mask==0]]

        return values_array, decoded_tokens, decoded_memory

    def generate_tokens(self, num_steps, memory_ids):
        tokens = self.input_ids.clone()
        for i in range(num_steps):
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
                input_attn_mask=None

            )

            logits = result.logits[:, -1, :]
            prediction = torch.argmax(logits, dim=-1).unsqueeze(0)
            tokens = torch.cat((tokens, prediction), dim=-1)

        return tokens

    def compute_shap_values(self, data, total_inds, nsamples=300):
        total_inds_int = total_inds.item() if isinstance(total_inds, torch.Tensor) else total_inds

        shap_values = np.zeros(total_inds_int)

        coalitions = np.random.randint(0, 2, size=(nsamples, total_inds_int))

        iterator = range(total_inds_int)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing SHAP values")

        for feature_idx in iterator:
            coalitions_with_feature = coalitions.copy()
            coalitions_without_feature = coalitions.copy()

            coalitions_with_feature[:, feature_idx] = 0
            coalitions_without_feature[:, feature_idx] = 1

            preds_with_feature = self.shap_predict(coalitions_with_feature)
            preds_without_feature = self.shap_predict(coalitions_without_feature)

            marginal_contributions = preds_with_feature - preds_without_feature

            coalition_sizes = np.sum(coalitions, axis=1)
            weights = np.array([self.compute_coalition_weight(size, total_inds_int) for size in coalition_sizes])

            weighted_contributions = weights * marginal_contributions.squeeze()
            feature_contribution = np.sum(weighted_contributions)

            shap_values[feature_idx] = feature_contribution / np.sum(weights)

        return shap_values
    
    def compute_coalition_weight(self, coalition_size, total_features):
        if coalition_size == 0 or coalition_size == total_features:
            return 1.0 / total_features

        n = total_features
        s = coalition_size

        binom_coeff = math.comb(n-1, s-1)
        
        return 1.0 / (n * binom_coeff)

    def shap_predict(self, masked_indices):
        masked_indices = torch.tensor(masked_indices, dtype=torch.bool, device=self.device)
        if masked_indices.ndim == 1:
            masked_indices = masked_indices.unsqueeze(0)

        batch_size = masked_indices.shape[0]

        valid_indices = (self.memory.gate_mask == 0)
        vocab_size = len(self.tokenizer.vocab)
        full_mask = torch.zeros_like(self.memory.gate_mask, dtype=torch.bool, device=self.device)
        full_mask = full_mask.repeat(batch_size, 1)
        full_mask[valid_indices.repeat(batch_size, 1)] = masked_indices.flatten()
        modified_gate_mask = self.memory.gate_mask.clone().repeat(batch_size, 1)
        modified_gate_mask[full_mask] = -torch.inf
        result = self.model.decode(
            memory=Model.Memory(
                pad_mask=self.memory.pad_mask.repeat(batch_size, 1),
                gate_mask=modified_gate_mask,
                embeds=self.memory.embeds.repeat(batch_size, 1, 1),
                volume=self.memory.volume.repeat(batch_size, 1),
            ),
            tokens=self.input_ids.repeat(batch_size, 1),
            pad_mask=torch.ones((batch_size, self.input_ids.shape[1]), dtype=torch.bool).to(self.device),
            attn_mask=None
        )
        logits = result[:, -1, :].unsqueeze(1)
        probs = torch.softmax(logits, dim=-1)
        correct_token_prob = probs[:, :, self.correct_token_id].cpu().detach().numpy()
        if np.isnan(correct_token_prob).any():
            uniform_prob = 1.0 / vocab_size
            correct_token_prob[np.isnan(correct_token_prob)] = uniform_prob
        return correct_token_prob

    def prepare_data(self):
        original_gate_mask = self.memory.gate_mask
        valid_indices = (original_gate_mask == 0).squeeze()
        total_inds = valid_indices.sum()
        num_valid_tokens = total_inds.item()
        data = torch.ones((1, num_valid_tokens))
        data = data.cpu().detach().numpy()
        return data, total_inds

def shap_values(start_sent,steps,nsamples = 300,init_tokens = 2, progress=False):
    global INIT_TOKENS, show_progress

    INIT_TOKENS = init_tokens
    show_progress = progress

    shaper = CustomSHAPexplainer()
    results, decoded_sent, token_ids = shaper.shap_eval(start_sent, steps=steps, nsamples=nsamples)
    raw_results = [results[i][0] for i in range(len(results))]
    accumulated_results = np.squeeze(np.concatenate(raw_results, axis=2), axis=0)

    plot_seaborn(decoded_sent, accumulated_results, token_ids, skip_tokens=INIT_TOKENS)


if __name__ == "__main__":
    start_sent = "This cafe is a hidden gem! The cozy atmosphere and excellent coffee make it the perfect spot to relax and unwind. The staff are friendly and attentive, always ensuring that your cup is full."
    shap_values(start_sent, 32)
