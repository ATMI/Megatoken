import torch
from transformers import AutoTokenizer

from .model import Model as CoTA
from .config import Config
from .inference import get_memory


def viz(model, tokenizer, text):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	memory, tokens = get_memory(model, tokenizer, text, device, return_tokens=True)

	attention_maps, gate_values = memory.attn_scores, memory.gates

	removed_token_indices = torch.tensor([], device=device)

	for layer_idx, (layer_attention, layer_gates) in enumerate(zip(attention_maps, gate_values)):
		print(f"Layer {layer_idx}:")

		# Find eliminated tokens at this layer!
		eliminated_indices = torch.where(layer_gates == -torch.inf)[0]
		mm = ~torch.isin(eliminated_indices, removed_token_indices)
		eliminated_indices = eliminated_indices[mm]
		if eliminated_indices.numel() == 0:
			print("No compression in this layer!\n")
			continue

		# Track eliminated tokens (to display shortened sequence)
		removed_token_indices = torch.cat((removed_token_indices, eliminated_indices))

		# Mean along all Attention Heads
		attention_matrix = layer_attention.mean(dim=0)

		# Find to where eliminated tokens were merged
		eliminated_attention = attention_matrix[eliminated_indices]
		topk_values, topk_indices = torch.topk(eliminated_attention, k=3, dim=1)

		# Convert indices to tokens
		eliminated_tokens = tokenizer.convert_ids_to_tokens(tokens[0][eliminated_indices])

		# Print eliminated tokens
		for token_idx, token in enumerate(eliminated_tokens):
			print(f"- Eliminated token: '{token}', (position: {eliminated_indices[token_idx]})")

			# Go through each parent token
			for rank, (value, parent_idx) in enumerate(
					zip(topk_values[token_idx], topk_indices[token_idx]),
					start=1
			):
				parent_token = tokenizer.convert_ids_to_tokens(tokens[0][parent_idx].item())
				print(f"\t\t{rank}. Merged into '{parent_token}' (score: {value.item():.4f})")

		# Show compressed sequence
		seq_len = tokens.size(1)
		active_mask = ~torch.isin(torch.arange(seq_len, device=device), removed_token_indices)
		compressed_sequence = tokenizer.decode(tokens[0][active_mask])

		print(f"\nCompressed sequence after layer {layer_idx}:")
		print(compressed_sequence)
		print("-" * 50 + "\n")

	# Final compressed output
	final_sequence = tokenizer.decode(tokens[0][active_mask])

	print("\n" + "=" * 50)

	print("Initial Sequence:")
	print(tokenizer.decode(tokens[0]))

	print("\nFinal compressed text:")
	print(final_sequence)

	print("=" * 50)





if __name__ == "__main__":
	tokenizer = AutoTokenizer.from_pretrained(Config.model)

	ckpt = "../checkpoint/29249.pth"
	checkpoint = torch.load(ckpt, weights_only=True)

	cota_model = CoTA(Config.model, Config.bias, Config.temperature)
	cota_model.load_state_dict(checkpoint["model"])

	text = "This place is so nice! The dishes are delicious. I especially recommend trying beef stroganoff!"

	viz(cota_model, tokenizer, text)