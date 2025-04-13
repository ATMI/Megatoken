import argparse

import torch
from transformers import AutoTokenizer, T5Tokenizer

import prepare
from config import Config


def inference(
	model: torch.nn.Module,
	tokenizer: T5Tokenizer,
	text: str,
	max_length: int = 512,
):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	tokens = tokenizer(
		text,
		padding=False,
		truncation=True,
		max_length=Config.max_length,
		return_tensors="pt",
	)["input_ids"].to(device)
	pad_mask = torch.ones_like(tokens, dtype=torch.bool).to(device)
	encoder_eos_mask = torch.tensor([[0], [tokens.size(1) - 1]])

	print("Initial length:", tokens.size(1))

	model.eval()
	with torch.no_grad():
		memory = model.encode(
			tokens=tokens,
			pad_mask=pad_mask,
			attn_mask=None,
			eos_mask=encoder_eos_mask,
		)

	out_size = (memory.gate_mask.exp() != 0).sum().item()
	ratio = out_size / tokens.size(1)
	print(f"Compression rate: {ratio:.2f}")
	print("Output size:", out_size)

	out = tokens[0][:Config.decoder_visibility].unsqueeze(0)

	while True:
		seq_length = out.size(1)
		decoder_mask = torch.full((seq_length, seq_length), -torch.inf)
		for i in range(seq_length):
			decoder_mask[i:i + Config.decoder_visibility + 1, i] = 0
		decoder_mask[:, 0] = 0

		decoder_eos_mask = torch.tensor([[0], [out.size(1) - 1]])
		with torch.no_grad():
			logits = model.decode(
				memory=memory,
				tokens=out,
				pad_mask=torch.ones_like(out, dtype=torch.bool).to(device),
				attn_mask=decoder_mask,
				eos_mask=decoder_eos_mask,
			).squeeze(0)[-1]

		next_tok = logits.argmax(dim=-1)

		if next_tok.item() == tokenizer.eos_token_id:
			print("\nModel reached EOS token!")
			break

		next_tok = torch.tensor([[next_tok]])
		out = torch.cat((out, next_tok), dim=1).to(device)

		if out.size(-1) == max_length:
			break

	output = tokenizer.decode(out.squeeze())
	print("")
	print("Initial text:\n", text, sep="")
	print("Predicted:\n", output, sep="")


def main():
	# "Good beer selection. Understaffed for a light Monday night crowd, it wasn't her fault she was the only server. But it took about an hour to get our sandwiches. Mine was one of the best reubens I've ever had."
	# "Very disappointed in the customer service. We ordered Reuben's and wanted coleslaw instead of kraut. They charged us $3.00 for the coleslaw. We will not be back . The iced tea is also terrible tasting."
	args = argparse.ArgumentParser()
	args.add_argument("checkpoint", type=str)
	args = args.parse_args()

	tokenizer = AutoTokenizer.from_pretrained(Config.model)
	checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))

	model = prepare.model()
	model.load_state_dict(checkpoint["model"])

	while True:
		text = input()
		if not text:
			break
		inference(model, tokenizer, text)


if __name__ == '__main__':
	main()
