from pathlib import Path
import random
import torch
from transformers import AutoTokenizer

from lm import LanguageModel

if __name__ == "__main__":
	torch.random.manual_seed(42)
	random.seed(42)

	ckpt_path = Path("checkpoint", "0225_1108", "2_3.pth")
	ckpt = torch.load(ckpt_path, weights_only=True)

	tokenizer_name = "FacebookAI/roberta-base"
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = LanguageModel(
		vocab_size=tokenizer.vocab_size,
		embed_dim=512,
		pad_idx=tokenizer.pad_token_id,
		head_num=8,
		layer_num=4,
		feedforward_dim=1024,
	)
	model.load_state_dict(ckpt["model"])
	model.eval()

	params = sum(p.numel() for p in model.parameters())
	print(params)

	model.to(device)

	# text = "Good. Went yesterday with family and enjoyed the oysters"
	text = "Awful! I want my money back. "
	t = 0.5

	x = tokenizer(
		[text],
		return_tensors="pt",
		return_token_type_ids=False,
		return_attention_mask=False,
	)["input_ids"]
	x = x[:, :-1]

	with torch.no_grad():
		for i in range(100):
			x = x.to(device)
			x_pad = torch.zeros_like(x, dtype=torch.bool)

			y = model(x, x_pad)
			y = y[:, -1]
			y = y / t
			y = y.softmax(dim=-1)
			y = torch.multinomial(y, num_samples=1)
			x = torch.cat((x, y), dim=-1)

			if y.item() == tokenizer.eos_token_id:
				break

	print(text)
	text = tokenizer.decode(x.tolist()[0])
	print(text)
