import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from autoencoder import AutoEncoder

if __name__ == "__main__":
	torch.random.manual_seed(42)
	random.seed(42)

	ckpt_path = Path("checkpoint", "0226_0815", "2_6.pth")
	ckpt = torch.load(ckpt_path, weights_only=True)

	tokenizer_name = "google-bert/bert-base-uncased"
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = AutoEncoder(
		vocab_size=tokenizer.vocab_size,
		pad_idx=tokenizer.pad_token_id,

		model_dim=512,
		max_len=512,

		encoder_head_num=4,
		decoder_head_num=4,

		encoder_layer_num=4,
		decoder_layer_num=4,

		encoder_fc_dim=2048,
		decoder_fc_dim=2048,
	)
	model.load_state_dict(ckpt["model"])
	model.eval()

	params = sum(p.numel() for p in model.parameters())
	print(params)

	model.to(device)

	# text = "Good. Went yesterday with family and enjoyed the oysters"
	text = "Love this place soooo much, never ever had a bad meal, big tasty portions and great friendly people! Be ready to stand in line."
	temp = 0.15

	x = tokenizer(
		[text],
		return_tensors="pt",
		return_token_type_ids=False,
		return_attention_mask=False,
	)["input_ids"]
	x = x[:, :-1]

	x = x.to(device)
	y = model.embedding(x)
	y = model.positional(y)
	y = model.encoder(y, None)
	y = y.unsqueeze(1)

	x = model.positional(y)
	t = []

	with torch.no_grad():
		for i in range(100):

			y = model.decoder(x, None)
			y = y[:, -1, :]
			y = model.classifier(y)

			y = y / temp
			y = y.softmax(dim=-1)
			y = torch.multinomial(y, num_samples=1)

			i_id = y.item()
			if i_id == 102:
				break
			t.append(i_id)

			y = model.embedding(y)
			y = model.positional(y)
			x = torch.cat((x, y), dim=1)


	print(text)
	text = tokenizer.decode(t)
	print(text)
