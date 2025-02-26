import random
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from torch import nn, optim
from torch.nn.utils import rnn
from torch.utils import data
from tqdm import tqdm
from transformers import AutoTokenizer

from autoencoder import AutoEncoder
from dataset import yelp_dataset


def collate(batch, pad):
	x = [torch.tensor(x["tokens"]) for x in batch]
	x = rnn.pad_sequence(x, batch_first=True, padding_value=pad)

	batch_size = x.size(0)
	seq_len = x.size(1)

	x_pad = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
	x_len = torch.zeros(batch_size, dtype=torch.int)

	for i in range(batch_size):
		tokens = batch[i]["tokens"]
		length = len(tokens)
		x_pad[i, :length] = False
		x_len[i] = length

	return x, x_pad, x_len


def epoch_pass(
	epoch: int,

	device: torch.device,
	model: nn.Module,
	criterion: nn.Module,
	loader: data.DataLoader,

	optimizer: optim.Optimizer | None = None,
	ckpt_dir: Path = None,
	ckpt_freq: int = 10,
):
	test = optimizer is None
	if test:
		model.eval()
		torch.set_grad_enabled(False)
	else:
		model.train()
		torch.set_grad_enabled(True)

	epoch_loss = 0
	loader_len = len(loader)

	ckpt_n = 1
	ckpt_freq = loader_len / ckpt_freq

	bar = tqdm(
		desc=f"{'Test ' if test else 'Train'} [{epoch}]",
		total=loader_len,
	)

	for i, (x, x_pad, x_len) in enumerate(loader):
		x = x.to(device)
		x_pad = x_pad.to(device)

		if optimizer is not None:
			optimizer.zero_grad()

		y = model(x, x_pad)
		y = torch.cat([yy[0:ll - 1] for yy, ll in zip(y, x_len)])
		x = torch.cat([xx[1:ll] for xx, ll in zip(x, x_len)])

		loss = criterion(y, x)
		if optimizer is not None:
			loss.backward()
			optimizer.step()
		torch.cuda.empty_cache()

		loss = loss.item()
		epoch_loss += loss
		avg_loss = epoch_loss / (i + 1)

		if (ckpt_dir is not None) and ((i + 1) >= round(ckpt_freq * ckpt_n)):
			ckpt = {
				"step": i,
				"epoch": epoch,
				"model": model.state_dict(),
				"optim": optimizer.state_dict(),
			}
			torch.save(ckpt, ckpt_dir / f"{epoch}_{ckpt_n}.pth")
			ckpt_n += 1

		bar.set_postfix(
			avg=f"{avg_loss:.4f}",
			cur=f"{loss:.4f}",
		)
		bar.update(1)
	bar.close()


if __name__ == "__main__":
	print("Hello, World!")

	ckpt_dir = datetime.now().strftime("%m%d_%H%M")
	ckpt_dir = Path("checkpoint", ckpt_dir)
	ckpt_dir.mkdir(parents=True, exist_ok=False)

	torch.random.manual_seed(42)
	random.seed(42)

	tokenizer_name = "google-bert/bert-base-uncased"
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	pad = tokenizer.pad_token_id

	dataset = yelp_dataset(tokenizer_name)
	train_loader = data.DataLoader(
		dataset["train"],
		batch_size=28,
		shuffle=True,
		collate_fn=partial(collate, pad=pad),
	)
	test_loader = data.DataLoader(
		dataset["test"],
		batch_size=32,
		collate_fn=partial(collate, pad=pad),
	)

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
	params = sum(p.numel() for p in model.parameters())
	print("Prams", params, "Vocab", tokenizer.vocab_size)
	model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0001)

	epochs = 10
	ckpt_freq = 10

	for i in range(epochs):
		epoch_pass(i, device, model, criterion, train_loader, optimizer, ckpt_dir, ckpt_freq)
		epoch_pass(i, device, model, criterion, test_loader)
