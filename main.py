import random
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List

import torch
from torch import nn, optim
from torch.nn.utils import rnn
from torch.utils import data
from tqdm import tqdm
from transformers import AutoTokenizer

from classifier import Classifier
from dataset import yelp_dataset


def collate(batch, pad):
	# y = torch.tensor([x["label"] >= 3 for x in batch], dtype=torch.float)
	y = torch.tensor([x["label"] for x in batch])
	x = [torch.tensor(x["tokens"]) for x in batch]
	x = rnn.pad_sequence(x, batch_first=True, padding_value=pad)

	batch_size = x.size(0)
	seq_len = x.size(1)

	x_pad = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
	for i in range(batch_size):
		tokens = batch[i]["tokens"]
		length = len(tokens)
		x_pad[i, :length] = False

	return x, x_pad, y


def ckpt_save(
	path: Path,
	model: nn.Module,
	optimizer: optim.Optimizer,
):
	ckpt = {
		"model": model.state_dict(),
		"optim": optimizer.state_dict(),
	}
	torch.save(ckpt, path)


def loss_save(path: Path, loss: List[float]):
	with path.open("w") as f:
		for i in loss:
			f.write(f"{i}\n")


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

	loader_len = len(loader)
	loss_history = [0] * loader_len
	loss_total = 0

	ckpt_freq = loader_len / ckpt_freq
	ckpt_n = 0

	desc = "test" if test else "train"
	if ckpt_dir:
		ckpt_dir = ckpt_dir / desc / str(epoch)
		ckpt_dir.mkdir(parents=True)

	bar = tqdm(desc=f"{desc} {epoch}", total=loader_len)
	pred_correct = 0
	pred_total = 0

	for i, (x, x_pad, y) in enumerate(loader):
		x = x.to(device)
		y = y.to(device)
		x_pad = x_pad.to(device)

		if optimizer is not None:
			optimizer.zero_grad()

		y_pred, com_len = model(x, x_pad)
		# y_pred = y_pred.squeeze(-1)
		loss = criterion(y_pred, y)

		if optimizer is not None:
			loss.backward()
			optimizer.step()
		torch.cuda.empty_cache()

		loss = loss.item()
		loss_history.append(loss)
		loss_total += loss
		loss_avg = loss_total / (i + 1)

		if ckpt_dir and (i + 1) >= round(ckpt_freq * (ckpt_n + 1)):
			if not test:
				ckpt_path = ckpt_dir / f"{ckpt_n}.pth"
				ckpt_save(ckpt_path, model, optimizer)

			loss_path = ckpt_dir / f"{epoch}.loss"
			loss_save(loss_path, loss_history)
			ckpt_n += 1

		pred_total += y_pred.size(0)
		# pred_correct += y_pred.sigmoid().ge(0.5).eq(y).sum().item()
		pred_correct += y_pred.argmax(dim=1).eq(y).sum().item()
		acc = pred_correct / pred_total

		seq_len = x.size(1)
		com = com_len / seq_len

		bar.set_postfix(
			avg_acc=f"{acc * 100:.2f}",
			avg_los=f"{loss_avg:.4f}",
			los=f"{loss:.4f}",
			com=f"{com * 100:.2f}",
		)
		bar.update(1)
	bar.close()


def main():
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
		batch_size=128,
		shuffle=True,
		collate_fn=partial(collate, pad=pad),
	)
	test_loader = data.DataLoader(
		dataset["test"],
		batch_size=256,
		collate_fn=partial(collate, pad=pad),
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Classifier(
		vocab_size=tokenizer.vocab_size,
		embed_dim=64,
		pad_idx=tokenizer.pad_token_id,

		encoder_layer_thresh=[0.75, 0.95, 1.00],
		encoder_heads_num=1,
		encoder_fc_dim=128,

		class_num=5,
	)

	params = sum(p.numel() for p in model.parameters())
	print("Prams", params, "Vocab", tokenizer.vocab_size)
	model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	epochs = 10
	ckpt_freq = 10

	for i in range(epochs):
		epoch_pass(i, device, model, criterion, train_loader, optimizer, ckpt_dir, ckpt_freq)
		epoch_pass(i, device, model, criterion, test_loader)


if __name__ == "__main__":
	main()
