import csv
import random
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn, optim
from torch.nn.utils import rnn
from torch.utils import data
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import prepare_dataset
from encoder import SoftGate
from model.model import coBERT
from utils.config import load_config


# torch.autograd.set_detect_anomaly(True)


def masking(
		x: torch.Tensor,
		pad_mask: torch.Tensor,
		mask_token_id: int,
		mask_prob: float = 0.15,
		device: torch.device = torch.cpu,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Randomly mask tokens in the sequence.

	Note: Initial sequence must not contain any special tokens except pad_token.
	:param x: Initial sequence, [batch_size, seq_len]
	:param pad_mask: Padding mask, [batch_size, seq_len]. True = pad token
	:param mask_token_id: Mask token id
	:param mask_prob: Percentage of tokens to mask
	:param device: Device
	:return: Masked inputs, labels
	"""
	x_lens = pad_mask.size(1) - pad_mask.sum(dim=1)
	num_to_mask = (mask_prob * x_lens).int().clamp(min=1)

	mask = torch.zeros_like(x, dtype=torch.bool)

	for i in range(x.size(0)):
		candidates = torch.where(~pad_mask[i])[0]
		perm = torch.randperm(x_lens[i])
		selected = candidates[perm[:num_to_mask[i]]]
		mask[i, selected] = True

	masked_input = x.clone()
	masked_input[mask] = mask_token_id

	labels = torch.full_like(x, fill_value=-100, device=device)  # -100 = ignore index for CrossEntropyLoss
	labels[mask] = x[mask]

	return masked_input, labels


def collate(batch, pad):
	x = [torch.tensor(x["tokens"]) for x in batch]
	x = rnn.pad_sequence(x, batch_first=True, padding_value=pad)

	batch_size = x.size(0)
	seq_len = x.size(1)

	x_pad = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
	for i in range(batch_size):
		tokens = batch[i]["tokens"]
		length = len(tokens)
		x_pad[i, :length] = False

	return x, x_pad


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


def log_save(path: Path, log: List):
	with path.open("w") as file:
		header = log[0].keys()
		writer = csv.DictWriter(file, fieldnames=header, delimiter="\t")
		writer.writeheader()
		writer.writerows(log)


def epoch_pass(
		epoch: int,

		device: torch.device,
		model: nn.Module,
		criterion: nn.Module,
		loader: data.DataLoader,
		mask_token_id: int,

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
	loss_total = 0
	log = []

	ckpt_freq = loader_len / ckpt_freq
	ckpt_n = 0

	desc = "test" if test else "train"
	if ckpt_dir:
		ckpt_dir = ckpt_dir / desc / str(epoch)
		ckpt_dir.mkdir(parents=True)

	bar = tqdm(desc=f"{desc} {epoch}", total=loader_len)
	correct_total = 0
	pred_total = 0

	for i, (x, x_pad) in enumerate(loader):
		x = x.to(device)
		x_pad = x_pad.to(device)

		x, y = masking(x, x_pad, mask_token_id, device=device)
		attn_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), dtype=torch.bool)

		if optimizer is not None:
			optimizer.zero_grad()

		y_pred, ratio = model(x, x_pad, attn_mask)
		print("Y_pred:", y_pred.shape)
		loss = criterion(y_pred.flatten(start_dim=0, end_dim=1), y.flatten(start_dim=0, end_dim=1))

		if optimizer is not None:
			loss.backward()
			optimizer.step()
		torch.cuda.empty_cache()

		loss = loss.item()
		loss_total += loss
		loss_avg = loss_total / (i + 1)

		pred = y_pred.argmax(dim=-1)
		correct = (pred == y).sum().item()
		num_labels = (y != -100).sum().item()  # Number of tokens to predict

		pred_total += num_labels
		correct_total += correct

		acc = correct / num_labels
		acc_avg = correct_total / pred_total

		log_ent = {
			"ratio": ratio,
			"acc": acc * 100,
			"acc_": acc_avg * 100,
			"loss": loss,
			"loss_": loss_avg,
		}
		log.append(log_ent)

		bar.set_postfix(**log_ent)
		bar.update(1)

		if ckpt_dir and (i + 1) >= round(ckpt_freq * (ckpt_n + 1)):
			if not test:
				ckpt_path = ckpt_dir / f"{ckpt_n}.pth"
				ckpt_save(ckpt_path, model, optimizer)

			log_path = ckpt_dir / f"{epoch}.tsv"
			log_save(log_path, log)
			ckpt_n += 1
	bar.close()


def main():
	print("Hello, World!")

	ckpt_dir = datetime.now().strftime("%m%d_%H%M")
	ckpt_dir = Path("checkpoint", ckpt_dir)
	ckpt_dir.mkdir(parents=True, exist_ok=False)

	torch.random.manual_seed(42)
	random.seed(42)

	dataset_name = "Yelp/yelp_review_full"
	tokenizer_name = "google-bert/bert-base-uncased"

	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	pad = tokenizer.pad_token_id

	dataset = prepare_dataset(
		dataset=dataset_name,
		tokenizer=tokenizer_name,
		tokenized_col="text",
	)

	train_loader = data.DataLoader(
		dataset["train"],
		batch_size=32,
		shuffle=True,
		collate_fn=partial(collate, pad=pad),
	)
	test_loader = data.DataLoader(
		dataset["test"],
		batch_size=256,
		collate_fn=partial(collate, pad=pad),
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model_cfg = load_config("configs/model_config.yaml")
	model = coBERT(
		cfg=model_cfg,
		c_gate=SoftGate,
		vocab_size=tokenizer.vocab_size,
		pad_idx=1,
	)

	params = sum(p.numel() for p in model.parameters())
	print("Prams", params, "Vocab", tokenizer.vocab_size)
	model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	epochs = 10
	ckpt_freq = 10

	mask_token_id = tokenizer.mask_token_id
	for i in range(epochs):
		epoch_pass(i, device, model, criterion, train_loader, mask_token_id, optimizer, ckpt_dir, ckpt_freq)
		epoch_pass(i, device, model, criterion, test_loader, mask_token_id)


if __name__ == "__main__":
	main()