import json
import signal
from dataclasses import dataclass

import torch
from torch import optim, Tensor
from torch.nn import functional as fn
from torch.nn.utils import rnn
from tqdm import tqdm

import prepare
from config import Config


class RollingMean:
	def __init__(self, n: int):
		self.n = n
		self.values = []
		self.sum = 0

	def __call__(self, value: float) -> float:
		self.sum += value
		self.values.append(value)

		if len(self.values) > self.n:
			value = self.values.pop(0)
			self.sum -= value

		mean = self.sum / len(self.values)
		return mean


@dataclass
class Batch:
	tokens: Tensor
	sparse: Tensor
	labels: Tensor
	pad_mask: Tensor

	def to(self, device) -> "Batch":
		return Batch(
			self.tokens.to(device),
			self.sparse.to(device),
			self.labels.to(device),
			self.pad_mask.to(device),
		)


def collate_batch(batch) -> Batch:
	batch = [row["tokens"] for row in batch]
	tokens = [torch.tensor(row) for row in batch]
	tokens = rnn.pad_sequence(tokens, batch_first=True, padding_value=Config.pad_token)

	sparse = tokens.clone()
	labels = torch.full_like(tokens, Config.ignore_token)
	pad_mask = torch.ones_like(tokens, dtype=torch.bool)

	for i, row in enumerate(batch):
		length = len(row)
		pad_mask[i, :length] = False

		sparsity = max(1, int(length * Config.sparsity))
		indices = torch.randperm(length)
		indices = indices[:sparsity]

		sparse[i, indices] = Config.mask_token
		labels[i, indices] = tokens[i, indices]

	return Batch(tokens, sparse, labels, pad_mask)


def accuracy(filled: Tensor, target: Tensor) -> float:
	mask = target.ne(Config.ignore_token)
	pred = filled[mask].argmax(dim=1)
	true = target[mask]

	corr = pred.eq(true)
	corr = corr.sum()
	total = true.numel()

	acc = corr / total
	acc = acc.item()

	return acc


def prompt(text: str) -> bool:
	while True:
		response = input(text).strip().lower()
		if response in ["yes", "y"]:
			return True
		elif response in ["no", "n"]:
			return False
		else:
			print("Please answer with 'yes' or 'no'.")


def main():
	torch.manual_seed(Config.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_loader, test_loader = prepare.dataloaders(collate_batch)

	encoder = prepare.encoder().to(device)
	decoder = prepare.decoder().to(device)

	params = list(encoder.parameters()) + list(decoder.parameters())
	optimizer = optim.Adam(params, Config.lr)

	step_num = len(train_loader)
	bar = tqdm(total=step_num)

	log_file = open("log.json", "w")
	roll_acc = RollingMean(Config.rolling_n)
	roll_loss = RollingMean(Config.rolling_n)

	def finish():
		log_file.close()
		bar.close()
		exit(0)

	def interrupt(sig, frame):
		if not prompt("Interrupt? "):
			return

		if prompt("Checkpoint? "):
			checkpoint = {
				"encoder": encoder.state_dict(),
				"decoder": decoder.state_dict(),
				"optimizer": optimizer.state_dict(),
			}
			torch.save(checkpoint, "checkpoint.pth")
		finish()

	signal.signal(signal.SIGINT, interrupt)
	for batch in train_loader:
		optimizer.zero_grad()

		batch = batch.to(device)
		dense = encoder(batch.tokens, batch.pad_mask, None)
		filled = decoder(
			batch.sparse, batch.pad_mask,
			dense.embeds, dense.pad_mask, dense.attn_mask,
		)

		cls_loss = fn.cross_entropy(filled.flatten(0, 1), batch.labels.flatten())
		loss = cls_loss

		loss.backward()
		for param in params:
			if param.isnan().any():
				raise RuntimeError()
		optimizer.step()
		torch.cuda.empty_cache()

		acc = accuracy(filled, batch.labels) * 100
		loss = loss.item()

		mean_acc = roll_acc(acc)
		mean_loss = roll_loss(loss)

		log = {
			"acc": f"{acc:.2f}",
			"acc~": f"{mean_acc:.2f}",
			"loss": f"{loss:.4f}",
			"loss~": f"{mean_loss:.4f}",
		}
		log_file.write(json.dumps(log) + "\n")
		log_file.flush()

		postfix = {
			"acc": f"{mean_acc:.2f}",
			"loss": f"{mean_loss:.4f}",
		}
		bar.set_postfix(**postfix)
		bar.update(1)
	finish()


if __name__ == "__main__":
	main()
