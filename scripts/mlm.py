import argparse
from pathlib import Path
from typing import Dict

import torch
from torch import nn, optim
from torch.utils import data
from transformers import get_scheduler
from torch.nn.utils import rnn

from pipeline.batch import Batch
from pipeline.checkpoint import Checkpoint
from pipeline.log import Log
from pipeline.step import Step
from pipeline.train import train
from utils.config import load_config


class MLMBatch(Batch):

	def __init__(
		self,
		batch,
		pad: int,
		mask_token_id: int,
		causal_mask: bool,
		mask_prob: float = 0.15,
		ignore_index: int = -100,
	):
		self.x, self.pad_mask = self.collate(batch)

		self.pad_index = pad
		self.mask_token_id = mask_token_id
		self.ignore_index = ignore_index
		self.mask_prob = mask_prob
		self.causal_mask = causal_mask

		self.masked_x = None
		self.label = None
		self.attn_mask = None

	def collate(self, batch):
		x = [torch.tensor(x["tokens"]) for x in batch]
		x = rnn.pad_sequence(x, batch_first=True, padding_value=self.pad_index)

		batch_size = x.size(0)
		seq_len = x.size(1)

		x_pad = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
		for i in range(batch_size):
			tokens = batch[i]["tokens"]
			length = len(tokens)
			x_pad[i, :length] = False

		return x, x_pad

	# TODO: make wider masks
	# TODO: make mode in masking
	def masking(self):
		x_lens = self.pad_mask.size(1) - self.pad_mask.sum(dim=1)
		num_to_mask = (self.mask_prob * x_lens).int().clamp(min=1)

		mask = torch.zeros_like(self.x, dtype=torch.bool)

		for i in range(self.x.size(0)):
			candidates = torch.where(~self.pad_mask[i])[0]
			perm = torch.randperm(x_lens[i])
			selected = candidates[perm[:num_to_mask[i]]]
			mask[i, selected] = True

		masked_input = self.x.clone()
		masked_input[mask] = self.mask_token_id

		labels = torch.full_like(self.x, fill_value=self.ignore_index)  # -100 = ignore index for CrossEntropyLoss
		labels[mask] = self.x[mask]

		return masked_input, labels

	def x(self) -> Dict[str, any]:
		self.masked_x, self.label = self.masking()

		attn_mask = None
		if self.causal_mask:
			attn_mask = nn.Transformer.generate_square_subsequent_mask(self.x.size(1), dtype=torch.bool)

		inputs = {
			"x": self.masked_x,
			"pad": self.pad_mask,
			"attn": attn_mask
		}

		return inputs

	def y(self) -> torch.Tensor:
		"""
		:return: RETURN TENSOR FOR CROSS-ENTROPY-LOSS
		"""
		flat_label = self.label.flatten()
		return flat_label


class MLMCheckpoint(Checkpoint):

	def condition(self, step: Step) -> bool:
		return step.is_last or ...


class MLMLog(Log):

	def info(self, step: Step) -> Dict[str, any]:
		pass


def main():
	args = argparse.ArgumentParser()
	args.add_argument("config", type=Path, required=True)
	args.add_argument("output", type=Path, required=True)
	args = args.parse_args()

	if args.output.exists():
		print("Output file already exists. Aborting.")
		exit(1)

	config = load_config(args.config)

	dataset = {}
	model = nn.Module()

	train_loader = data.DataLoader(
		dataset=dataset["train"],
		batch_size=config.train.batch,
		shuffle=True,
		collate_fn=MLMBatch,
	)

	test_loader = data.DataLoader(
		dataset=dataset["test"],
		batch_size=config.test.batch,
		shuffle=False,
		collate_fn=MLMBatch,
	)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=config.lr)
	scheduler = get_scheduler(
		name=config.scheduler,
		optimizer=optimizer,
		num_warmup_steps=int(config.train.warmup * len(train_loader)),
		num_training_steps=config.epochs * len(train_loader),
	)

	log = MLMLog(args.output)
	checkpoint = MLMCheckpoint(args.output)

	train(
		epochs=config.epochs,
		model=model,

		criterion=criterion,
		optimizer=optimizer,
		scheduler=scheduler,

		train_loader=train_loader,
		test_loader=test_loader,

		log=log,
		checkpoint=checkpoint,
	)


if __name__ == "__main__":
	main()
