import argparse
from pathlib import Path
from typing import Dict

import torch
from torch import nn, optim
from torch.utils import data
from transformers import get_scheduler

from model.transformer import Transformer
from pipeline.batch import Batch
from pipeline.checkpoint import Checkpoint
from pipeline.log import Log
from pipeline.step import Step
from pipeline.train import train
from utils.config import load_config


class MLMBatch(Batch):

	def __init__(self, batch):
		pass

	def x(self) -> Dict[str, any]:
		pass

	def y(self) -> Dict[str, any]:
		pass


class MLMCheckpoint(Checkpoint):

	def condition(self, step: Step) -> bool:
		return step.is_last or ...


class MLMLog(Log):

	def info(self, step: Step) -> Dict[str, any]:
		pass


class MLM(nn.Module):
	def __init__(self, config):
		super(MLM, self).__init__()

		self.transformer = Transformer.from_config(config)
		self.classifier = nn.Sequential(
			nn.Dropout(config.classifier.dropout),
			nn.Linear(config.dim, config.vocab_size),
		)

	def forward(
		self,
		x: torch.Tensor,
		x_pad: torch.Tensor,

		y: torch.Tensor,
		y_pad: torch.Tensor,
	) -> any:
		y = self.transformer(x, x_pad, y, y_pad)
		y = self.classifier(y)
		return y


def main():
	args = argparse.ArgumentParser()
	args.add_argument("config", type=Path)
	args.add_argument("output", type=Path)
	args = args.parse_args()

	# if args.output.exists():
	# 	print("Output file already exists. Aborting.")
	# 	exit(1)

	config = load_config(args.config)
	############################################################################
	model = MLM(config.model)
	batch = 3
	seq = 10

	x = torch.randint(
		low=0,
		high=config.model.vocab_size,
		size=(batch, seq),
	)
	x_pad = torch.tensor(
		[
			[0] * (seq - i) + [1] * i
			for i in range(batch)
		],
		dtype=torch.bool,
	)

	y = model(x, x_pad, x, x_pad)
	############################################################################

	dataset = {}
	model = MLM(config)

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
