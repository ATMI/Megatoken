import argparse
from pathlib import Path
from typing import Dict

import torch
from torch import nn, optim
from torch.utils import data
from transformers import get_scheduler

from model.transformer import GatedTransformer
from pipeline.batch import Batch
from pipeline.checkpoint import Checkpoint
from pipeline.log import Log
from pipeline.step import Step
from pipeline.train import train
from scripts.dataset import prepare_dataset
from utils.config import load_config


class MLMBatch(Batch):

	@property
	def x(self) -> Dict[str, any]:
		pass

	@property
	def y(self) -> Dict[str, any]:
		pass

	def __init__(self, batch):
		pass


class MLMCheckpoint(Checkpoint):

	def condition(self, step: Step) -> bool:
		return step.is_last or ...


class MLMLog(Log):

	def info(self, step: Step) -> Dict[str, any]:
		pass


class MLM(nn.Module):
	def __init__(self, model, tokenizer):
		super(MLM, self).__init__()

		self.transformer = GatedTransformer(model, tokenizer)
		self.classifier = nn.Sequential(
			nn.Dropout(model.classifier.dropout),
			nn.Linear(model.dim, tokenizer.vocab),
		)

	def forward(
		self,
		x: torch.Tensor,
		x_pad: torch.Tensor,

		y: torch.Tensor,
		y_pad: torch.Tensor,
	) -> any:
		y, y_pad, ratio = self.transformer(x, x_pad, y, y_pad)
		y = self.classifier(y)
		return y, y_pad, ratio


def main():
	args = argparse.ArgumentParser()
	args.add_argument("config", type=Path)
	args.add_argument("output", type=Path)
	args = args.parse_args()

	# if args.output.exists():
	# 	print("Output file already exists. Aborting.")
	# 	exit(1)

	config = load_config(args.config)

	model = MLM(
		model=config.model,
		tokenizer=config.tokenizer,
	)

	log = MLMLog(
		directory=args.output,
	)

	checkpoint = MLMCheckpoint(
		directory=args.output,
	)

	dataset = prepare_dataset(
		dataset=config.dataset.path,
		tokenizer=config.tokenizer.path,
		tokenized_col=config.dataset.text_col,
	)

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

	optimizer = optim.AdamW(
		params=model.parameters(),
		lr=config.train.lr
	)

	scheduler = get_scheduler(
		name=config.train.scheduler,
		optimizer=optimizer,
		num_warmup_steps=int(config.train.warmup * len(train_loader)),
		num_training_steps=config.train.epochs * len(train_loader),
	)

	train(
		epochs=config.train.epochs,
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
