import argparse
from pathlib import Path
from typing import Dict

from torch import nn, optim
from torch.utils import data
from transformers import get_scheduler

from pipeline.batch import Batch
from pipeline.checkpoint import Checkpoint
from pipeline.epoch import Step
from pipeline.log import Log
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
