import argparse
from functools import partial
from pathlib import Path

import torch
from torch import optim
from torch.utils import data
from transformers import get_scheduler

from pipeline.base.train import train
from pipeline.mlm.batch import MaskModelBatch
from pipeline.mlm.checkpoint import MaskModelCheckpoint
from pipeline.mlm.log import MaskModelLog
from pipeline.mlm.loss import MaskModelLoss
from pipeline.mlm.model import MaskModel
from scripts.dataset import prepare_dataset
from utils.config import load_config


def main():
	torch.random.manual_seed(42)

	args = argparse.ArgumentParser()
	args.add_argument("config", type=Path)
	args = args.parse_args()

	# if args.output.exists():
	# 	print("Output file already exists. Aborting.")
	# 	exit(1)

	config = load_config(args.config)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = MaskModel(
		model=config.model,
		tokenizer=config.tokenizer,
	)

	log = MaskModelLog(
		directory=config.log.output,
		k=config.log.k,
	)

	checkpoint = MaskModelCheckpoint(
		directory=config.checkpoint.output,
		time_interval=config.checkpoint.interval,
	)

	dataset = prepare_dataset(
		dataset=config.dataset.path,
		tokenizer=config.tokenizer.path,
		tokenized_col=config.dataset.text_col,
	)

	batch_factory = partial(
		MaskModelBatch,
		pad_token=config.tokenizer.pad,
		mask_token=config.tokenizer.mask,
		ignore_token=-100,
		prob=0.15,
		causal=False,
	)

	train_loader = data.DataLoader(
		dataset=dataset["train"],
		batch_size=config.train.batch,
		shuffle=True,
		collate_fn=batch_factory,
	)

	test_loader = data.DataLoader(
		dataset=dataset["test"],
		batch_size=config.test.batch,
		shuffle=False,
		collate_fn=batch_factory,
	)

	criterion = MaskModelLoss()

	optimizer = optim.Adam(
		params=model.parameters(),
		lr=config.train.lr,
	)

	scheduler = get_scheduler(
		name=config.train.scheduler,
		optimizer=optimizer,
		num_warmup_steps=int(config.train.warmup * len(train_loader)),
		num_training_steps=config.train.epochs * len(train_loader),
	)

	train(
		epochs=config.train.epochs,
		device=device,
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
