import argparse
from functools import partial
from pathlib import Path

from torch import nn, optim
from torch.utils import data
from transformers import get_scheduler

from pipeline.base.train import train
from scripts.dataset import prepare_dataset
from pipeline.mlm.batch import MaskModelBatch
from pipeline.mlm.checkpoint import MaskModelCheckpoint
from pipeline.mlm.log import MaskModelLog
from pipeline.mlm.model import MaskModel
from utils.config import load_config


def main():
	args = argparse.ArgumentParser()
	args.add_argument("config", type=Path)
	args.add_argument("output", type=Path)
	args = args.parse_args()

	# if args.output.exists():
	# 	print("Output file already exists. Aborting.")
	# 	exit(1)

	config = load_config(args.config)

	model = MaskModel(
		model=config.model,
		tokenizer=config.tokenizer,
	)

	log = MaskModelLog(
		directory=args.output,
	)

	checkpoint = MaskModelCheckpoint(
		directory=args.output,
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
