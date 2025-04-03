from pathlib import Path
from typing import Tuple

import datasets
from torch.utils import data
from transformers import AutoTokenizer

from batch import Batch
from config import Config
from model import Model


def dataset():
	path = Path("dataset")
	if path.exists():
		return datasets.load_from_disk(path)

	def tokenize(batch):
		tokenizer = AutoTokenizer.from_pretrained(Config.model)
		tokens = tokenizer(
			batch["text"],
			padding=False,
			truncation=True,
			max_length=Config.max_length,
			return_attention_mask=False,
		)
		tokens = tokens["input_ids"]
		return {"tokens": tokens}

	ds = datasets.load_dataset(Config.dataset)
	ds = ds.map(tokenize, batched=True, num_proc=4)
	# ds = ds.select_columns(["tokens"])
	ds.save_to_disk(path)

	return ds


def dataloaders() -> Tuple[
	data.DataLoader[Batch],
	data.DataLoader[Batch]
]:
	ds = dataset()
	train_loader = data.DataLoader(
		dataset=ds["train"],
		batch_size=Config.batch_size,
		shuffle=True,
		collate_fn=Batch.collate,
	)
	test_loader = data.DataLoader(
		dataset=ds["test"],
		batch_size=Config.batch_size,
		shuffle=False,
		collate_fn=Batch.collate,
	)

	return train_loader, test_loader


def model() -> Model:
	return Model(Config.model, Config.bias, Config.temperature)
