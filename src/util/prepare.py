import random
from pathlib import Path
from typing import Tuple, Callable, List, Any

import datasets
import torch

from torch.utils import data
from transformers import AutoTokenizer


def rnd(seed: int):
	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)


def dataset():
	path = Path("dataset")
	if not path.exists():
		raise FileNotFoundError(f"Dataset not found at {path}")
	return datasets.load_from_disk(path)


def dataloaders(
	batch_size: int,
	collate: Callable[[List], Any],
) -> Tuple[
	data.DataLoader,
	data.DataLoader
]:
	ds = dataset()
	train_loader = data.DataLoader(
		dataset=ds["train"],
		batch_size=batch_size,
		collate_fn=collate,
	)
	test_loader = data.DataLoader(
		dataset=ds["test"],
		batch_size=batch_size,
		shuffle=False,
		collate_fn=collate,
	)
	return train_loader, test_loader


def main() -> None:
	from src.autoencoder.config import Config

	def tokenize(batch, idx):
		tokenizer = AutoTokenizer.from_pretrained(Config.model)
		tokens = tokenizer(
			batch["text"],
			padding=False,
			truncation=True,
			max_length=Config.max_length,
			return_attention_mask=False,
		)
		tokens = tokens["input_ids"]
		return {
			"id": idx,
			"tokens": tokens,
		}

	ds = datasets.load_dataset(Config.dataset)
	ds = ds.map(tokenize, with_indices=True, batched=True, num_proc=4)
	ds.save_to_disk("dataset")


if __name__ == "__main__":
	main()
