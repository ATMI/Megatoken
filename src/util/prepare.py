import random
from pathlib import Path

import datasets
import torch
from transformers import AutoTokenizer


# TODO: do something with this file :)

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
