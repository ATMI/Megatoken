from functools import partial
from typing import List, Dict, Any

import datasets
import torch
from torch.utils import data
from transformers import AutoTokenizer


class AutoEncoderDataset(data.Dataset):
	@staticmethod
	def tokenize(
		tokenizer: str,
		text_column: str,

		batch: Dict[str, List[Any]],
	) -> Dict[str, List[Any]]:
		tokenizer = AutoTokenizer.from_pretrained(tokenizer)

		batch = batch[text_column]
		batch = tokenizer(
			text=batch,
			padding=False,
			truncation=False,
			add_special_tokens=False,
			return_attention_mask=False,
		)

		max_length = tokenizer.model_max_length
		result = []

		for row in batch:
			for i in range(len(row), max_length):
				result.append(row[i: i + max_length])

		return {
			"tokens": result,
		}

	def __init__(
		self,
		name: str,
		version: str,
		split: str,

		tokenizer: str,
		text_column: str,
	):
		dataset = datasets.load_dataset(name, version, split=split)
		columns = list(dataset.column_names)
		dataset = dataset.map(
			function=partial(
				AutoEncoderDataset.tokenize,
				tokenizer=tokenizer,
				text_column=text_column,
			),
			batched=True,
			with_indices=True,
			remove_columns=columns,
		)
		self.dataset = dataset

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		sample = self.dataset[index]
		sample = sample["tokens"]
		sample = torch.tensor(sample)
		return sample
