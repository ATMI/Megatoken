from functools import partial
from typing import List, Dict, Any

import datasets
import torch
from torch.utils import data
from transformers import AutoTokenizer, PreTrainedTokenizer

tokenizer: PreTrainedTokenizer | None = None

class ClassifierDataset(data.Dataset):
	@staticmethod
	def tokenize(
		model_name: str,
		text_column: str,

		batch: Dict[str, List[Any]],
	) -> Dict[str, List[Any]]:
		global tokenizer
		tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)

		batch = batch[text_column]
		batch = tokenizer(
			text=batch,
			padding=False,
			truncation=True,
			add_special_tokens=True,
			return_attention_mask=False,
		)

		return {
			"input_ids": batch["input_ids"],
		}

	def __init__(
		self,
		name: str,
		version: str | None,
		split: str,

		model_name: str,
		text_column: str,
		label_column: str,
	):
		dataset = datasets.load_dataset(name, version, split=split)
		columns = list(col for col in dataset.column_names if col != label_column)
		dataset = dataset.map(
			function=partial(ClassifierDataset.tokenize, model_name, text_column),
			batched=True,
			remove_columns=columns,
		)

		self.label_column = label_column
		self.dataset = dataset

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		sample = self.dataset[index]

		input_ids = torch.tensor(sample["input_ids"])
		label = sample[self.label_column]

		return input_ids, label
