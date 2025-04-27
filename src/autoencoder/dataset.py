from functools import partial
from typing import List, Dict, Any

import datasets
import torch
from torch.utils import data
from transformers import AutoTokenizer


class AutoEncoderDataset(data.Dataset):
	MIN_LENGTH = 16
	MAX_LENGTH = 10_000

	@staticmethod
	def tokenize(
		tokenizer: str,
		text_column: str,

		batch: Dict[str, List[Any]],
	) -> Dict[str, List[Any]]:
		tokenizer = AutoTokenizer.from_pretrained(tokenizer)

		batch = batch[text_column]
		batch = [sample.lower() for sample in batch]
		batch = tokenizer(
			text=batch,
			padding=False,
			truncation=True,
			max_length=AutoEncoderDataset.MAX_LENGTH,
			add_special_tokens=True,
			return_attention_mask=False,
		)

		max_length = tokenizer.model_max_length
		batch = batch["input_ids"]
		result = []

		for row in batch:
			for start in range(0, len(row), max_length):
				chunk = row[start: start + max_length]
				if len(chunk) < AutoEncoderDataset.MIN_LENGTH:
					continue
				result.append(chunk)

		return {
			"tokens": result,
		}

	def __init__(
		self,
		name: str,
		version: str,
		split: str,

		tokenizer: str,
		ign_token: int,
		text_column: str,
	):
		dataset = datasets.load_dataset(name, version, split=split)
		columns = list(dataset.column_names)
		dataset = dataset.map(
			function=partial(AutoEncoderDataset.tokenize, tokenizer, text_column),
			batched=True,
			remove_columns=columns,
		)

		tokenizer = AutoTokenizer.from_pretrained(tokenizer)
		self.bos_token = tokenizer.pad_token_id
		self.eos_token = tokenizer.eos_token_id
		self.ign_token = ign_token
		self.dataset = dataset

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		sample = self.dataset[index]
		#	a		b		c		<eos>	input_ids
		#	<bos>	a		b		c		decoder_input_ids
		#	a		b		c		<eos>	labels

		#	a		b		c		input_ids
		#	<bos>	a		b		decoder_input_ids
		#	a		b		c		labels

		input_ids = sample["tokens"]
		decoder_input_ids = [self.bos_token] + input_ids[:-1]

		input_ids = torch.tensor(input_ids)
		decoder_input_ids = torch.tensor(decoder_input_ids)

		return input_ids, decoder_input_ids
