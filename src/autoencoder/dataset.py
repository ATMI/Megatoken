from typing import List, Dict, Any, Tuple

import torch
from torch import LongTensor

import datasets
from torch.utils import data
from transformers import AutoTokenizer, PreTrainedTokenizer

tokenizer_: PreTrainedTokenizer | None = None


class AutoEncoderDataset(data.Dataset):
	MIN_LENGTH = 16
	MAX_LENGTH = 10_000

	@staticmethod
	def tokenize(
		batch: Dict[str, List[Any]],
		model_name: str,
	) -> Dict[str, List[Any]]:
		global tokenizer_
		tokenizer_ = tokenizer_ or AutoTokenizer.from_pretrained(model_name)

		kwargs = {
			"add_special_tokens": True,
			"truncation": True,
			"padding": False,
			"return_attention_mask": False,
		}

		article = tokenizer_(batch["article"], **kwargs)
		summary = tokenizer_(batch["highlights"], **kwargs)

		return {
			"article": article["input_ids"],
			"summary": summary["input_ids"],
		}

	def __init__(
		self,
		split: str,
		model_name: str,
		ign_token: int,
	):
		dataset = datasets.load_dataset(
			path="abisee/cnn_dailymail",
			name="3.0.0",
			split=split,
		)

		dataset = dataset.map(
			function=AutoEncoderDataset.tokenize,
			fn_kwargs={
				"model_name": model_name,
			},
			batched=True,
			remove_columns=list(dataset.column_names),
		)

		tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.bos_token = tokenizer.pad_token_id
		self.eos_token = tokenizer.eos_token_id
		self.ign_token = ign_token
		self.dataset = dataset

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index) -> Tuple[LongTensor, LongTensor, LongTensor]:
		sample = self.dataset[index]
		#	a		b		c		<eos>	input_ids
		#	<bos>	a		b		c		decoder_input_ids
		#	a		b		c		<eos>	labels = input_ids

		#	a		b		c		input_ids
		#	<bos>	a		b		decoder_input_ids
		#	a		b		c		labels = input_ids

		label_ids = sample["summary"]
		encoder_input_ids = sample["article"]
		decoder_input_ids = [self.bos_token] + label_ids[:-1]

		label_ids = torch.tensor(label_ids)
		encoder_input_ids = torch.tensor(encoder_input_ids)
		decoder_input_ids = torch.tensor(decoder_input_ids)

		return encoder_input_ids, decoder_input_ids, label_ids
