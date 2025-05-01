from typing import List, Dict, Any, Tuple

import datasets
import torch
from torch import LongTensor
from torch.utils import data
from transformers import PreTrainedTokenizer, AutoTokenizer

tokenizer: PreTrainedTokenizer | None = None


class SummarizerDataset(data.Dataset):

	@staticmethod
	def tokenize(
		batch: Dict[str, List[Any]],
		model_name: str,
	) -> Dict[str, List[Any]]:
		global tokenizer
		tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)

		kwargs = {
			"add_special_tokens": True,
			"truncation": True,
			"padding": False,
			"return_attention_mask": False,
		}

		article_tokens = tokenizer(batch["article"], **kwargs)["input_ids"]
		summary_tokens = tokenizer(batch["highlights"], **kwargs)["input_ids"]

		return {
			"article_tokens": article_tokens,
			"summary_tokens": summary_tokens,
		}

	def __init__(
		self,
		split: str,
		model_name: str,
		bos_token: int,
	):
		super().__init__()

		dataset = datasets.load_dataset(
			path="abisee/cnn_dailymail",
			name="3.0.0",
			split=split,
		)
		dataset = dataset.select_columns(["article", "highlights"])
		dataset = dataset.map(
			function=SummarizerDataset.tokenize,
			fn_kwargs={"model_name": model_name},
			batched=True,
		)
		dataset = dataset.select_columns(["article_tokens", "summary_tokens"])

		self.dataset = dataset
		self.bos_token = bos_token

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index) -> Tuple[LongTensor, LongTensor, LongTensor]:
		sample = self.dataset[index]

		article_tokens = sample["article_tokens"]
		summary_tokens = sample["summary_tokens"]
		decoder_tokens = [self.bos_token] + summary_tokens[:-1]

		article_tokens = torch.tensor(article_tokens)
		summary_tokens = torch.tensor(summary_tokens)
		decoder_tokens = torch.tensor(decoder_tokens)

		return article_tokens, decoder_tokens, summary_tokens
