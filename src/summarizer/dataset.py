from functools import partial
from typing import List, Dict, Any, Tuple

import datasets
import torch
from torch.utils import data
from transformers import AutoTokenizer


class SummarizerDataset(data.Dataset):
	def __init__(
		self,
		dataset: datasets.Dataset,
		bos_token: int,
	):
		super().__init__()
		self.dataset = dataset
		self.bos_token = bos_token

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		sample = self.dataset[index]

		# article = sample["article"]
		# labels_str = sample["highlights"]
		labels = sample["highlights_token"]  # TODO: highlights_tokens

		input_tokens = [self.bos_token] + labels[:-1]
		input_embeds = sample["article_embeds"]

		labels = torch.tensor(labels)
		input_embeds = torch.tensor(input_embeds)
		input_tokens = torch.tensor(input_tokens)

		# return article, labels, labels_str, input_embeds, input_tokens
		return labels, input_embeds, input_tokens
