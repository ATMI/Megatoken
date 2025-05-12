from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils import rnn


@dataclass
class ClassifierBatch:
	labels: Tensor
	pad_mask: Tensor
	input_ids: Tensor

	def to(self, device) -> "ClassifierBatch":
		return ClassifierBatch(
			labels=self.labels.to(device),
			pad_mask=self.pad_mask.to(device),
			input_ids=self.input_ids.to(device),
		)

	@staticmethod
	def collate(
		batch: List[Tuple[Tensor, int]],
		pad_token: int,
	) -> "ClassifierBatch":
		input_ids, labels = tuple(map(list, zip(*batch)))

		labels = torch.tensor(labels, dtype=torch.long)
		lengths = torch.tensor([len(sample) for sample in input_ids], dtype=torch.long)
		input_ids = rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token)

		pad_mask = torch.arange(input_ids.size(1))
		pad_mask = pad_mask.unsqueeze(0) < lengths.unsqueeze(1)

		return ClassifierBatch(
			labels=labels,
			pad_mask=pad_mask,
			input_ids=input_ids,
		)

	@staticmethod
	def collate_fn(pad_token: int):
		return partial(
			ClassifierBatch.collate,
			pad_token=pad_token,
		)
