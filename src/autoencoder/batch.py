from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils import rnn


@dataclass
class AutoEncoderBatch:
	tokens: Tensor
	labels: Tensor
	lengths: Tensor


	def to(self, device) -> "AutoEncoderBatch":
		return AutoEncoderBatch(
			tokens=self.tokens.to(device),
			labels=self.labels.to(device),
			lengths=self.lengths.to(device),

		)

	@staticmethod
	def collate(
		batch: List[Tuple[Tensor, Tensor]],
		pad_token: int,
		ign_token: int,
	) -> "AutoEncoderBatch":
		batch_size = len(batch)
		tokens, labels = tuple(map(list, zip(*batch)))

		lengths = torch.empty(batch_size, dtype=torch.long)
		for i, sample in enumerate(tokens):
			lengths[i] = len(sample)

		tokens = rnn.pad_sequence(tokens, batch_first=True, padding_value=pad_token)
		labels = rnn.pad_sequence(labels, batch_first=True, padding_value=ign_token)

		return AutoEncoderBatch(
			tokens=tokens,
			labels=labels,
			lengths=lengths,

		)

	@staticmethod
	def collate_fn(pad_token: int, ign_token: int):
		return partial(
			AutoEncoderBatch.collate,
			pad_token=pad_token,
			ign_token=ign_token,
		)
