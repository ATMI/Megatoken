from dataclasses import dataclass
from functools import partial
from typing import List

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
		batch: List[Tensor],
		pad_token: int,
		ign_token: int,
	) -> "AutoEncoderBatch":
		tokens = rnn.pad_sequence(
			batch,
			batch_first=True,
			padding_value=pad_token,
		)

		batch_size = tokens.size(0)
		lengths = torch.tensor((batch_size,), dtype=torch.long)
		labels = torch.roll(tokens, -1)

		for i, sample in enumerate(tokens):
			length = len(sample)
			lengths[i] = length
			labels[i, length:] = ign_token

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
