from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils import rnn


@dataclass
class AutoEncoderBatch:
	input_ids: Tensor
	attention_mask: Tensor
	decoder_attention_mask: Tensor

	lengths: Tensor
	labels: Tensor

	def to(self, device) -> "AutoEncoderBatch":
		return AutoEncoderBatch(
			input_ids=self.input_ids.to(device),
			lengths=self.lengths.to(device),
			attention_mask=self.attention_mask.to(device),
			decoder_attention_mask=self.decoder_attention_mask.to(device),
			labels=self.labels.to(device),
		)

	@staticmethod
	def pad_mask(lengths: Tensor) -> Tensor:
		batch_size = lengths.size(0)
		input_length = lengths.max()

		mask = (batch_size, input_length)
		mask = torch.ones(mask, dtype=torch.bool)

		for i in range(batch_size):
			length = lengths[i]
			mask[i, length:] = 0

		return mask

	@staticmethod
	def visibility_mask(visibility: int, lengths: Tensor) -> Tensor:
		batch_size = lengths.size(0)
		input_length = lengths.max()

		mask = (batch_size, input_length, input_length)
		mask = torch.zeros(mask, dtype=torch.bool)

		for i in range(input_length):
			start = max(0, i - visibility)
			mask[:, i, start: i + 1] = 1

		for i in range(batch_size):
			start = lengths[i]
			mask[i, start:, 0] = 1

		return mask

	@staticmethod
	def collate(
		batch: List[Tuple[Tensor, Tensor]],
		visibility: int,
		pad_token: int,
		ign_token: int,
	) -> "AutoEncoderBatch":
		tokens, labels = tuple(map(list, zip(*batch)))

		lengths = torch.tensor([len(sample) for sample in tokens], dtype=torch.long)
		tokens = rnn.pad_sequence(tokens, batch_first=True, padding_value=pad_token)
		labels = rnn.pad_sequence(labels, batch_first=True, padding_value=ign_token)

		attention_mask = AutoEncoderBatch.pad_mask(lengths)
		decoder_attention_mask = AutoEncoderBatch.visibility_mask(visibility, lengths)

		return AutoEncoderBatch(
			input_ids=tokens,
			lengths=lengths,
			attention_mask=attention_mask,
			decoder_attention_mask=decoder_attention_mask,
			labels=labels,
		)

	@staticmethod
	def collate_fn(visibility: int, pad_token: int, ign_token: int):
		return partial(
			AutoEncoderBatch.collate,
			visibility=visibility,
			pad_token=pad_token,
			ign_token=ign_token,
		)
