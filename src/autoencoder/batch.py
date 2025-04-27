from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Dict

import torch
from torch import Tensor
from torch.nn.utils import rnn


@dataclass
class AutoEncoderBatch:
	lengths: Tensor
	labels: Tensor

	encoder_input_ids: Tensor
	decoder_input_ids: Tensor
	pad_mask: Tensor

	def to(self, device) -> "AutoEncoderBatch":
		return AutoEncoderBatch(
			lengths=self.lengths.to(device),
			labels=self.labels.to(device),

			encoder_input_ids=self.encoder_input_ids.to(device),
			decoder_input_ids=self.decoder_input_ids.to(device),
			pad_mask=self.pad_mask.to(device),
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
		pad_token: int,
		ign_token: int,
	) -> "AutoEncoderBatch":
		encoder_input_ids, decoder_input_ids = tuple(map(list, zip(*batch)))
		lengths = torch.tensor([len(sample) for sample in encoder_input_ids], dtype=torch.long)
		labels = encoder_input_ids

		labels = rnn.pad_sequence(labels, batch_first=True, padding_value=ign_token)
		encoder_input_ids = rnn.pad_sequence(encoder_input_ids, batch_first=True, padding_value=pad_token)
		decoder_input_ids = rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=pad_token)
		pad_mask = AutoEncoderBatch.pad_mask(lengths)

		return AutoEncoderBatch(
			lengths=lengths,
			labels=labels,

			encoder_input_ids=encoder_input_ids,
			decoder_input_ids=decoder_input_ids,
			pad_mask=pad_mask,
		)

	@staticmethod
	def collate_fn(pad_token: int, ign_token: int):
		return partial(
			AutoEncoderBatch.collate,
			pad_token=pad_token,
			ign_token=ign_token,
		)
