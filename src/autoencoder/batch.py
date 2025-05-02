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
	def collate(
		batch: List[Tuple[Tensor, Tensor]],
		pad_token: int,
		ign_token: int,
	) -> "AutoEncoderBatch":
		encoder_input_ids, decoder_input_ids, label_ids = tuple(map(list, zip(*batch)))
		lengths = torch.tensor([len(sample) for sample in encoder_input_ids], dtype=torch.long)

		label_ids = rnn.pad_sequence(label_ids, batch_first=True, padding_value=ign_token)
		encoder_input_ids = rnn.pad_sequence(encoder_input_ids, batch_first=True, padding_value=pad_token)
		decoder_input_ids = rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=pad_token)

		pad_mask = torch.arange(encoder_input_ids.size(1))
		pad_mask = pad_mask.unsqueeze(0) < lengths.unsqueeze(1)

		return AutoEncoderBatch(
			lengths=lengths,
			labels=label_ids,

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
