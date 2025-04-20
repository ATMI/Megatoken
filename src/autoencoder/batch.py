from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor
from torch.nn.utils import rnn

from .config import Config


@dataclass
class Batch:
	ids: List[int]
	inputs: Tensor
	labels: Tensor

	eos_mask: Tensor
	pad_mask: Tensor
	decoder_mask: Tensor

	def to(self, device) -> "Batch":
		return Batch(
			self.ids,
			self.inputs.to(device),
			self.labels.to(device),
			self.eos_mask.to(device),
			self.pad_mask.to(device),
			self.decoder_mask.to(device),
		)

	@staticmethod
	def collate(batch) -> "Batch":
		ids = [row["id"] for row in batch]
		tokens = [row["tokens"] for row in batch]
		inputs = [torch.tensor(row) for row in tokens]
		inputs = rnn.pad_sequence(inputs, batch_first=True, padding_value=Config.pad_token)

		batch_size = inputs.size(0)
		seq_length = inputs.size(1)

		eos_mask = torch.empty(batch_size, dtype=torch.long)
		pad_mask = torch.ones((batch_size, seq_length), dtype=torch.bool)
		decoder_mask = torch.full((batch_size, seq_length, seq_length), -torch.inf)

		for i in range(seq_length):
			decoder_mask[:, i:i + Config.decoder_visibility + 1, i] = 0

		for i, sample in enumerate(tokens):
			length = len(sample)
			pad_mask[i, length:] = 0
			eos_mask[i] = length - 1
			decoder_mask[i, length:, 0] = 0

		eos_mask = torch.stack((torch.arange(batch_size), eos_mask))
		labels = torch.full((batch_size, 1), Config.ignore_token)
		labels = torch.cat((inputs[:, 1:], labels), dim=1)
		labels[~pad_mask] = Config.ignore_token

		return Batch(
			ids,
			inputs,
			labels,
			eos_mask,
			pad_mask,
			decoder_mask,
		)
