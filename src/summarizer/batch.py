from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils import rnn


@dataclass
class SummarizerBatch:
	# article: List[str]
	labels: Tensor
	# labels_str: List[str]
	pad_mask: Tensor
	input_embeds: Tensor
	input_tokens: Tensor

	def to(self, device) -> "SummarizerBatch":
		return SummarizerBatch(
			# article=self.article,
			labels=self.labels.to(device),
			# labels_str=self.labels_str,
			pad_mask=self.pad_mask.to(device),
			input_embeds=self.input_embeds.to(device),
			input_tokens=self.input_tokens.to(device),
		)

	@staticmethod
	def collate(
		batch: List[Tuple[Tensor, Tensor]],
		pad_token: int,
		ign_token: int,
	) -> "SummarizerBatch":
		labels, input_embeds, input_tokens = tuple(map(list, zip(*batch)))
		# article, labels, labels_str, input_embeds, input_tokens = tuple(map(list, zip(*batch)))
		lengths = torch.tensor([len(sample) for sample in input_embeds])

		labels = rnn.pad_sequence(labels, batch_first=True, padding_value=ign_token)
		input_embeds = rnn.pad_sequence(input_embeds, batch_first=True, padding_value=0.0)
		input_tokens = rnn.pad_sequence(input_tokens, batch_first=True, padding_value=pad_token)

		pad_mask = torch.arange(input_embeds.size(1))
		pad_mask = pad_mask.unsqueeze(0) < lengths.unsqueeze(1)

		return SummarizerBatch(
			# article=article,
			labels=labels,
			# labels_str=labels_str,
			pad_mask=pad_mask,
			input_embeds=input_embeds,
			input_tokens=input_tokens,
		)

	@staticmethod
	def collate_fn(pad_token: int, ign_token: int):
		return partial(
			SummarizerBatch.collate,
			pad_token=pad_token,
			ign_token=ign_token,
		)
