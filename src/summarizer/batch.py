from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import torch
from torch import Tensor, BoolTensor, LongTensor
from torch.nn.utils import rnn


@dataclass
class SummarizerBatch:
	article_padding: BoolTensor
	article_tokens: LongTensor
	summary_tokens: LongTensor
	decoder_tokens: LongTensor

	def to(self, device) -> "SummarizerBatch":
		return SummarizerBatch(
			article_padding=self.article_padding.to(device),
			article_tokens=self.article_tokens.to(device),
			summary_tokens=self.summary_tokens.to(device),
			decoder_tokens=self.decoder_tokens.to(device),
		)

	@staticmethod
	def collate(
		batch: List[Tuple[Tensor, Tensor]],
		pad_token: int,
		ign_token: int,
	) -> "SummarizerBatch":
		article_tokens, decoder_tokens, summary_tokens = tuple(map(list, zip(*batch)))
		article_lengths = torch.tensor([len(sample) for sample in article_tokens])

		article_tokens = rnn.pad_sequence(article_tokens, batch_first=True, padding_value=pad_token)
		decoder_tokens = rnn.pad_sequence(decoder_tokens, batch_first=True, padding_value=pad_token)
		summary_tokens = rnn.pad_sequence(summary_tokens, batch_first=True, padding_value=ign_token)

		article_padding = torch.arange(article_tokens.size(1))
		article_padding = article_padding.unsqueeze(0) < article_lengths.unsqueeze(1)

		return SummarizerBatch(
			article_padding=article_padding,
			article_tokens=article_tokens,
			summary_tokens=summary_tokens,
			decoder_tokens=decoder_tokens,
		)

	@staticmethod
	def collate_fn(pad_token: int, ign_token: int):
		return partial(
			SummarizerBatch.collate,
			pad_token=pad_token,
			ign_token=ign_token,
		)
