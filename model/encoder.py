from typing import Tuple

import torch
from torch import nn

from model.gate import GatedEncoderLayer
from model.positional import AbsolutePositionalEncoding


class Encoder(nn.Module):
	def __init__(
		self,

		model_dim: int,
		layer_num: int,
		head_num: int,
		fc_dim: int,

		vocab_size: int,
		pad_idx: int,
		max_len: int,

		dropout: float,
	):
		super(Encoder, self).__init__()

		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=model_dim,
			padding_idx=pad_idx,
		)

		self.positional = AbsolutePositionalEncoding(
			model_dim=model_dim,
			max_len=max_len,
			dropout=dropout,
		)

		self.encoder = nn.ModuleList(
			GatedEncoderLayer(
				embed_dim=model_dim,
				head_num=head_num,
				fc_dim=fc_dim,
			)
			for _ in range(layer_num)
		)

	def forward(
		self,
		x: torch.Tensor,
		x_pad: torch.Tensor,
	) -> Tuple[
		torch.Tensor,
		torch.Tensor,
	]:
		e = self.embedding(x)
		e = self.positional(e)
		e, e_mask = self.encoder(e, x_pad)
		return e, e_mask
