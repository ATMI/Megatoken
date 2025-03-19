import math

import torch
from torch import nn


class AbsolutePositionalEncoding(nn.Module):
	def __init__(
		self,
		model_dim: int,
		max_len: int,
		dropout: float,
	):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
		pe = torch.zeros(1, max_len, model_dim)
		pe[0, :, 0::2] = torch.sin(position * div_term)
		pe[0, :, 1::2] = torch.cos(position * div_term)
		self.register_buffer("pe", pe)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Arguments:
			x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
		"""
		x = x + self.pe[:, :x.size(1)]
		return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
	def __init__(
		self,
		model_dim: int,
		max_len: int,
		dropout: float,
	):
		super(LearnablePositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.positional = nn.Embedding(
			num_embeddings=max_len,
			embedding_dim=model_dim,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		indices = torch.arange(x.size(1), device=x.device)

		pos = self.positional(indices)
		pos = pos.unsqueeze(0)

		x = x + pos
		x = self.dropout(x)

		return x
