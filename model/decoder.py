from typing import Tuple

import torch
from torch import nn


class Decoder(nn.Module):
	def __init__(
		self,

		model_dim: int,
		layer_num: int,
		head_num: int,
		fc_dim: int,

		dropout: float,
	):
		super(Decoder, self).__init__()

		self.decoder = nn.TransformerDecoder(
			decoder_layer=nn.TransformerDecoderLayer(
				d_model=model_dim,
				nhead=head_num,
				dim_feedforward=fc_dim,
				dropout=dropout,
				batch_first=True,
			),
			num_layers=layer_num,
		)

	def forward(
		self,
		x: torch.Tensor,
		x_pad: torch.Tensor,

		e: torch.Tensor,
		e_pad: torch.Tensor,
	) -> Tuple[
		torch.Tensor,
		torch.Tensor,
	]:
		y = self.decoder(x, e, None, None, x_pad, e_pad, False, False)
		return y, x_pad
