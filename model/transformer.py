from typing import Tuple

import torch
from torch import nn

from model.decoder import Decoder
from model.encoder import Encoder


class Transformer(nn.Module):
	def __init__(
		self,

		model_dim: int,
		vocab_size: int,
		pad_idx: int,
		max_len: int,

		encoder_layer_num: int,
		encoder_head_num: int,
		encoder_fc_dim: int,
		encoder_dropout: float,

		decoder_layer_num: int,
		decoder_head_num: int,
		decoder_fc_dim: int,
		decoder_dropout: float,
	):
		super(Transformer, self).__init__()

		self.encoder = Encoder(
			model_dim=model_dim,
			layer_num=encoder_layer_num,
			head_num=encoder_head_num,
			fc_dim=encoder_fc_dim,
			vocab_size=vocab_size,
			pad_idx=pad_idx,
			max_len=max_len,
			dropout=encoder_dropout,
		)

		self.decoder = Decoder(
			model_dim=model_dim,
			layer_num=decoder_layer_num,
			head_num=decoder_head_num,
			fc_dim=decoder_fc_dim,
			dropout=decoder_dropout,
		)

	def forward(
		self,

		x: torch.Tensor,
		x_pad: torch.Tensor,

		y: torch.Tensor,
		y_pad: torch.Tensor,
	) -> Tuple[
		torch.Tensor,
		torch.Tensor,
	]:
		e, e_mask = self.encoder(x, x_pad)
		y, y_pad = self.decoder(y, y_pad, e, e_mask)
		return y, y_pad

	@staticmethod
	def from_config(config):
		return Transformer(
			model_dim=config.dim,
			vocab_size=config.vocab_size,
			pad_idx=config.pad_idx,
			max_len=config.max_len,

			encoder_layer_num=config.encoder.layer_num,
			encoder_head_num=config.encoder.head_num,
			encoder_fc_dim=config.encoder.fc_dim,
			encoder_dropout=config.encoder.dropout,

			decoder_layer_num=config.decoder.layer_num,
			decoder_head_num=config.decoder.head_num,
			decoder_fc_dim=config.decoder.fc_dim,
			decoder_dropout=config.decoder.dropout,
		)
