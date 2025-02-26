import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
	def __init__(self, model_dim, max_len=512):
		super(PositionalEncoding, self).__init__()
		pe = torch.zeros(max_len, model_dim)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer("pe", pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return x


class Encoder(nn.Module):
	def __init__(
		self,
		model_dim: int,
		fc_dim: int,
		head_num: int,
		layer_num: int,
	):
		super(Encoder, self).__init__()
		self.encoder = nn.TransformerEncoder(
			encoder_layer=nn.TransformerEncoderLayer(
				d_model=model_dim,
				nhead=head_num,
				dim_feedforward=fc_dim,
				batch_first=True,
			),
			num_layers=layer_num,
		)

	def forward(
		self,
		x: torch.Tensor,
		x_pad: torch.Tensor,
	) -> torch.Tensor:
		x = self.encoder(x, None, x_pad, False)
		x = x[:, 0]

		return x


class Decoder(nn.Module):
	def __init__(
		self,
		model_dim: int,
		fc_dim: int,
		head_num: int,
		layer_num: int,
	):
		super(Decoder, self).__init__()
		self.decoder = nn.TransformerEncoder(
			encoder_layer=nn.TransformerEncoderLayer(
				d_model=model_dim,
				nhead=head_num,
				dim_feedforward=fc_dim,
				batch_first=True,
			),
			num_layers=layer_num,
		)

	def forward(
		self,
		x: torch.Tensor,
		x_pad: torch.Tensor,
	) -> torch.Tensor:
		x_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), x.device, torch.bool)
		x = self.decoder(x, x_mask, x_pad, True)

		return x


class AutoEncoder(nn.Module):
	def __init__(
		self,

		vocab_size: int,
		pad_idx: int,

		model_dim: int,
		max_len: int,

		encoder_head_num: int,
		decoder_head_num: int,

		encoder_layer_num: int,
		decoder_layer_num: int,

		encoder_fc_dim: int,
		decoder_fc_dim: int,
	):
		super(AutoEncoder, self).__init__()
		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=model_dim,
			padding_idx=pad_idx,
		)
		self.positional = PositionalEncoding(
			model_dim=model_dim,
			max_len=max_len,
		)
		self.encoder = Encoder(
			model_dim=model_dim,
			fc_dim=encoder_fc_dim,
			head_num=encoder_head_num,
			layer_num=encoder_layer_num,
		)
		self.decoder = Decoder(
			model_dim=model_dim,
			fc_dim=decoder_fc_dim,
			head_num=decoder_head_num,
			layer_num=decoder_layer_num,
		)
		self.classifier = nn.Linear(
			in_features=model_dim,
			out_features=vocab_size,
		)

	def forward(
		self,
		x: torch.Tensor,
		x_pad: torch.Tensor,
	) -> torch.Tensor:
		x = self.embedding(x)
		x = self.positional(x)

		e = self.encoder(x, x_pad)
		e = e.unsqueeze(1)
		e = self.positional(e)
		e = e.squeeze(1)
		x[:, 0] = e

		x = self.decoder(x, x_pad)
		x = self.classifier(x)

		return x
