import importlib
from typing import Tuple, List

from torch import nn, Tensor

from model.positional import AbsolutePositionalEncoding


class GatedEncoderLayer(nn.Module):
	def __init__(
		self,
		model_dim: int,
		head_num: int,
		fc_dim: int,
		dropout: float,
		gate: nn.Module,
	):
		super(GatedEncoderLayer, self).__init__()
		self.encoder = nn.TransformerEncoderLayer(
			d_model=model_dim,
			nhead=head_num,
			dim_feedforward=fc_dim,
			dropout=dropout,
			batch_first=True,
		)
		self.gate = gate

	def forward(self, x: Tensor, x_pad: Tensor) -> Tuple[Tensor, Tensor]:
		x = self.encoder(x, None, x_pad, False)
		x, x_pad = self.gate(x, x_pad)
		return x, x_pad


class GatedTransformer(nn.Module):
	def __init__(self, config):
		super(GatedTransformer, self).__init__()

		self.positional = AbsolutePositionalEncoding(
			model_dim=config.dim,
			max_len=config.max_len,
			dropout=config.positional.dropout,
		)

		self.embedding = nn.Embedding(
			num_embeddings=config.embedding.size,
			embedding_dim=config.dim,
			padding_idx=config.embedding.pad,
		)

		gate = config.encoder.gate
		mod, cls = gate.name.rsplit(".", 1)
		mod = importlib.import_module(mod)

		cls = getattr(mod, cls)
		del gate.name
		gate = vars(gate)

		self.encoder = nn.ModuleList(
			GatedEncoderLayer(
				model_dim=config.dim,
				head_num=config.encoder.head_num,
				fc_dim=config.encoder.fc_dim,
				dropout=config.encoder.dropout,
				gate=cls(**gate)
			)
			for _ in range(config.encoder.layer_num)
		)

		self.decoder = nn.TransformerDecoder(
			decoder_layer=nn.TransformerDecoderLayer(
				d_model=config.dim,
				nhead=config.decoder.head_num,
				dim_feedforward=config.decoder.fc_dim,
				dropout=config.decoder.dropout,
				batch_first=True,
			),
			num_layers=config.decoder.layer_num,
		)

	def forward(
		self,
		x: Tensor, x_pad: Tensor,
		y: Tensor, y_pad: Tensor,
	) -> Tuple[Tensor, Tensor, List[float]]:
		ratio = []

		x = self.embedding(x)
		x = self.positional(x)

		for encoder in self.encoder:
			e, e_pad = encoder(x, x_pad)

			e_len = e_pad.size(1) - e_pad.sum(dim=1)
			x_len = x_pad.size(1) - x_pad.sum(dim=1)
			r = (e_len / x_len).mean(dim=0).item()
			ratio.append(r)

			x, x_pad = e, e_pad

		y = self.embedding(y)
		y = self.positional(y)

		y = self.decoder(
			y, x,
			None, None,
			y_pad, x_pad,
			False, False,
		)

		return y, y_pad, ratio
