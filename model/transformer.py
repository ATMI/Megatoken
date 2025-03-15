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
	def __init__(self, model, tokenizer):
		super(GatedTransformer, self).__init__()

		self.positional = AbsolutePositionalEncoding(
			model_dim=model.dim,
			max_len=model.max_len,
			dropout=model.positional.dropout,
		)

		self.embedding = nn.Embedding(
			num_embeddings=tokenizer.vocab,
			embedding_dim=model.dim,
			padding_idx=tokenizer.pad,
		)

		gate = model.encoder.gate
		mod = gate.path
		mod = importlib.import_module(mod)

		cls = getattr(mod, "Gate")
		gate = vars(gate.args) if hasattr(gate, "args") else {}

		self.encoder = nn.ModuleList(
			GatedEncoderLayer(
				model_dim=model.dim,
				head_num=model.encoder.head_num,
				fc_dim=model.encoder.fc_dim,
				dropout=model.encoder.dropout,
				gate=cls(**gate)
			)
			for _ in range(model.encoder.layer_num)
		)

		self.decoder = nn.TransformerDecoder(
			decoder_layer=nn.TransformerDecoderLayer(
				d_model=model.dim,
				nhead=model.decoder.head_num,
				dim_feedforward=model.decoder.fc_dim,
				dropout=model.decoder.dropout,
				batch_first=True,
			),
			num_layers=model.decoder.layer_num,
		)

	def forward(
		self,
		x: Tensor, x_pad: Tensor,
		y: Tensor, y_pad: Tensor,
	) -> Tuple[Tensor, Tensor, List[float]]:
		ratios = []

		x = self.embedding(x)
		x = self.positional(x)

		e = x
		e_pad = x_pad
		x_len = x_pad.size(1) - x_pad.sum(dim=1)

		for encoder in self.encoder:
			t, t_pad = encoder(e, e_pad)
			t_len = t_pad.size(1) - t_pad.sum(dim=1)

			ratio = (t_len / x_len).mean(dim=0).item()
			ratios.append(ratio)

			e, e_pad = t, t_pad

		y = self.embedding(y)
		y = self.positional(y)

		y = self.decoder(
			y, e,
			None, None,
			y_pad, e_pad,
			False, False,
		)

		return y, y_pad, ratios
