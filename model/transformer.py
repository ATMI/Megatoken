import importlib
from typing import Tuple, List, Optional

from torch import nn, Tensor
from torch.nn import functional as fn

from model.positional import LearnablePositionalEncoding, AbsolutePositionalEncoding


class GatedEncoderLayer(nn.Module):
	def __init__(
		self,
		model_dim: int,
		head_num: int,
		fc_dim: int,
		dropout: float,
		gate: Optional[nn.Module],
	):
		super(GatedEncoderLayer, self).__init__()
		self.encoder = nn.TransformerEncoderLayer(
			d_model=model_dim,
			nhead=head_num,
			dim_feedforward=fc_dim,
			activation=fn.gelu,
			dropout=dropout,
			batch_first=True,
		)
		self.gate = gate

	def forward(self, x: Tensor, x_pad: Tensor) -> Tuple[Tensor, Tensor]:
		x = self.encoder(x, None, x_pad, False)
		if self.gate is not None:
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
		# self.positional_z = LearnablePositionalEncoding(
		# 	model_dim=model.dim,
		# 	max_len=model.max_len,
		# 	dropout=model.positional.dropout,
		# )

		self.embedding = nn.Embedding(
			num_embeddings=model.dim,
			embedding_dim=model.dim,
			padding_idx=tokenizer.pad,
		)
		# self.embedding_z = nn.Embedding(
		# 	num_embeddings=tokenizer.vocab,
		# 	embedding_dim=model.dim,
		# 	padding_idx=tokenizer.pad,
		# )

		# gate = model.encoder.gate
		# mod = gate.path
		# mod = importlib.import_module(mod)
		#
		# cls = getattr(mod, "Gate")
		# gate = vars(gate.args) if hasattr(gate, "args") else {}

		# self.encoder = nn.ModuleList(
		# 	GatedEncoderLayer(
		# 		model_dim=model.dim,
		# 		head_num=model.encoder.head_num,
		# 		fc_dim=model.encoder.fc_dim,
		# 		dropout=model.encoder.dropout,
		# 		gate=cls(**gate)
		# 	)
		# 	for _ in range(model.encoder.layer_num)
		# )
		self.encoder = nn.TransformerEncoder(
			encoder_layer=nn.TransformerEncoderLayer(
				d_model=model.dim,
				nhead=model.encoder.head_num,
				dim_feedforward=model.encoder.fc_dim,
				dropout=model.encoder.dropout,
				activation=fn.gelu,
				batch_first=True,
			),
			num_layers=model.encoder.layer_num,
		)

		# self.decoder = nn.TransformerDecoder(
		# 	decoder_layer=nn.TransformerDecoderLayer(
		# 		d_model=model.dim,
		# 		nhead=model.decoder.head_num,
		# 		dim_feedforward=model.decoder.fc_dim,
		# 		dropout=model.decoder.dropout,
		# 		activation=fn.gelu,
		# 		batch_first=True,
		# 	),
		# 	num_layers=model.decoder.layer_num,
		# )

	@staticmethod
	def compression(inp_pad: Tensor, out_pad: Tensor) -> float:
		inp_lengths = inp_pad.size(1) - inp_pad.sum(dim=1)
		out_lengths = out_pad.size(1) - out_pad.sum(dim=1)

		ratio = (out_lengths / inp_lengths).mean()
		ratio = ratio.item()

		return ratio

	def forward(
		self,
		x: Tensor, x_pad: Tensor,
		z: Tensor, z_pad: Tensor,
	) -> Tuple[Tensor, Tensor, List[float]]:
		ratios = [1.0]

		x = z
		x_pad = z_pad

		x = self.embedding(x)
		x = self.positional(x)

		m = self.encoder(x)
		m_pad = x_pad

		# z = self.embedding(z)
		# z = self.positional(z)
		#
		# y = self.decoder(
		# 	z, m,
		# 	None, None,
		# 	z_pad, m_pad,
		# 	False, False,
		# )

		return m, m_pad, ratios
