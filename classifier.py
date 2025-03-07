from typing import Tuple, List

import torch
from torch import nn

from encoder import GatedEncoder


class Classifier(nn.Module):
	def __init__(
		self,
		vocab_size: int,
		embed_dim: int,
		pad_idx: int,

		encoder_gates_num: int,
		encoder_heads_num: int,
		encoder_fc_dim: int,

		class_num: int,
	):
		super(Classifier, self).__init__()
		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=embed_dim,
			padding_idx=pad_idx,
		)
		self.encoder = GatedEncoder(
			gates_num=encoder_gates_num,
			embed_dim=embed_dim,
			heads_num=encoder_heads_num,
			fc_dim=encoder_fc_dim,
		)
		# self.encoder = nn.TransformerEncoder(
		# 	encoder_layer=nn.TransformerEncoderLayer(
		# 		d_model=embed_dim,
		# 		nhead=encoder_heads_num,
		# 		dim_feedforward=encoder_fc_dim,
		# 		batch_first=True,
		# 	),
		# 	num_layers=len(encoder_gates),
		# )
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(embed_dim, class_num),
		)

	def forward(
		self,
		src: torch.Tensor,
		src_pad: torch.Tensor,
	) -> Tuple[
		torch.Tensor,
		torch.Tensor,
		float,
	]:
		x = self.embedding(src)
		x, x_pad = self.encoder(x, src_pad)
		# x = self.encoder(x, None, src_pad, False)

		z = (x ** 2).mean()
		x = x[:, 0]
		x = self.classifier(x)

		src_len = src_pad.size(1) - src_pad.sum(dim=1)
		x_len = x_pad.size(1) - x_pad.sum(dim=1)
		ratio = (x_len / src_len).mean(dim=0).item()
		# ratio = 1.0

		# z = 2 * (x_len / src_len).mean(dim=0)

		return x, z, ratio
