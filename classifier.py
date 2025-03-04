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

		encoder_throughput: List[float],
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
			throughput=encoder_throughput,
			embed_dim=embed_dim,
			heads_num=encoder_heads_num,
			fc_dim=encoder_fc_dim,
		)
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
		int,
	]:
		x = self.embedding(src)
		x, x_pad = self.encoder(x, src_pad)

		x = x[:, 0]
		x = self.classifier(x)

		return x
