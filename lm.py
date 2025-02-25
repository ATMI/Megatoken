import torch
from torch import nn


class LanguageModel(nn.Module):
	def __init__(
		self,

		vocab_size: int,
		embed_dim: int,
		pad_idx: int,

		head_num: int,
		layer_num: int,
		feedforward_dim: int,
	):
		super().__init__()

		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=embed_dim,
			padding_idx=pad_idx,
		)

		self.encoder = nn.TransformerEncoder(
			encoder_layer=nn.TransformerEncoderLayer(
				d_model=embed_dim,
				nhead=head_num,
				dim_feedforward=feedforward_dim,
				batch_first=True,
			),
			num_layers=layer_num,
		)

		self.classifier = nn.Linear(
			in_features=embed_dim,
			out_features=vocab_size,
		)

	def forward(
		self,
		x: torch.Tensor,
		x_pad: torch.Tensor,
	) -> torch.Tensor:
		x = self.embedding(x)
		x_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), x.device, torch.bool)
		x = self.encoder(x, x_mask, x_pad, True)
		x = self.classifier(x)

		return x
