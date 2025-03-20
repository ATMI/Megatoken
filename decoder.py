from typing import Callable, Union

from torch import Tensor
from torch import nn

from positional import PositionalEncoding


class Decoder(nn.Module):
	def __init__(
		self,
		vocab_size: int,
		padding_idx: int,
		max_len: int,

		model_dim: int,
		head_num: int,
		fc_dim: int,
		activation: Union[str, Callable[[Tensor], Tensor]],

		layer_num: int,
	):
		super(Decoder, self).__init__()
		self.head_num = head_num
		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=model_dim,
			padding_idx=padding_idx,
		)
		self.positional = PositionalEncoding(
			model_dim=model_dim,
			max_len=max_len,
		)
		self.decoder = nn.TransformerDecoder(
			decoder_layer=nn.TransformerDecoderLayer(
				d_model=model_dim,
				nhead=head_num,
				dim_feedforward=fc_dim,
				activation=activation,
				batch_first=True,
			),
			num_layers=layer_num,
		)
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(model_dim, vocab_size),
		)

	def forward(
		self,
		sparse_tokens: Tensor,
		sparse_pad_mask: Tensor,

		dense: Tensor,
		dense_pad_mask: Tensor,
		dense_attn_mask: Tensor,
	) -> Tensor:
		sparse = self.embedding(sparse_tokens)
		sparse = self.positional(sparse)

		head_num = self.head_num
		# dense_attn_mask = dense_attn_mask.repeat_interleave(head_num, dim=0)

		outputs = self.decoder(
			sparse, dense,
			None, dense_attn_mask,
			sparse_pad_mask, dense_pad_mask,
		)
		outputs = self.classifier(outputs)

		return outputs
