from torch import nn, Tensor

import prepare


class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.embedding = prepare.embedding()
		self.encoder = prepare.encoder()
		self.decoder = prepare.decoder()

	def forward(
		self,
		dense_tokens: Tensor,
		dense_tokens_pad: Tensor,

		sparse_tokens: Tensor,
		sparse_tokens_pad: Tensor,
	):
		dense_embeds = self.embedding(dense_tokens)
		dense = self.encoder(dense_embeds, dense_tokens_pad, None)

		sparse_embeds = self.embedding(sparse_tokens)
		filled = self.decoder(
			sparse_embeds, sparse_tokens_pad,
			dense.embeds, dense.pad_mask, dense.attn_mask,
		)

		return dense, filled
