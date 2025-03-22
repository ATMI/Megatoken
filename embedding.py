from torch import nn, Tensor

from positional import PositionalEncoding


class Embedding(nn.Module):
	def __init__(
		self,
		model_dim: int,
		vocab_size: int,
		pad_token: int,
		max_len: int,
	):
		super(Embedding, self).__init__()
		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=model_dim,
			padding_idx=pad_token,
		)
		self.positional = PositionalEncoding(
			model_dim=model_dim,
			max_len=max_len,
		)

	def forward(self, tokens: Tensor) -> Tensor:
		outputs = self.embedding(tokens)
		outputs = self.positional(outputs)
		return outputs
