import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
	def __init__(self, model_dim: int, max_len: int):
		super().__init__()
		self.positional = nn.Embedding(
			num_embeddings=max_len,
			embedding_dim=model_dim,
		)

	def forward(self, inputs: Tensor) -> Tensor:
		indices = torch.arange(inputs.size(1), device=inputs.device)

		positional = self.positional(indices)
		positional = positional.unsqueeze(0)

		outputs = inputs + positional
		return outputs