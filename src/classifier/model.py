import torch
from torch import nn, Tensor


class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(512, 128),
			nn.ReLU(),
			nn.Linear(128, 1),
		)

	def forward(self, embeds: Tensor, indices: Tensor) -> Tensor:
		probs = self.mlp(embeds).squeeze(1)
		zeros = torch.zeros(indices[-1] + 1, device=embeds.device)
		probs = zeros.scatter_add(0, indices, probs)
		return probs
