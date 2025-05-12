import torch
from torch import nn, Tensor


class Classifier(nn.Module):
	def __init__(self, num_classes: int):
		super(Classifier, self).__init__()
		self.num_classes = num_classes
		self.mlp = nn.Sequential(
			nn.Linear(512, 128),
			nn.ReLU(),
			nn.Linear(128, num_classes),
		)

	def forward(self, embeds: Tensor, indices: Tensor) -> Tensor:
		probs = self.mlp(embeds)
		zeros = torch.zeros((indices[-1] + 1, self.num_classes), device=embeds.device)
		probs = zeros.index_add(0, indices, probs)
		return probs
