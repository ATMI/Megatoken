from dataclasses import dataclass
from typing import List

import torch
from torch import nn, Tensor

from model.transformer import GatedTransformer


class MaskModel(nn.Module):
	@dataclass
	class Output:
		y: Tensor
		y_pad: Tensor
		ratios: List[float]

	def __init__(self, model, tokenizer):
		super(MaskModel, self).__init__()

		self.transformer = GatedTransformer(model, tokenizer)
		self.classifier = nn.Sequential(
			nn.Dropout(model.classifier.dropout),
			nn.Linear(model.dim, tokenizer.vocab),
		)

	def forward(
		self,
		x: torch.Tensor,
		x_pad: torch.Tensor,

		z: torch.Tensor,
		z_pad: torch.Tensor,
	) -> Output:
		y, y_pad, ratio = self.transformer(x, x_pad, z, z_pad)
		y = self.classifier(y)
		return MaskModel.Output(y, y_pad, ratio)
