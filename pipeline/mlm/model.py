import torch
from torch import nn

from model.transformer import GatedTransformer


class MaskModel(nn.Module):
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
	) -> any:
		y, y_pad, ratio = self.transformer(x, x_pad, z, z_pad)
		y = self.classifier(y)
		return y, y_pad, ratio
