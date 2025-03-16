from torch import nn, Tensor
from torch.nn import functional as fn

from pipeline.mlm.model import MaskModel


class MaskModelLoss(nn.Module):
	def __init__(self):
		super(MaskModelLoss, self).__init__()

	def forward(self, pred: MaskModel.Output, y: Tensor) -> Tensor:
		y_pred = pred.y.flatten(0, 1)
		y = y.flatten()
		return fn.cross_entropy(y_pred, y)
