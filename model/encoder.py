import copy
from typing import Optional

from torch import nn, Tensor
from torch.nn import ModuleList


def _get_clones(module, N):
	return ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
	def __init__(
			self,
			encoder_layer,
			num_layers,
			norm: Optional[nn.Module] = None
	):
		super(Encoder, self).__init__()
		self.layers = _get_clones(encoder_layer, num_layers)
		self.norm = norm


	def forward(
			self,
			x,
			mask = None,
			src_key_padding_mask = None,
	) -> Tensor:
		for layer in self.layers:
			x = layer(
				x,
				mask=mask,
				src_key_padding_mask=src_key_padding_mask
			)

		if self.norm is not None:
			x = self.norm(x)

		return x