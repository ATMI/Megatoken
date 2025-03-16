from typing import Dict, Tuple

import torch
from torch import nn, Tensor
from torch.nn.utils import rnn

from pipeline.base.batch import Batch


class MaskModelBatch(Batch):

	def __init__(
		self,
		batch: any,

		pad_token: int,
		mask_token: int,
		ignore_token: int,
		prob: float,

		causal: bool,
	):
		super(MaskModelBatch, self).__init__(batch)

		x, pad = self.collate(batch, pad_token)
		z, y = self.masking(x, pad, mask_token, ignore_token, prob)

		if causal:
			attn = nn.Transformer.generate_square_subsequent_mask(
				x.size(1),
				torch.bool,
			)
			raise NotImplementedError("Attn is not used by the model for now")
		else:
			attn = None

		self.x_, self.pad = x, pad
		self.z_ = z
		self.y_ = y
		self.attn = attn

	@staticmethod
	def collate(batch, pad_token):
		x = [torch.tensor(x["tokens"]) for x in batch]
		x = rnn.pad_sequence(x, batch_first=True, padding_value=pad_token)

		batch_size = x.size(0)
		seq_len = x.size(1)

		x_pad = torch.ones(
			batch_size,
			seq_len,
			dtype=torch.bool,
			device=x.device
		)

		for i in range(batch_size):
			tokens = batch[i]["tokens"]
			length = len(tokens)
			x_pad[i, :length] = False

		return x, x_pad

	# TODO: make wider masks and mode
	@staticmethod
	def masking(
		x: Tensor,
		x_pad: Tensor,
		mask_token: int,
		ignore_token: int,
		prob: float,
	) -> Tuple[Tensor, Tensor]:
		x_len = x_pad.size(1) - x_pad.sum(dim=1)

		num_to_mask = (prob * x_len).int().clamp(min=1)
		mask = torch.zeros_like(x, dtype=torch.bool)

		for i in range(x.size(0)):
			candidates = torch.where(~x_pad[i])[0]
			perm = torch.randperm(x_len[i])
			selected = candidates[perm[:num_to_mask[i]]]
			mask[i, selected] = True

		y = x.clone()
		y[mask] = mask_token

		# -100 = ignore index for CrossEntropyLoss
		labels = torch.full_like(x, fill_value=ignore_token)
		labels[mask] = x[mask]

		return y, labels

	@property
	def x(self) -> Dict[str, any]:
		x = {
			"x": self.x_,
			"x_pad": self.pad,

			"z": self.z_,
			"z_pad": self.pad,
		}
		return x

	@property
	def y(self) -> torch.Tensor:
		"""
		Shape: [batch_size * seq_len]
		:return: RETURN TENSOR FOR CROSS-ENTROPY-LOSS
		"""
		# y = self.y_.flatten()
		return self.y_
