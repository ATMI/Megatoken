from typing import Dict

import torch
from torch import nn
from torch.nn.utils import rnn

from pipeline.base.batch import Batch


class MaskModelBatch(Batch):

	def __init__(
		self,
		batch,
		pad_index: int,
		mask_index: int,
		causal_mask: bool = True,
		mask_prob: float = 0.15,
		ignore_index: int = -100,
	):
		self.batch, self.pad_mask = self.collate(batch)
		self.pad_index = pad_index
		self.mask_token_id = mask_index
		self.ignore_index = ignore_index
		self.mask_prob = mask_prob
		self.causal_mask = causal_mask

		self.masked_x = None
		self.label = None
		self.attn_mask = None

	def collate(self, batch):
		x = [torch.tensor(x["tokens"]) for x in batch]
		x = rnn.pad_sequence(x, batch_first=True, padding_value=self.pad_index)

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
	def masking(self):
		x_lens = self.pad_mask.size(1) - self.pad_mask.sum(dim=1)
		num_to_mask = (self.mask_prob * x_lens).int().clamp(min=1)

		mask = torch.zeros_like(self.batch, dtype=torch.bool)

		for i in range(self.batch.size(0)):
			candidates = torch.where(~self.pad_mask[i])[0]
			perm = torch.randperm(x_lens[i])
			selected = candidates[perm[:num_to_mask[i]]]
			mask[i, selected] = True

		masked_input = self.batch.clone()
		masked_input[mask] = self.mask_token_id

		labels = torch.full_like(
			self.batch,
			fill_value=self.ignore_index
		)  # -100 = ignore index for CrossEntropyLoss
		labels[mask] = self.batch[mask]

		return masked_input, labels

	@property
	def x(self) -> Dict[str, any]:
		self.masked_x, self.label = self.masking()

		attn_mask = None
		if self.causal_mask:
			attn_mask = nn.Transformer.generate_square_subsequent_mask(
				self.batch.size(1),
				dtype=torch.bool
			)

		# TODO: fix the input names
		inputs = {
			"x": self.batch,
			"mask_x": self.masked_x,
			"pad": self.pad_mask,
			"attn": attn_mask
		}

		return inputs

	@property
	def y(self) -> torch.Tensor:
		"""
		Shape: [batch_size * seq_len]
		:return: RETURN TENSOR FOR CROSS-ENTROPY-LOSS
		"""
		flat_label = self.label.flatten()
		return flat_label
