import random
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
		max_window: int,
		min_window: int,

		causal: bool,
	):
		super(MaskModelBatch, self).__init__(batch)

		x, pad = self.collate(batch, pad_token)
		z, y = self.span_masking(x, pad, mask_token, ignore_token, prob, max_window, min_window)

		if causal:
			attn = nn.Transformer.generate_square_subsequent_mask(
				x.size(1),
				dtype=torch.bool,
			)
			raise NotImplementedError("Attn is not used by the model for now")
		else:
			attn = None

		self.x_ = x
		self.z_ = z
		self.y_ = y

		self.pad = pad
		self.attn = attn
		self.ignore_token = ignore_token

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

	@staticmethod
	def span_masking(
		x,
		x_pad,
		mask_token: int,
		ignore_index: int,
		mask_prob: float,
		min_span: int = 3,
		max_span: int = 5,
	):
		x_lens = x_pad.size(1) - x_pad.sum(dim=1)
		nums_to_mask = (mask_prob * x_lens).int().clamp(min=1)

		mask = torch.zeros_like(x, dtype=torch.bool)

		for i in range(x.size(0)):
			span_size = random.randint(min_span, max_span)
			x_len = x_lens[i].item()
			num_mask_tokens = nums_to_mask[i].item()  # Number of tokens to mask in current sequence

			num_splits = int(x_len // span_size)  # Split sequence into equal spans
			num_spans = max(num_mask_tokens // span_size, 1)  # Number of spans we must mask

			# Randomly select spans to mask
			spans = torch.arange(num_splits, dtype=torch.int)
			perm = torch.randperm(num_splits)
			selected_spans = spans[perm[:num_spans]]

			# Convert spans idx to token idx
			spans_start_ids = selected_spans * span_size
			masked_tokens_ids = (spans_start_ids.unsqueeze(1) + torch.arange(span_size)).flatten()

			mask[i, masked_tokens_ids] = True

		masked_input = x.clone()
		masked_input[mask] = mask_token

		labels = torch.full_like(x, fill_value=ignore_index)
		labels[mask] = x[mask]

		return masked_input, labels


	def to(self, device: torch.device):
		self.x_ = self.x_.to(device)
		self.y_ = self.y_.to(device)
		self.z_ = self.z_.to(device)

		self.pad = self.pad.to(device)
		if self.attn is not None:
			self.attn = self.attn.to(device)

		return self

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
