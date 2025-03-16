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
		window_size: int,

		causal: bool,
	):
		super(MaskModelBatch, self).__init__(batch)

		x, pad = self.collate(batch, pad_token)
		z, y = self.masking(x, pad, mask_token, ignore_token, prob, window_size)

		if causal:
			attn = nn.Transformer.generate_square_subsequent_mask(
				x.size(1),
				dtype=torch.bool,
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

	@staticmethod
	def masking(
		x: Tensor,
		x_pad: Tensor,
		mask_token: int,
		ignore_index: int,
		mask_prob: float,
		window_size: int = 3,
	):
		# 1) Bias with WS
		# 2) Arrange number of spans
		# 3) Permute
		# 4) randomly choose span id
		# 5) Fill mask of span
		x_lens = x_pad.size(1) - x_pad.sum(dim=1)
		num_to_mask = (mask_prob * x_lens).int().clamp(min=1)

		mask = torch.zeros_like(x, dtype=torch.bool)

		for i in range(x.size(0)):
			x_len = x_lens[i].item()
			seq_num_to_mask = num_to_mask[i].item()

			bias = torch.randint(0, window_size - 1, (1,)).item()

			num_splist = (x_len - bias) // window_size
			num_spans = seq_num_to_mask // window_size
			print(num_spans)

			spans = torch.arange(num_splist, dtype=torch.int)
			perm = torch.randperm(len(spans))
			selected_spans = spans[perm[:num_spans]]

			# Convert spans idx to first index of each span
			spans_beg = [window_size * i + bias for i in selected_spans]

			selected_tok = [torch.arange(start, start + window_size) for start in spans_beg]
			selected_tok = torch.concat(selected_tok)
			mask[i, selected_tok] = True

		masked_input = x.clone()
		masked_input[mask] = mask_token

		labels = torch.full_like(x, fill_value=ignore_index)  # -100 = ignore index for CrossEntropyLoss
		labels[mask] = x[mask]

		return masked_input, labels


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
