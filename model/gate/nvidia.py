from typing import Tuple

import torch
from torch import nn, Tensor


class Gate(nn.Module):
	def __init__(self):
		super(Gate, self).__init__()

	def forward(self, x: Tensor, x_pad: Tensor) -> Tuple[Tensor, Tensor]:
		return x, x_pad

		batch_size = x.size(0)
		x_max_len = x.size(1)
		embed_dim = x.size(2)

		a = x[:, :, 0]
		a = a.sigmoid()
		a = a.round()
		a = a.bool() & ~x_pad
		a[:, 0] = False

		w = x[:, :, 1]
		w = w.sigmoid()
		w = w.unsqueeze(2)

		x[:, :, 0] = 0
		x[:, :, 1] = 0

		y_len = a.sum(dim=1) + 1
		y_max_len = y_len.max().item()

		y = torch.zeros(
			(batch_size, y_max_len, embed_dim),
			dtype=x.dtype,
			device=x.device,
		)

		index = torch.zeros((batch_size,), dtype=torch.int, device=x.device)
		z_i = torch.zeros((batch_size, 1), dtype=x.dtype, device=x.device)

		for i in range(x_max_len):
			x_i = x[:, i]
			w_i = w[:, i]
			a_i = a[:, i]

			index = index + a_i
			a_i = a_i.unsqueeze(1)

			y_i = y[:, index] * z_i + x_i * w_i
			z_i = torch.where(a_i, w_i, z_i + w_i)
			y[:, index] = torch.where(a_i, x_i, y_i / z_i)

		y_pad = torch.arange(y_max_len, device=x_pad.device)
		y_pad = y_pad >= y_len.unsqueeze(1)

		return y, y_pad
