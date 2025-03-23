from dataclasses import dataclass
from typing import Tuple, Callable

import torch
from torch import Tensor
from torch import nn


class GatedEncoderLayer(nn.Module):
	def __init__(
		self,
		model_dim: int,
		head_num: int,
		fc_dim: int,
		activation: str | Callable[[Tensor], Tensor],

		bias: float,
		temperature: float,
	):
		super(GatedEncoderLayer, self).__init__()
		self.head_num = head_num
		self.encoder = nn.TransformerEncoderLayer(
			d_model=model_dim,
			nhead=head_num,
			dim_feedforward=fc_dim,
			activation=activation,
			batch_first=True,
		)
		self.bias = bias
		self.temperature = temperature

	def forward(
		self,
		inputs: Tensor,
		pad_mask: Tensor,
		attn_mask: Tensor,
	) -> Tuple[Tensor, Tensor, Tensor]:
		head_mask = attn_mask.repeat_interleave(self.head_num, dim=0)
		outputs = self.encoder(inputs, head_mask, pad_mask, False)
		del head_mask

		valves = (outputs[:, :, 0] + self.bias) / self.temperature
		valves = valves.sigmoid()
		valves = valves.clamp(min=1e-10, max=1)

		outputs = outputs * valves.unsqueeze(2)

		device = inputs.device
		input_length = inputs.size(1)
		indices = torch.arange(input_length, device=device)

		mask = valves.log()
		mask = mask.unsqueeze(2)
		mask = mask.repeat(1, 1, input_length)
		mask[:, indices, indices] = 0

		attn_mask = attn_mask + mask
		return outputs, attn_mask, valves


class Encoder(nn.Module):
	@dataclass
	class Outputs:
		embeds: Tensor
		lengths: Tensor
		pad_mask: Tensor
		attn_mask: Tensor

	def __init__(
		self,
		model_dim: int,
		head_num: int,
		fc_dim: int,
		activation: str | Callable[[Tensor], Tensor],
		layer_num: int,

		bias: float,
		temperature: float,
	):
		super(Encoder, self).__init__()
		self.layer_num = layer_num
		self.layers = nn.ModuleList(
			GatedEncoderLayer(
				model_dim=model_dim,
				head_num=head_num,
				fc_dim=fc_dim,
				activation=activation,
				bias=bias,
				temperature=temperature,
			)
			for _ in range(layer_num)
		)

	def forward(
		self,
		inputs: Tensor,
		pad_mask: Tensor,
		attn_mask: Tensor,
	) -> Outputs:
		device = inputs.device
		batch_size = inputs.size(0)

		input_mask = torch.where(pad_mask, 0, 1)
		pad_mask = torch.where(pad_mask, -torch.inf, 0)

		outputs = inputs
		output_lengths = torch.zeros((batch_size, self.layer_num), device=device)
		for i, layer in enumerate(self.layers):
			outputs, attn_mask, valves = layer(outputs, pad_mask, attn_mask)
			lengths = (valves * input_mask).sum(dim=1)
			output_lengths[:, i] = lengths

		return Encoder.Outputs(
			embeds=outputs,
			lengths=output_lengths,
			pad_mask=pad_mask,
			attn_mask=attn_mask,
		)
