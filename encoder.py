from dataclasses import dataclass
from typing import Tuple, Callable, Optional

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
		device = inputs.device
		input_length = inputs.size(1)

		valves = (inputs[:, :, 0] + self.bias) / self.temperature
		valves = valves.sigmoid()
		valves = valves.clamp(min=1e-10, max=1)

		inputs = inputs * valves.unsqueeze(2)

		mask = valves.log()  # batch, input_length
		mask = mask.unsqueeze(1)
		mask = mask.repeat(1, input_length, 1)  # batch, 1, input_length

		indices = torch.arange(input_length, device=device)
		mask[:, indices, indices] = 0

		# repeating mask for each head
		attn_mask = attn_mask + mask
		head_mask = attn_mask.repeat_interleave(self.head_num, dim=0)
		outputs = self.encoder(inputs, head_mask, pad_mask, False)

		return outputs, attn_mask, valves


class Encoder(nn.Module):
	@dataclass
	class Outputs:
		embeds: Tensor
		pad_mask: Tensor
		attn_mask: Tensor
		pressure: Tensor

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
		attn_mask: Optional[Tensor],
	) -> Outputs:
		device = inputs.device
		batch_size = inputs.size(0)

		input_mask = ~pad_mask
		input_length = inputs.size(1)

		pad_mask = torch.where(pad_mask, -torch.inf, 0)
		if attn_mask is None:
			attn_mask = torch.zeros((batch_size, input_length, input_length), device=device)

		outputs = inputs
		pressure = torch.zeros((self.layer_num,), device=device)
		for i, layer in enumerate(self.layers):
			outputs, attn_mask, valves = layer(outputs, pad_mask, attn_mask)
			pressure[i] = valves[input_mask].sum()

		return Encoder.Outputs(
			embeds=outputs,
			pad_mask=pad_mask,
			attn_mask=attn_mask,
			pressure=pressure,
		)
