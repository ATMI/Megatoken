from dataclasses import dataclass
from typing import Tuple, Callable, Optional

import torch
from torch import Tensor
from torch import nn

from positional import PositionalEncoding


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
		batch_size = inputs.size(0)
		input_length = inputs.size(1)

		valves = (inputs[:, :, 0] + self.bias) / self.temperature
		valves = valves.sigmoid()

		mask_vals = valves.log()  # batch, input_length
		mask_vals = mask_vals.unsqueeze(1)  # batch, 1, input_length

		mask = torch.zeros((batch_size, input_length, input_length), device=device)
		mask[:] = mask_vals[:]

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
		vocab_size: int,
		pad_token: int,
		max_length: int,

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
		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=model_dim,
			padding_idx=pad_token,
		)
		self.positional = PositionalEncoding(
			model_dim=model_dim,
			max_len=max_length,
		)
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
		tokens: Tensor,
		pad_mask: Tensor,
		attn_mask: Optional[Tensor],
	) -> Outputs:
		device = tokens.device
		batch_size = tokens.size(0)

		input_mask = ~pad_mask
		input_length = tokens.size(1)
		inputs = self.embedding(tokens)
		inputs = self.positional(inputs)

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
