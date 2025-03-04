from dataclasses import dataclass, field
from typing import Tuple, List

import torch
from torch import nn


@dataclass
class Gate:
	stride: int = 0
	batch: List[int] = field(default_factory=list)
	x: List[int] = field(default_factory=list)
	y: List[int] = field(default_factory=list)


class AttentionGate(nn.Module):
	def __init__(self, throughput: float):
		super(AttentionGate, self).__init__()
		self.throughput = throughput

	def forward(
		self,
		pad: torch.Tensor,
		attn: torch.Tensor,
	) -> Gate:
		x_batch = attn.size(0)
		x_len = (pad.size(1) - pad.sum(dim=1)).tolist()
		gate = Gate()

		for i in range(x_batch):
			j = x_len[i]
			k = max(1, int(j * self.throughput))

			scores = attn[i, :j, :j].detach()
			scores = scores.sum(dim=0).cpu()

			_, indices = scores.topk(k=k)
			indices = indices.sort().values

			gate.stride = max(gate.stride, k)
			gate.batch += [i] * k
			gate.x += indices.tolist()
			gate.y += [i for i in range(k)]

		return gate


class GatedEncoderLayer(nn.Module):
	def __init__(
		self,
		throughput: float,
		embed_dim: int,
		heads_num: int,
		fc_dim: int,
	):
		super(GatedEncoderLayer, self).__init__()
		if throughput < 0.0 or throughput > 1.0:
			raise ValueError("The throughput should be between 0.0 and 1.0")

		self.attention = nn.MultiheadAttention(
			embed_dim=embed_dim,
			num_heads=heads_num,
			batch_first=True,
		)
		self.gate = AttentionGate(throughput)
		self.norm1 = nn.LayerNorm(embed_dim)
		self.norm2 = nn.LayerNorm(embed_dim)
		self.ff = nn.Sequential(
			nn.Linear(embed_dim, fc_dim),
			nn.SiLU(),
			nn.Dropout(),

			nn.Linear(fc_dim, embed_dim),
			nn.Dropout(),
		)

	def forward(
		self,
		src: torch.Tensor,
		src_pad: torch.Tensor,
	) -> Tuple[
		torch.Tensor,
		torch.Tensor,
	]:
		x, attn = self.attention(
			query=src,
			key=src,
			value=src,
			key_padding_mask=src_pad,
			need_weights=True,
			attn_mask=None,
			average_attn_weights=True,
			is_causal=False,
		)

		gate = self.gate(
			pad=src_pad,
			attn=attn,
		)

		y = torch.zeros(x.size(0), gate.stride, x.size(2), dtype=x.dtype, device=x.device)
		y[gate.batch, gate.y] = x[gate.batch, gate.x]

		x = torch.zeros_like(y)
		x[gate.batch, gate.y] = src[gate.batch, gate.x]

		y = self.norm1(x + y)
		y = self.norm2(y + self.ff(y))

		y_pad = torch.ones(x.size(0), gate.stride, dtype=torch.bool, device=x.device)
		y_pad[gate.batch, gate.y] = False

		return y, y_pad


class GatedEncoder(nn.Module):
	def __init__(
		self,
		throughput: List[float],
		embed_dim: int,
		heads_num: int,
		fc_dim: int,
	):
		super(GatedEncoder, self).__init__()
		self.layers = nn.ModuleList([
			GatedEncoderLayer(
				throughput=t,
				embed_dim=embed_dim,
				heads_num=heads_num,
				fc_dim=fc_dim,
			)
			for t in throughput
		])

	def forward(
		self,
		src: torch.Tensor,
		src_pad: torch.Tensor,
	):
		for layer in self.layers:
			src, src_pad = layer(src, src_pad)
		return src, src_pad
