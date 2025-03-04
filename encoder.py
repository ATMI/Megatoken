from typing import Tuple, List

import torch
from torch import nn


class AttentionGate(nn.Module):
	def __init__(self, thresh: float):
		super(AttentionGate, self).__init__()
		self.thresh = thresh

	def forward(
		self,
		attn: torch.Tensor,
	) -> Tuple[
		torch.Tensor,
		torch.Tensor,
	]:
		x_gate = attn.sum(dim=1)
		x_gate = x_gate.ge(self.thresh)

		sel_num = x_gate.sum(dim=1)
		seq_len = sel_num.max().item()

		y_gate = torch.arange(seq_len, device=sel_num.device)
		y_gate = y_gate.lt(sel_num.unsqueeze(1))

		return x_gate, y_gate


class GatedEncoderLayer(nn.Module):
	def __init__(
		self,
		gate_thresh: float,
		embed_dim: int,
		heads_num: int,
		fc_dim: int,
	):
		super(GatedEncoderLayer, self).__init__()
		self.attention = nn.MultiheadAttention(
			embed_dim=embed_dim,
			num_heads=heads_num,
			batch_first=True,
		)
		self.gate = AttentionGate(gate_thresh)
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

		x_gate, y_gate = self.gate(attn)
		y = torch.zeros(y_gate.size(0), y_gate.size(1), x.size(2), dtype=x.dtype, device=x.device)
		y[y_gate] = x[x_gate]

		x = torch.zeros_like(y)
		x[y_gate] = src[x_gate]

		y = self.norm1(x + y)
		y = self.norm2(y + self.ff(y))

		y_pad = y_gate.logical_not()
		return y, y_pad


class GatedEncoder(nn.Module):
	def __init__(
		self,
		layer_tresh: List[float],
		embed_dim: int,
		heads_num: int,
		fc_dim: int,
	):
		super(GatedEncoder, self).__init__()
		self.layers = nn.ModuleList([
			GatedEncoderLayer(
				gate_thresh=gate_thresh,
				embed_dim=embed_dim,
				heads_num=heads_num,
				fc_dim=fc_dim,
			)
			for gate_thresh in layer_tresh
		])

	def forward(
		self,
		src: torch.Tensor,
		src_pad: torch.Tensor,
	):
		for layer in self.layers:
			src, src_pad = layer(src, src_pad)
		return src, src_pad
