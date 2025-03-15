from typing import Tuple

import torch
from torch import nn


class SoftGate(nn.Module):
	def __init__(
		self,
		embed_dim: int,
	) -> None:
		super(SoftGate, self).__init__()
		bottleneck_dim = max(2, embed_dim // 2)
		self.threshold = 1e-1
		self.fc = nn.Sequential(
			nn.Linear(embed_dim, bottleneck_dim),
			nn.ReLU(),
			nn.Linear(bottleneck_dim, 1),
		)

	def forward(
		self,
		x: torch.Tensor,
		pad: torch.Tensor,
	) -> Tuple[
		torch.Tensor,
		torch.Tensor,
	]:
		batch = x.size(0)
		length = pad.size(1) - pad.sum(dim=1)
		stride = length.max().item()

		y = self.fc(x)
		y = (1 + torch.tanh(10 * y)) / 2

		for i in range(batch):
			n = length[i].item()
			if n == stride:
				continue
			y[i, n:] = 0

		adjust = 1e-5 + self.threshold
		adjust = adjust - y.max(dim=1).values
		adjust = adjust.clamp(min=0).unsqueeze(dim=-1)

		y = y + adjust
		u = x * y
		y = y.squeeze(dim=-1)

		u_mask = (y > self.threshold) & (~pad)
		length = u_mask.sum(dim=1)
		stride = length.max().item()

		size = list(u.size())
		size[1] = stride

		v = torch.zeros(size, dtype=u.dtype, device=u.device)
		v_mask = torch.arange(stride, device=u.device)
		v_mask = v_mask < length.unsqueeze(dim=-1)

		v[v_mask] = u[u_mask]
		v_pad = ~v_mask

		return v, v_pad


class GatedEncoderLayer(nn.Module):
	def __init__(
		self,
		model_dim: int,
		head_num: int,
		fc_dim: int,
	):
		super(GatedEncoderLayer, self).__init__()
		self.encoder = nn.TransformerEncoderLayer(
			d_model=model_dim,
		)

	def forward(
		self,
		x: torch.Tensor,
		pad: torch.Tensor,
	) -> Tuple[
		torch.Tensor,
		torch.Tensor,
	]:
		y, attn = self.attention(
			query=x,
			key=x,
			value=x,
			key_padding_mask=pad,
			need_weights=False,
			attn_mask=None,
			average_attn_weights=False,
			is_causal=False,
		)

		y = self.norm1(y + x)
		y = self.norm2(y + self.ff(y))
		y, y_pad = self.gate(y, pad)

		return y, y_pad


class GatedEncoder(nn.Module):
	def __init__(
		self,
		gates_num: int,
		embed_dim: int,
		heads_num: int,
		fc_dim: int,
	):
		super(GatedEncoder, self).__init__()
		self.layers = nn.ModuleList([
			GatedEncoderLayer(
				embed_dim=embed_dim,
				head_num=heads_num,
				fc_dim=fc_dim,
			)
			for _ in range(gates_num)
		])

	def forward(
		self,
		src: torch.Tensor,
		src_pad: torch.Tensor,
	):
		for layer in self.layers:
			src, src_pad = layer(src, src_pad)
		return src, src_pad


def main():
	torch.autograd.set_detect_anomaly(True)

	encoder = GatedEncoder(
		gates_num=4,
		embed_dim=16,
		heads_num=4,
		fc_dim=16,
	)

	x = torch.randn(8, 10, 16, requires_grad=True)
	pad = torch.zeros(8, 10, dtype=torch.bool)

	y, _ = encoder(x, pad)
	y.sum().backward()

	print(x.grad)


if __name__ == "__main__":
	main()
