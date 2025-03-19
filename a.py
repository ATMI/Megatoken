from typing import Tuple

import torch
from torch import nn, Tensor


class Gate(nn.Module):
	def __init__(self, c: float, temperature: float, hard: bool):
		super(Gate, self).__init__()
		self.c = c
		self.temperature = temperature
		self.hard = hard

	@staticmethod
	def sample_gumbel(shape, eps=1e-20):
		"""Sample from Gumbel(0,1)"""
		U = torch.rand(shape)
		return -torch.log(-torch.log(U + eps) + eps)

	@staticmethod
	def gumbel_sigmoid(logits, temperature=1.0, hard=False):
		"""Apply Gumbel noise and sigmoid"""
		gumbel_noise = Gate.sample_gumbel(logits.size())
		gumbel_noise = gumbel_noise.to(logits.device)
		noisy_logits = (logits + gumbel_noise) / temperature

		# Apply sigmoid
		output = torch.sigmoid(noisy_logits)

		if hard:
			# Discretize output using straight-through estimator
			output_hard = (output > 0.5).float()
			return output_hard - output.detach() + output
		else:
			return output

	def forward(self, x: Tensor, x_pad: Tensor) -> Tuple[Tensor, Tensor]:
		length = x.size(1)

		alpha = x[:, :, 0]
		alpha = Gate.gumbel_sigmoid(alpha - self.c, self.temperature, self.hard)
		omega = x[:, :, 1].sigmoid()

		y = [x[:, 0]]
		z = [omega[:, 0]]

		for i in range(1, length):
			alpha_i = alpha[:, i]
			omega_i = omega[:, i]

			z_i_1 = z[i - 1]
			z_i = z_i_1 * alpha_i + omega_i
			z.append(z_i)

			alpha_i = alpha_i.unsqueeze(1)
			omega_i = omega_i.unsqueeze(1)
			z_i_1 = z_i_1.unsqueeze(1)
			z_i = z_i.unsqueeze(1)

			y_i_1 = y[i - 1]
			x_i = x[:, i]

			y_i = (alpha_i * y_i_1 * z_i_1 + x_i * omega_i) / z_i
			y.append(y_i)

		y = [i.unsqueeze(1) for i in y]
		y = torch.cat(y, dim=1)

		return y, x_pad


def main():
	torch.random.manual_seed(42)
	batch_size = 2
	embed_dim = 16
	seq_len = 128

	x = torch.randn(
		(batch_size, seq_len, embed_dim),
		dtype=torch.float,
		requires_grad=True,
	)
	x_pad = torch.tensor(
		[[0] * (seq_len - i) + [1] * i for i in range(batch_size)],
		dtype=torch.bool,
	)

	torch.autograd.set_detect_anomaly(True)
	gate = Gate(c=0, temperature=0.1, hard=True)
	y, y_pad = gate(x, x_pad)

	y.sum().backward()
	print(x.grad)


if __name__ == "__main__":
	main()
