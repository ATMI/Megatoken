import torch
from torch import nn


class Encoder(nn.Module):
	def __init__(
		self,
		window_size: int,
		window_step: int,

		vocab_size: int,
		embed_dim: int,
		pad_idx: int,

		head_num: int,
		encoder_layer_num: int,
		decoder_layer_num: int,
		feedforward_dim: int,
	):
		super().__init__()

		if window_step > window_size:
			raise ValueError("Window step can't be bigger that size")

		self.window_size = window_size
		self.window_step = window_step

		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=embed_dim,
			padding_idx=pad_idx,
		)

		self.encoder = nn.Transformer(
			d_model=embed_dim,
			nhead=head_num,
			num_encoder_layers=encoder_layer_num,
			num_decoder_layers=decoder_layer_num,
			dim_feedforward=feedforward_dim,
			batch_first=True,
		)

	def forward(self, src: torch.Tensor):
		b_dim, x_len = src.shape
		y_len = x_len // self.window_step

		x = self.embedding(src)
		y = torch.tensor(b_dim, y_len, self.encoder.d_model)

		for i in range(y_len):
			x_start = i * self.window_step
			x_end = x_start + self.window_size

			x_i = x[:, x_start:x_end, :]
			y_i = y[:, 0:i, :]

			pass


class Decoder(nn.Module):
	def __init__(self):
		super().__init__()


class Model(nn.Module):
	def __init__(self):
		super().__init__()


if __name__ == "__main__":
	print("Hello, World!")
