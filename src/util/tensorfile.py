import io
import os

import numpy as np
import torch
from torch import Tensor


class TensorWriter:
	def __init__(self, path: str) -> None:
		if not os.path.exists(path):
			os.makedirs(path, exist_ok=True)

		buffer = os.path.join(path, "tensor.npy")
		buffer = open(buffer, "wb")

		self.buffer = io.BufferedWriter(buffer)
		self.id2pos = []
		self.path = path

	def write(self, id: int, tensor: Tensor) -> None:
		position = self.buffer.tell()
		np.save(self.buffer, tensor.numpy())

		header = np.array([id, position], dtype=np.long)
		self.id2pos.append(header)

	def close(self) -> None:
		self.buffer.close()
		layout = os.path.join(self.path, "layout.npy")
		np.save(layout, self.id2pos)


class TensorReader:
	def __init__(self, path: str) -> None:
		buffer = os.path.join(path, "tensor.npy")
		layout = os.path.join(path, "layout.npy")

		self.buffer = open(buffer, "rb")
		self.layout = np.load(layout)
		self.id2pos = {
			header[0]: header[1]
			for header in self.layout
		}

	def by_id(self, id: int) -> Tensor:
		position = self.id2pos[id]
		return self.by_pos(position)

	def by_index(self, index: int) -> Tensor:
		pos = self.layout[index][1]
		return self.by_pos(pos)

	def by_pos(self, pos: int) -> Tensor:
		self.buffer.seek(pos)
		tensor = np.load(self.buffer)
		tensor = torch.from_numpy(tensor.copy())
		return tensor

	def close(self) -> None:
		self.buffer.close()
