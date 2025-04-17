import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor

DIM_MAX = 4
DIM_BYTES = 2
HEADER_BYTES = (DIM_MAX * DIM_BYTES)


@dataclass
class Header:
	# TODO: dtype
	shape: Tuple[int, ...]

	@property
	def size(self) -> int:
		# TODO: dtype shape
		size = 4 * math.prod(self.shape)
		return size

	@staticmethod
	def from_tensor(tensor: Tensor) -> "Header":
		shape = tuple(tensor.shape)
		return Header(
			shape=shape,
		)

	@staticmethod
	def from_bytes(data: bytes) -> "Header":
		shape = []
		for i in range(DIM_MAX):
			dim, data = data[:DIM_BYTES], data[DIM_BYTES:]
			dim = int.from_bytes(dim)
			if dim == 0:
				continue
			shape.append(dim)
		shape = tuple(shape)

		# size, data = data[:SIZE_STRIDE], data[SIZE_STRIDE:]
		# size = int.from_bytes(size)

		return Header(
			shape=shape,
		)

	def to_bytes(self) -> bytes:
		padding = DIM_MAX - len(self.shape)
		padding = (0,) * padding

		shape = self.shape + padding
		shape = (dim.to_bytes(DIM_BYTES) for dim in shape)
		shape = b''.join(shape)

		return shape


class Writer:
	def __init__(self, path: str) -> None:
		self.file = open(path, "wb")

	def write(self, tensor: Tensor) -> None:
		header = Header.from_tensor(tensor)
		header = header.to_bytes()

		tensor = tensor.numpy()
		tensor = tensor.tobytes()

		self.file.write(header)
		self.file.write(tensor)

	def close(self) -> None:
		self.file.close()


class Reader:
	def __init__(self, path: str) -> None:
		self.header = Reader.read_header(path)
		self.file = open(path, "rb")

	@staticmethod
	def read_header(path: str) -> List[Tuple[int, Header]]:
		position = 0
		headers = []

		file = open(path, "rb")
		while True:
			header = file.read(HEADER_BYTES)
			if not header:
				break

			position += HEADER_BYTES
			header = Header.from_bytes(header)
			headers.append((position, header))

			seek = header.size
			file.seek(seek, 1)
			position += seek
		file.close()

		return headers

	def read(self, index: int) -> Tensor:
		position, header = self.header[index]
		self.file.seek(position, 0)

		tensor = self.file.read(header.size)
		tensor = np.frombuffer(tensor, dtype=np.float32)
		tensor = tensor.reshape(header.shape)
		tensor = torch.from_numpy(tensor)

		return tensor

	def close(self) -> None:
		self.file.close()
