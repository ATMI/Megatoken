from operator import sub, add
from typing import Tuple, List, Sequence
from torch import Tensor
from config import Config


class RollingMean:
	def __init__(self, n: int):
		self.n = n
		self.values = []
		self.sums = None

	F = float | Sequence[float]

	@staticmethod
	def flatten(values: Tuple[F, ...]) -> List[float]:
		flat = []

		for v in values:
			if isinstance(v, float):
				flat.append(v)
			elif isinstance(v, Sequence):
				flat.extend(v)
			else:
				raise ValueError(f"Unsupported type {type(v)}")

		return flat

	@staticmethod
	def unflatten(original: Tuple[F, ...], flat: Sequence[float]) -> Tuple[F, ...]:
		i, values = 0, []

		for v in original:
			if isinstance(v, float):
				values.append(flat[i])
				i += 1
			elif isinstance(v, Sequence):
				j = i + len(v)
				values.append(tuple(flat[i:j]))
				i = j

		return tuple(values)

	def __call__(self, *values: F) -> Tuple[F, ...]:
		original = values

		values = self.flatten(values)
		self.values.append(values)

		if self.sums is None:
			self.sums = values
		else:
			self.sums = tuple(map(add, self.sums, values))

		if len(self.values) > self.n:
			values = self.values.pop(0)
			self.sums = tuple(map(sub, self.sums, values))

		n = len(self.values)
		mean = tuple(s / n for s in self.sums)
		mean = self.unflatten(original, mean)

		return mean


def accuracy(logits: Tensor, target: Tensor) -> float:
	mask = target.ne(Config.ignore_token)
	pred = logits[mask].argmax(dim=1)
	true = target[mask]

	corr = pred.eq(true)
	corr = corr.sum()
	total = true.numel()

	acc = corr / total
	acc = acc.item()

	return acc
