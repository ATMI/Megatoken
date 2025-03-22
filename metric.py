from operator import sub, add
from typing import Tuple
from torch import Tensor

from config import Config


class RollingMean:
	def __init__(self, n: int):
		self.n = n
		self.values = []
		self.sums = None

	def __call__(self, *values: float) -> Tuple[float, ...]:
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

		return mean


def accuracy(filled: Tensor, target: Tensor) -> float:
	mask = target.ne(Config.ignore_token)
	pred = filled[mask].argmax(dim=1)
	true = target[mask]

	corr = pred.eq(true)
	corr = corr.sum()
	total = true.numel()

	acc = corr / total
	acc = acc.item()

	return acc
