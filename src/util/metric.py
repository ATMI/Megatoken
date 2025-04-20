from dataclasses import dataclass
from operator import sub, add
from typing import Tuple, List, Sequence, Optional
from torch import Tensor


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

		if len(values) > 1:
			values = tuple(values)
		else:
			values = values[0]

		return values

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


def accuracy(
	logits: Tensor,
	target: Tensor,
	ignore: Optional[int] = None,
) -> float:
	if ignore is not None:
		mask = target.ne(ignore)
		logits = logits[mask]
		target = target[mask]

	pred = logits.argmax(dim=1)
	corr = pred.eq(target)
	corr = corr.sum()
	total = target.numel()

	acc = corr / total
	acc = acc.item()

	return acc


@dataclass
class Confusion:
	tp: int
	fp: int
	fn: int
	total: int

	@property
	def precision(self) -> float:
		p = self.tp + self.fp
		return self.tp / p if p > 0 else 0

	@property
	def recall(self) -> float:
		p = self.tp + self.fn
		return self.tp / p if p > 0 else 0

	@property
	def accuracy(self) -> float:
		return 1 - (self.fp + self.fn) / self.total


def confusion(
	logits: Tensor,
	target: Tensor,
) -> Confusion:
	pred = logits > 0

	tp = ((pred == 1) & (target == 1)).sum().item()
	fp = ((pred == 1) & (target == 0)).sum().item()
	fn = ((pred == 0) & (target == 1)).sum().item()
	total = pred.size(0)

	return Confusion(tp=tp, fp=fp, fn=fn, total=total)
