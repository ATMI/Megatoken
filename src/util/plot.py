import json
from collections import defaultdict
from time import time
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np

from .metric import RollingMean


def load(path: str) -> Tuple[int, Dict[str, List[float]]]:
	n, result = 0, defaultdict(list)

	with open(path) as f:
		records = (json.loads(line) for line in f)
		for record in records:
			for k, v in record.items():
				result[k].append(v)
			n += 1

	return n, result


def main():
	start = time()
	x, data = load("autoencoder.log")
	stop = time()
	print(stop - start)

	def rolling(values):
		mean = RollingMean(100)
		mean = [mean(v) for v in values]
		return mean

	x = np.arange(x)
	acc = rolling(data["acc"])
	abs_comp = rolling(data["abs_comp"])
	rel_comp = rolling(data["rel_comp"])
	rel_comp = tuple(zip(*rel_comp))

	fig, ax1 = plt.subplots()
	ax1.set_ylim(0, 1)
	ax1.set_yticks(np.arange(0, 1.1, 0.1))
	ax1.plot(
		x, acc,
		label="accuracy",
		color="k",
	)

	ax2 = ax1.twinx()
	ax2.set_ylim(0, 1)
	ax2.set_yticks(np.arange(0, 1.1, 0.1))

	for i, r in enumerate(rel_comp):
		ax2.plot(x, r, label=f"{i}")
	ax2.plot(x, abs_comp, label="total")

	ax1.set_xlim(0, 34254)
	ax2.set_xlim(0, 34254)

	# ax1.set_xlim(0, 1000)
	# ax2.set_xlim(0, 1000)

	ax1.legend(loc="lower right")
	ax2.legend(loc="lower left")

	plt.grid(True)
	plt.show()


if __name__ == "__main__":
	main()
