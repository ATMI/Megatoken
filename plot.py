import json
from collections import defaultdict
from time import time
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np

from metric import RollingMean


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
	x, data = load("log.json")
	# x, data = load("log/a4d2a56656e1a1e4233a3d830cc0617205f467ed.json")
	stop = time()
	print(stop - start)

	def rolling(values):
		mean = RollingMean(100)
		mean = [mean(v) for v in values]
		return mean

	x = np.arange(x)
	acc = rolling(data["acc"])
	comp = rolling(data["comp"])
	ratio = rolling(data["rat"])
	ratio = tuple(zip(*ratio))

	fig, ax1 = plt.subplots()
	ax1.set_ylim(0, 100)
	ax1.set_yticks(np.arange(0, 110, 10))
	ax1.plot(
		x, acc,
		label="accuracy",
		color="g",
	)

	ax2 = ax1.twinx()
	ax2.set_ylim(0, 1)
	ax2.set_yticks(np.arange(0, 1.1, 0.1))

	# ratio = list(zip((*ratio, comp), "cmykcmykr", ["gate 0", "gate 1", "gate 2", "gate 3", "gate 5", "gate 6", "gate 7", "comp"]))
	# for r, color, label in ratio:
	# 	ax2.plot(
	# 		x, r,
	# 		label=label,
	# color=color,
	# )
	for r in ratio:
		ax2.plot(x, r)
	ax2.plot(x, comp)

	ax1.set_xlim(0, 32500)
	ax2.set_xlim(0, 32500)

	ax1.legend(loc="lower left")
	# ax2.legend(loc="upper right")

	plt.grid(True)
	plt.show()


if __name__ == "__main__":
	main()
