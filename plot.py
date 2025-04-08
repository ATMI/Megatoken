import json
from metric import RollingMean
from collections import defaultdict
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def load(path: str):
	result = defaultdict(list)
	with open(path) as f:
		records = (json.loads(line) for line in f)
		for step, record in enumerate(records):
			result["step"].append(step)
			for k, v in record.items():
				result[k].append(v)
	return result


def dual_plot(
	x, y, z,

	y_label: str,
	z_label: str,

	y_color: str,
	z_color: str,

	y_ticks = None,
	z_ticks = None,

	y_loc: str = "lower left",
	z_loc: str = "upper right",

	x_lim: Tuple[float, float] | None = None,
	y_lim: Tuple[float, float] | None = None,
	z_lim: Tuple[float, float] | None = None,
):
	fig, ax1 = plt.subplots()
	ax1.plot(
		x, y,
		label=y_label,
		color=y_color,
	)

	ax2 = ax1.twinx()
	ax2.plot(
		x, z,
		label=z_label,
		color=z_color,
	)

	ax1.legend(loc=y_loc)
	ax2.legend(loc=z_loc)

	if y_lim is not None:
		ax1.set_ylim(*y_lim)
	if y_ticks is not None:
		ax1.set_yticks(y_ticks)

	if z_lim is not None:
		ax2.set_ylim(*z_lim)
	if z_ticks is not None:
		ax2.set_yticks(z_ticks)

	if x_lim is not None:
		ax1.set_xlim(*x_lim)
		ax2.set_xlim(*x_lim)

	plt.grid(True)
	plt.show()


def main():
	data = load("log.json")
	# data = load("log/f8b654b3502b128ee81d05ab989266b9c1456cdb.json")

	# Volume and accuracy
	# acc, volume = data["acc"], data["volume"]
	mean = RollingMean(100)
	acc, entropy = tuple(zip(*(mean(*x) for x in zip(data["acc"], data["vol"]))))

	dual_plot(
		x=data["step"],
		x_lim=(0, 32500),

		y=acc,
		y_label="accuracy",
		y_color="green",
		y_loc="lower left",
		y_lim=(0, 100),
		y_ticks=np.arange(0, 110, 10),

		# z=data["volume~"],
		z=entropy,
		z_label="volume",
		z_color="red",
		z_loc="lower right",
		z_lim=(0, 1.0),
		z_ticks=np.arange(0, 1.1, 0.1),
		# z_lim=(0, 1),
		# z_ticks=np.arange(0, 1.1, 0.1),
	)


if __name__ == "__main__":
	main()
