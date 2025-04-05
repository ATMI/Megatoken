import json
import math
from collections import defaultdict
from typing import Tuple

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

	if z_lim is not None:
		ax2.set_ylim(*z_lim)

	if x_lim is not None:
		ax1.set_xlim(*x_lim)
		ax2.set_xlim(*x_lim)

	plt.show()


def main():
	# data = load("log/65ce082a56d48fc41c065e382e3513e4c5917dcf.json")
	data = load("log.json")

	# Volume and accuracy
	dual_plot(
		x=data["step"],
		x_lim=(0, 30000),

		y=data["acc~"],
		y_label="accuracy",
		y_color="green",
		y_loc="lower left",
		y_lim=(0, 100),

		# z=[math.sqrt(v) for v in data["volume~"]],
		z=data["volume~"],
		z_label="volume",
		z_color="red",
		z_loc="lower right",
		z_lim=(0, 1),
	)


if __name__ == "__main__":
	main()
