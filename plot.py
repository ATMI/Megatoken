import json
from collections import defaultdict

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

	plt.show()


def main():
	data = load("log.json")

	# Volume and accuracy
	dual_plot(
		x=data["step"],

		y=data["acc~"],
		y_label="accuracy",
		y_color="green",
		y_loc="lower left",

		# z=[math.sqrt(v) for v in data["volume~"]],
		z=data["ratio~"],
		z_label="ratio",
		z_color="red",
		z_loc="upper right",
	)

	# Class and volume losses
	"""
	dual_plot(
		x=data["step"],

		y=data["class~"],
		y_label="class",
		y_color="green",

		z=data["volume~"],
		z_label="volume",
		z_color="red",
	)

	# Loss and accuracy
	dual_plot(
		x=data["step"],

		y=data["loss~"],
		y_label="loss",
		y_color="green",

		z=data["acc~"],
		z_label="accuracy",
		z_color="red",
	)
	"""


if __name__ == "__main__":
	main()
