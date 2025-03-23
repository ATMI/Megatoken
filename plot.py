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


def plot_losses(step, class_loss, valve_loss):
	fig, ax1 = plt.subplots()
	ax1.plot(
		step, class_loss,
		label="class",
		color="green",
	)

	ax2 = ax1.twinx()
	ax2.plot(
		step, valve_loss,
		label="valve",
		color="red",
	)

	ax1.legend(loc="lower left")
	ax2.legend(loc="upper right")

	plt.show()


def plot_accuracy(step, loss, accuracy):
	fig, ax1 = plt.subplots()
	ax1.plot(
		step, loss,
		label="loss",
		color="green",
	)

	ax2 = ax1.twinx()
	ax2.plot(
		step, accuracy,
		label="accuracy",
		color="red",
	)

	ax1.legend(loc="lower left")
	ax2.legend(loc="upper right")

	plt.show()


def main():
	data = load("log/1.json")

	plot_accuracy(
		step=data["step"],
		loss=data["loss~"],
		accuracy=data["acc~"],
	)

	plot_losses(
		step=data["step"],
		class_loss=data["cls~"],
		valve_loss=data["valve~"],
	)


if __name__ == "__main__":
	main()
