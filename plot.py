import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def log_load(path: Path):
	data = defaultdict(list)

	with path.open("r") as file:
		reader = csv.DictReader(file, delimiter="\t")
		for row in reader:
			for k, v in row.items():
				data[k].append(float(v))

	return data


def main():
	prop = "acc_"
	# prop = "ratio"
	paths = [
		# "checkpoint/0305_1128/train/0/0.tsv",

		("checkpoint/0305_1514/train/0/0.tsv", "vanilla"), # Vanilla
		("checkpoint/0305_1452/train/0/0.tsv", "0.68 0.95 0.99"), # 0.68 0.95 0.99
	]

	plt.figure()
	for path, label in paths:
		path = Path(path)
		data = log_load(path)

		y = data[prop] #[1000:]
		x = [i for i in range(len(y))]

		plt.plot(x, y, label=label)

	plt.legend(loc="best")
	plt.show()


if __name__ == "__main__":
	main()
