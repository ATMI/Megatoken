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
	path_a = Path("checkpoint/0304_2010/train/0/0.tsv")
	path_b = Path("checkpoint/0304_2032/train/0/0.tsv")

	data_a = log_load(path_a)
	data_b = log_load(path_b)

	y_a = data_a["acc_"]
	y_b = data_b["acc_"]

	x_a = [i for i in range(len(y_a))]
	x_b = [i for i in range(len(y_b))]

	plt.figure()
	plt.plot(x_a, y_a)
	plt.plot(x_b, y_b)
	plt.show()


if __name__ == "__main__":
	main()
