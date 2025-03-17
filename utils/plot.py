import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def log_load(path: str | Path):
	if isinstance(path, str):
		path = Path(path)

	data = defaultdict(list)
	paths = sorted(path.glob("*.json")) if path.is_dir() else [path]

	for path in paths:
		with path.open("r") as file:
			for i, line in enumerate(file):
				row = json.loads(line)
				for k, v in row.items():
					data[k].append(v)

	return data


def main():
	# prop = "acc_"
	prop = "acc@100"
	paths = [
		("output/zeroBERT/log/train", "zero-BERT"),
		("output/crossBERT/log/train", "cross-BERT"),
	]

	plt.figure()
	for path, label in paths:
		log = log_load(path)

		y = log[prop]
		x = [i for i in range(len(y))]

		plt.plot(x, y, label=label)

	plt.legend(loc="best")
	plt.show()


if __name__ == "__main__":
	main()
