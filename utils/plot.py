import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def log_load(path: str | Path, start: int = 0, end: int = -1):
	if isinstance(path, str):
		path = Path(path)

	data = defaultdict(list)
	paths = sorted(path.glob("*.json")) if path.is_dir() else [path]

	for path in paths:
		with path.open("r") as file:
			for i, line in enumerate(file):
				if i < start:
					continue
				if 0 < end <= i:
					break
				row = json.loads(line)
				for k, v in row.items():
					data[k].append(v)

	return data


def main():
	start = 1000
	end = 8000
	prop = "acc@100"
	paths = [
		("output/log/zeroBERT0/train/0.json", "zero-BERT0"),
		("output/log/zeroBERT1/train/0.json", "zero-BERT1"),
		("output/log/crossBERT0/train/0.json", "cross-BERT0"),
		("output/log/crossBERT1/train/0.json", "cross-BERT1"),
	]

	plt.figure()
	for path, label in paths:
		log = log_load(path, start, end)

		y = log[prop]
		x = [start + i for i in range(len(y))]

		plt.plot(x, y, label=label)

	plt.legend(loc="best")
	plt.show()


if __name__ == "__main__":
	main()
