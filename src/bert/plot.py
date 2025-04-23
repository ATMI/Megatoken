import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from src.util.metric import RollingMean


def load(path: str):
	n, result = 0, defaultdict(list)

	with open(path) as f:
		records = (json.loads(line) for line in f)
		for record in records:
			for k, v in record.items():
				result[k].append(v)
			n += 1

	return n, result


def get_metrics(filename):
	def rolling(values):
		mean = RollingMean(100)
		mean = [mean(v) for v in values]
		return mean

	_, data = load(filename)

	x = 10000

	acc = rolling(data["acc"][:x])
	pr = rolling(data["precision"][:x])
	rc = rolling(data["recall"][:x])
	loss = rolling(data["loss"][:x])

	x = np.arange(x)
	metrics = {
		'acc': [d for d in acc],
		'loss': [d for d in loss],
		'precision': [d for d in pr],
		'recall': [d for d in rc]
	}

	return x, metrics


def main():
	x, cota_metrics = get_metrics("../../cls-log.json")
	x, bert_metrics = get_metrics("../../bert-logs.json")

	fig, ax1 = plt.subplots()

	ax1.plot(x, bert_metrics['acc'], label='BERT Acc', color="green")
	ax1.plot(x, cota_metrics['acc'], label='CoTA Acc', color="orange")
	# ax1.plot(x, metrics['precision'], label='Precision', color="b")
	# ax1.plot(x, metrics['recall'], label='Recall', color="orange")

	# ax1.set_xlabel('Training Steps')
	ax1.set_ylabel('Score')
	ax1.tick_params(axis='y', labelcolor='k')
	ax1.set_ylim(0.1, 1.1)
	ax1.set_xticks(np.arange(0.0, 10001, 1000))
	ax1.set_yticks(np.arange(0.1, 1.2, 0.1))

	ax2 = ax1.twinx()
	ax2.plot(x, cota_metrics['loss'], label='CoTA Loss', color="blue")
	ax2.plot(x, bert_metrics['loss'], label='BERT Loss', color="red")
	ax2.set_ylabel('Loss')
	ax2.tick_params(axis='y')

	fig.legend(loc='upper right')
	ax2.set_yticks(np.arange(0.2, 1.1, 0.1))

	ax1.grid(True)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
