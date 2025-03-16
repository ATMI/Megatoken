from pathlib import Path
from typing import Tuple

import math

from pipeline.base.log import Log, ConsoleLog, FileLog
from pipeline.base.step import Step


class MaskModelLog(Log):

	def __init__(
		self,
		directory: str | Path,
		k: int = 1000,
	):
		super().__init__(directory)

		self.k = k

		self.losses = []
		self.losses_sum = 0

		self.accuracies = []
		self.accuracies_sum = 0

	def info(self, step: Step) -> Tuple[ConsoleLog, FileLog]:
		acc, acc_k = self.accuracy(step)
		ppl = self.perplexity(step.loss)
		loss_k = self.k_loss(step.loss)

		console = {
			f"loss@{self.k}": f"{loss_k:.3f}",
			f"acc@{self.k}": f"{acc_k * 100:.2f}",
			"ppl": f"{ppl:.2f}",
			"ratio": f"{step.pred.ratios[-1] * 100:.2f}",
			"loss": f"{step.loss:.3f}",
			"acc": f"{acc * 100:.2f}",
		}

		file = {
			f"loss@{self.k}": loss_k,
			f"acc@{self.k}": acc_k,
			"ppl": ppl,
			"ratios": step.pred.ratios,
			"loss": step.loss,
			"acc": acc,
		}

		return console, file

	def accuracy(self, step: Step) -> Tuple[float, float]:
		out = step.pred.y
		# TODO: fix access for label and ignore_token
		label = step.batch.y
		ignore_index = step.batch.ignore_token

		y_pred = out.argmax(dim=-1)
		correct = (y_pred == label).sum().item()

		# Number of tokens to predict
		num_labels = (label != ignore_index).sum().item()
		acc = correct / num_labels

		# Calculate Accuracy@k
		self.accuracies.append(acc)
		self.accuracies_sum += acc

		if len(self.accuracies) > self.k:
			a = self.accuracies.pop(0)
			self.accuracies_sum -= a

		acc_avg = self.accuracies_sum / len(self.accuracies)
		return acc, acc_avg

	def k_loss(self, loss: float) -> float:
		"""
		Calculates k loss
		:param loss: Loss at step
		:return: Average loss over K steps
		"""
		self.losses.append(loss)
		self.losses_sum += loss

		if len(self.losses) > self.k:
			l = self.losses.pop(0)
			self.losses_sum -= l

		loss_avg = self.losses_sum / len(self.losses)
		return loss_avg

	def perplexity(self, loss) -> float:
		return math.exp(loss)
