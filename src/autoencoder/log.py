import json
from typing import Tuple, List, Dict

from ..util.metric import RollingMean


class AutoEncoderLog:
	def __init__(self, file: str, rolling_n: int):
		self.file = open(file, "w")
		self.rolling = RollingMean(rolling_n)

	def __call__(
		self,
		acc: float,
		loss: float,
	) -> Dict[str, str]:
		log = {
			"acc": acc,
			"loss": loss
		}

		json.dump(log, self.file)
		self.file.write("\n")
		self.file.flush()

		acc, loss = self.rolling(acc, loss)
		return {
			"acc": f"{acc * 100:.2f}",
			"los": f"{loss:.2f}",
		}

	def close(self):
		self.file.close()
