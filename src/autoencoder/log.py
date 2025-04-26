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
		abs_comp: float,
		rel_comp: List[float],
		loss: float,
		loss_cls: float,
		loss_vol: float,
	) -> Dict[str, str]:
		log = {
			"acc": acc,
			"abs_comp": abs_comp,
			"rel_comp": rel_comp,
			"loss": loss,
			"loss_cls": loss_cls,
			"loss_vol": loss_vol,
		}

		json.dump(log, self.file)
		self.file.write("\n")

		return {
			"acc": f"{acc:.2f}",
			"com": f"{abs_comp:.2f}",
			"cls": f"{loss_cls:.2f}",
			"vol": f"{loss_vol:.2f}",
		}

	def close(self):
		self.file.close()
