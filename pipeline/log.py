import json
from abc import abstractmethod
from pathlib import Path
from typing import Dict

from pipeline.step import Step


class Log:
	def __init__(
		self,
		directory: str | Path,
		top_k: int = 1000,
	):
		if isinstance(directory, str):
			directory = Path(directory)

		self.directory = directory
		self.directory.mkdir(parents=True, exist_ok=True)

		# TODO: leave here or move upper?
		self.top_k = top_k
		self.losses = []
		self.losses_sum = []
		self.accuracies = []
		self.accuracies_sum = []

	def __call__(
		self,
		step: Step,
	) -> Dict:
		info = self.info(step)
		self.save(step.epoch, info)
		return info

	@abstractmethod
	def info(self, step: Step) -> Dict[str, any]:
		pass

	def save(self, epoch: int, info: Dict[str, any]):
		file = self.directory / f"{epoch}" / "log.json"
		file.parent.mkdir(parents=True, exist_ok=True)

		with file.open("a") as f:
			json.dump(info, f)
			f.write("\n")
