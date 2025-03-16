import json
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Tuple

from pipeline.base.step import Step

ConsoleLog = Dict[str, str]
FileLog = Dict[str, any]


class Log:
	def __init__(
		self,
		directory: str | Path,
	):
		if isinstance(directory, str):
			directory = Path(directory)

		self.directory = directory
		self.directory.mkdir(parents=True, exist_ok=True)

	def __call__(self, step: Step) -> Dict:
		console, file = self.info(step)
		self.save(step.epoch, file)
		return console

	@abstractmethod
	def info(self, step: Step) -> Tuple[ConsoleLog, FileLog]:
		pass

	def save(self, epoch: int, info: Dict[str, any]):
		file = self.directory / f"{epoch}" / "log.json"
		file.parent.mkdir(parents=True, exist_ok=True)

		with file.open("a") as f:
			json.dump(info, f)
			f.write("\n")
