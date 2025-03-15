from abc import abstractmethod
from typing import Dict


class Batch:
	@abstractmethod
	def __init__(self, batch):
		pass

	@property
	@abstractmethod
	def x(self) -> Dict[str, any]:
		pass

	@property
	@abstractmethod
	def y(self) -> Dict[str, any]:
		pass
