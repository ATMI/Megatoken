from abc import abstractmethod
from typing import Dict

import torch


class Batch:
	@abstractmethod
	def __init__(self, batch):
		pass

	@abstractmethod
	def to(self, device: torch.device) -> "Batch":
		pass

	@property
	@abstractmethod
	def x(self) -> Dict[str, any]:
		pass

	@property
	@abstractmethod
	def y(self) -> Dict[str, any]:
		pass
