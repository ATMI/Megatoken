from abc import abstractmethod
from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from pipeline.base.step import Step


class Checkpoint:
	def __init__(
		self,
		directory: str | Path,
	):
		if isinstance(directory, str):
			directory = Path(directory)

		self.directory = directory
		self.directory.mkdir(parents=True, exist_ok=True)

	@abstractmethod
	def condition(self, step: Optional[Step]) -> bool:
		pass

	def save(
		self,
		model: nn.Module,
		optimizer: optim.Optimizer,
		scheduler: lr_scheduler.LRScheduler,
		step: Step,
	) -> None:
		path = self.directory / f"{step.epoch}" / f"{step.curr}.ckpt"
		path.parent.mkdir(parents=True, exist_ok=True)

		state = {
			"epoch": step.epoch,
			"step": step.curr,
			"model": model.state_dict(),
			"optimizer": optimizer.state_dict(),
			"scheduler": scheduler.state_dict(),
		}
		torch.save(state, path)

	def __call__(
		self,
		model: nn.Module,
		optimizer: optim.Optimizer,
		scheduler: lr_scheduler.LRScheduler,
		step: Optional[Step],
	) -> bool:
		if not self.condition(step):
			return False

		self.save(model, optimizer, scheduler, step)
		return True
