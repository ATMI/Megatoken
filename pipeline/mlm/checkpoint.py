import time
from pathlib import Path
from typing import Optional

from torch import nn, optim
from torch.optim import lr_scheduler

from pipeline.base.checkpoint import Checkpoint
from pipeline.base.step import Step


class MaskModelCheckpoint(Checkpoint):
	def __init__(
		self,
		directory: Path,
		loss_thresh: float = 0.05,
		time_interval: int = 20,
		step_interval: int = 1000,
	):
		"""

		:param directory: Directory to save checkpoints to.
		:param loss_thresh: Threshold for saving checkpoints based on loss difference
		:param time_interval: Interval (mins) for saving checkpoints
		:param step_interval: Interval in steps for saving checkpoints
		"""
		super().__init__(directory)

		self.loss_thresh = loss_thresh
		self.time_interval = time_interval
		self.step_interval = step_interval

		self.prev_loss = None
		self.timestamp = time.time()

	def save(
		self,
		model: nn.Module,
		optimizer: optim.Optimizer,
		scheduler: lr_scheduler.LRScheduler,
		step: Step,
	) -> None:
		super().save(model, optimizer, scheduler, step)

		self.timestamp = time.time()
		self.prev_loss = step.loss

	def condition(self, step: Optional[Step]) -> bool:
		# Keyboard Interruption
		# Last step in the epoch
		if step is None or step.is_last:
			return True

		# FIXME: maybe its better to remove steps condition.
		# Checkpoint conditions for steps and time may overlap.
		# Time sounds more reasonable condition for checkpoint.

		# Save each N steps
		if step.curr % self.step_interval == 0:
			return True

		# Save each T minutes
		if self.timestamp - time.time() >= self.time_interval * 60:
			return True

		# FIXME: this condition seems to fire only in the beginning when the loss drop is significant
		# Idx what to do with it)))
		# Actually I like idea with time, didn't come to my mind before))
		# Maybe it's enough

		# Also, this class is pretty based.
		# It can be moved somewhere)

		# Conditions based on previous checkpoints
		if self.prev_loss is not None:
			# Save if loss dropped greatly after last checkpoint
			loss_vel = self.prev_loss - step.loss
			if loss_vel >= self.loss_thresh:
				return True

		return False
