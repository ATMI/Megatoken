from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from pipeline.batch import Batch

T = torch.Tensor | tuple[torch.Tensor]


@dataclass
class StepResult:  # bad name :)
	batch: Batch
	pred: T
	loss: float


@dataclass
class Step:
	epoch: int
	idx: int
	num: int
	lr: any
	result: StepResult
	is_abort: bool = False

	@property
	def is_last(self) -> bool:
		return self.num == self.idx + 1


def train_step(
	model: nn.Module,
	criterion: nn.Module,
	optimizer: optim.Optimizer,
	scheduler: lr_scheduler.LRScheduler,
	batch: Batch,
):
	torch.set_grad_enabled(True)
	model.train()
	optimizer.zero_grad()

	x = batch.x
	pred = model(x)
	y = batch.y

	loss = criterion(pred.y, y)
	loss.backward()

	optimizer.step()
	scheduler.step()
	torch.cuda.empty_cache()

	return StepResult(
		batch=batch,
		pred=pred,
		loss=loss,
	)


def test_step(
	model: nn.Module,
	criterion: nn.Module,
	batch: Batch,
):
	torch.set_grad_enabled(False)
	model.eval()

	x = batch.x
	pred = model(x)

	y = batch.y
	loss = criterion(pred.y, y)
	torch.cuda.empty_cache()

	return StepResult(
		batch=batch,
		pred=pred,
		loss=loss,
	)
