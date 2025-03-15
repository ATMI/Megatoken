from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from pipeline.batch import Batch


@dataclass
class Step:
	epoch: int
	curr: int
	last: int

	batch: Batch
	pred: any
	loss: float
	lr: any

	@property
	def is_last(self) -> bool:
		return self.last == self.curr + 1


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

	loss = criterion(pred, y)
	loss.backward()

	optimizer.step()
	scheduler.step()
	torch.cuda.empty_cache()

	return pred, loss


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
	loss = criterion(pred, y)
	torch.cuda.empty_cache()

	return pred, loss
