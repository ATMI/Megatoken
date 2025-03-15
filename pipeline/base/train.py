from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data

from pipeline.base.batch import Batch
from pipeline.base.checkpoint import Checkpoint
from pipeline.base.log import Log
from pipeline.base.epoch import train_epoch, test_epoch


def train(
	epochs: int,
	model: nn.Module,

	criterion: nn.Module,
	optimizer: optim.Optimizer,
	scheduler: lr_scheduler.LRScheduler,

	train_loader: data.DataLoader[Batch],
	test_loader: data.DataLoader[Batch],

	log: Log,
	checkpoint: Checkpoint,
):
	for epoch in range(epochs):
		train_epoch(
			epoch=epoch,

			model=model,
			criterion=criterion,
			optimizer=optimizer,
			scheduler=scheduler,

			loader=train_loader,
			log=log,
			checkpoint=checkpoint,
		)

		test_epoch(
			epoch=epoch,

			model=model,
			criterion=criterion,

			loader=test_loader,
			log=log,
		)
