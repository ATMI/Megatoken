import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data
from tqdm import tqdm

from pipeline.base.batch import Batch
from pipeline.base.checkpoint import Checkpoint
from pipeline.base.log import Log
from pipeline.base.step import train_step, test_step, Step


def train_epoch(
	epoch: int,
	model: nn.Module,
	device: torch.device,

	criterion: nn.Module,
	optimizer: optim.Optimizer,
	scheduler: lr_scheduler.LRScheduler,

	loader: data.DataLoader[Batch],
	checkpoint: Checkpoint,
	log: Log,
):
	step_num = len(loader)
	progress_bar = tqdm(total=step_num, desc=f"Train {epoch}")

	for step, batch in enumerate(loader):
		batch = batch.to(device)

		try:
			pred, loss = train_step(
				model=model,
				criterion=criterion,
				optimizer=optimizer,
				scheduler=scheduler,
				batch=batch,
			)
		except KeyboardInterrupt as _:
			checkpoint(model, optimizer, scheduler, None)
			break

		step = Step(
			epoch=epoch,
			curr=step,
			last=step_num,
			batch=batch,
			pred=pred,
			loss=loss,
			lr=scheduler.get_last_lr(),
		)

		info = log(step)
		progress_bar.set_postfix(**info)
		progress_bar.update(1)

		checkpoint(model, optimizer, scheduler, step)

	progress_bar.close()


def test_epoch(
	epoch: int,
	model: nn.Module,
	device: torch.device,

	criterion: nn.Module,

	loader: data.DataLoader[Batch],
	log: Log,
):
	step_num = len(loader)
	progress_bar = tqdm(total=step_num, desc=f"Test  {epoch}")

	for step, batch in enumerate(loader):
		batch = batch.to(device)

		pred, loss = test_step(
			model=model,
			criterion=criterion,
			batch=batch,
		)

		step = Step(
			epoch=epoch,
			curr=step,
			last=step_num,
			batch=batch,
			pred=pred,
			loss=loss,
			lr=None,
		)

		info = log(step)
		progress_bar.set_postfix(**info)
		progress_bar.update(1)

	progress_bar.close()
