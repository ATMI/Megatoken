from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data
from tqdm import tqdm

from pipeline.batch import Batch
from pipeline.checkpoint import Checkpoint
from pipeline.log import Log
from pipeline.step import train_step, test_step


def train_epoch(
	epoch: int,
	model: nn.Module,

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
		result = train_step(
			model=model,
			criterion=criterion,
			optimizer=optimizer,
			scheduler=scheduler,
			batch=batch,
		)

		step = Step(
			epoch=epoch,
			idx=step,
			num=step_num,
			lr=scheduler.get_last_lr(),
			result=result,
		)

		info = log(step)
		progress_bar.set_postfix(**info)
		progress_bar.update(1)

		checkpoint(model, optimizer, scheduler, step)

	progress_bar.close()


def test_epoch(
	epoch: int,
	model: nn.Module,

	criterion: nn.Module,

	loader: data.DataLoader[Batch],
	log: Log,
):
	step_num = len(loader)
	progress_bar = tqdm(total=step_num, desc=f"Test  {epoch}")

	for step, batch in enumerate(loader):
		result = test_step(
			model=model,
			criterion=criterion,
			batch=batch,
		)

		step = Step(
			epoch=epoch,
			idx=step,
			num=step_num,
			lr=None,
			result=result,
		)

		info = log(step)
		progress_bar.set_postfix(**info)
		progress_bar.update(1)

	progress_bar.close()
