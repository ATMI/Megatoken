from typing import Dict

from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data

from pipeline.batch import Batch
from pipeline.checkpoint import Checkpoint
from pipeline.epoch import Step
from pipeline.log import Log
from pipeline.train import train


class MLMBatch(Batch):

	@staticmethod
	def collate_x(batch):
		pass

	@staticmethod
	def collate_y(batch):
		pass


class MLMCheckpoint(Checkpoint):

	def condition(self, step: Step) -> bool:
		return step.is_last or ...


class MLMLog(Log):

	def info(self, step: Step) -> Dict:
		pass


def main():
	epochs = 2
	train_batch = 64
	test_batch = 64
	lr = 0.001

	directory = ""
	dataset = {}
	model = nn.Module()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.LinearLR(optimizer)

	train_loader = data.DataLoader(
		dataset=dataset["train"],
		batch_size=train_batch,
		shuffle=True,
		collate_fn=MLMBatch,
	)

	test_loader = data.DataLoader(
		dataset=dataset["test"],
		batch_size=test_batch,
		shuffle=False,
		collate_fn=MLMBatch,
	)

	log = MLMLog(directory)
	checkpoint = MLMCheckpoint(directory)

	train(
		epochs=epochs,
		model=model,

		criterion=criterion,
		optimizer=optimizer,
		scheduler=scheduler,

		train_loader=train_loader,
		test_loader=test_loader,

		log=log,
		checkpoint=checkpoint,
	)


if __name__ == "__main__":
	main()
