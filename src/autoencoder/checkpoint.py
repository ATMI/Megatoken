import os

import torch
from torch import optim
from torch.utils import data

from .model import AutoEncoder


class AutoEncoderCheckpoint:
	def __init__(
		self,
		model: AutoEncoder,
		optimizer: optim.Optimizer,
		scheduler: optim.lr_scheduler,
		dataloader: data.DataLoader,

		freq_step: int,
		limit: int,
	):
		self.model = model
		self.optimizer = optimizer
		self.scheduler = scheduler

		self.epoch_steps = len(dataloader)
		self.freq_step = freq_step
		self.step = 0

		self.limit = limit
		self.history = []

	def __call__(self, epoch: int, step: int):
		last_step = (epoch + 1) * self.epoch_steps - 1
		step = epoch * self.epoch_steps + step

		if (step - self.step) < self.freq_step and step != last_step:
			return
		self.step = step

		file = f"autoencoder_{epoch:2d}_{step:4d}.pth"
		state = {
			"epoch": epoch,
			"step": step,
			"model": self.model.state_dict(),
			"optimizer": self.optimizer.state_dict(),
			"scheduler": self.scheduler.state_dict(),
		}

		torch.save(state, file)
		self.history.append(file)

		while self.history and len(self.history) > self.limit:
			file = self.history.pop(0)
			try:
				os.remove(file)
			except FileNotFoundError:
				print(f"Couldn't remove old checkpoint '{file}'")
