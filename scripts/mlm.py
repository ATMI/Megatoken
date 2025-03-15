import argparse
import time
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn, optim
from torch.utils import data
from transformers import get_scheduler
from torch.nn.utils import rnn

from model.transformer import GatedTransformer
from pipeline.batch import Batch
from pipeline.checkpoint import Checkpoint
from pipeline.log import Log
from pipeline.step import Step
from pipeline.train import train
from scripts.dataset import prepare_dataset
from utils.config import load_config


class MaskModelBatch(Batch):

	def __init__(
		self,
		batch,
		pad_index: int,
		mask_index: int,
		causal_mask: bool = True,
		mask_prob: float = 0.15,
		ignore_index: int = -100,
	):
		self.batch, self.pad_mask = self.collate(batch)
		self.pad_index = pad_index
		self.mask_token_id = mask_index
		self.ignore_index = ignore_index
		self.mask_prob = mask_prob
		self.causal_mask = causal_mask

		self.masked_x = None
		self.label = None
		self.attn_mask = None

	def collate(self, batch):
		x = [torch.tensor(x["tokens"]) for x in batch]
		x = rnn.pad_sequence(x, batch_first=True, padding_value=self.pad_index)

		batch_size = x.size(0)
		seq_len = x.size(1)

		x_pad = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
		for i in range(batch_size):
			tokens = batch[i]["tokens"]
			length = len(tokens)
			x_pad[i, :length] = False

		return x, x_pad

	# TODO: make wider masks and mode
	def masking(self):
		x_lens = self.pad_mask.size(1) - self.pad_mask.sum(dim=1)
		num_to_mask = (self.mask_prob * x_lens).int().clamp(min=1)

		mask = torch.zeros_like(self.batch, dtype=torch.bool)

		for i in range(self.batch.size(0)):
			candidates = torch.where(~self.pad_mask[i])[0]
			perm = torch.randperm(x_lens[i])
			selected = candidates[perm[:num_to_mask[i]]]
			mask[i, selected] = True

		masked_input = self.batch.clone()
		masked_input[mask] = self.mask_token_id

		labels = torch.full_like(self.batch, fill_value=self.ignore_index)  # -100 = ignore index for CrossEntropyLoss
		labels[mask] = self.batch[mask]

		return masked_input, labels

	@property
	def x(self) -> Dict[str, any]:
		self.masked_x, self.label = self.masking()

		attn_mask = None
		if self.causal_mask:
			attn_mask = nn.Transformer.generate_square_subsequent_mask(self.batch.size(1), dtype=torch.bool)

		inputs = {
			"x": self.batch,
			"mask_x": self.masked_x,
			"pad": self.pad_mask,
			"attn": attn_mask
		}

		return inputs

	@property
	def y(self) -> torch.Tensor:
		"""
		Shape: [batch_size * seq_len]
		:return: RETURN TENSOR FOR CROSS-ENTROPY-LOSS
		"""
		flat_label = self.label.flatten()
		return flat_label


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
		self.timestamp = None

	def condition(self, step: Step) -> bool:
		save_condition = False

		# Keyboard Interruption
		if step is None:
			save_condition = True

		# Save each N steps
		if step.curr % self.step_interval == 0:
			save_condition = True

		# Conditions based on previous checkpoints
		if self.prev_loss is not None:
			# Save if loss dropped greatly after last checkpoint
			loss_vel = self.prev_loss - step.loss
			if loss_vel >= self.loss_thresh:
				save_condition = True

			# Save each T minutes
			if self.timestamp - time.time() >= self.time_interval * 60:
				save_condition = True

		if save_condition:
			self.timestamp = time.time()
			self.prev_loss = step.loss

		return step.is_last or save_condition


class MaskModelLog(Log):

	def __init__(
		self,
		directory: str | Path,
		top_k: int = 1000,
	):
		super().__init__(directory)

		self.top_k = top_k

		self.losses = []
		self.losses_sum = []

		self.accuracies = []
		self.accuracies_sum = []

	def info(self, step: Step) -> Dict[str, any]:
		acc, acc_ = self.accuracy(step)
		ppl = self.perplexity(step.loss)
		loss_ = self.topk_loss(step.loss)

		logs = {
			"loss": step.loss,
			"loss@K": loss_,
			"acc": acc,
			"acc@K": acc_,
			"ratios": step.pred[2],
			"PPL": ppl,
		}

		return logs

	def accuracy(self, step: Step) -> Tuple[float, float]:
		out = step.pred
		# TODO: fix access for label and ignore_index
		label = step.batch.label
		ignore_index = step.batch.ignore_index

		y_pred = out.argmax(dim=-1)
		correct = (y_pred == label).sum().item()
		num_labels = (label != ignore_index).sum().item()  # Number of tokens to predict

		acc = correct / num_labels

		# Calculate Accuracy@k
		self.accuracies.append(acc)
		self.accuracies_sum += acc

		if len(self.accuracies) > self.top_k:
			a = self.accuracies_sum.pop(0)
			self.accuracies_sum -= a

		acc_avg = self.accuracies_sum / self.accuracies
		return acc, acc_avg

	def topk_loss(self, loss: float) -> float:
		"""
		Calculates top-k loss
		:param loss: Loss at step
		:return: Average loss over K steps
		"""
		self.losses.append(loss)
		self.losses_sum += loss

		if len(self.losses) > self.top_k:
			l = self.losses.pop(0)
			self.losses_sum -= l

		loss_avg = self.losses_sum / len(self.losses)
		return loss_avg

	def perplexity(self, loss) -> float:
		return torch.exp(loss).item()


class MaskModel(nn.Module):
	def __init__(self, model, tokenizer):
		super(MaskModel, self).__init__()

		self.transformer = GatedTransformer(model, tokenizer)
		self.classifier = nn.Sequential(
			nn.Dropout(model.classifier.dropout),
			nn.Linear(model.dim, tokenizer.vocab),
		)

	def forward(
		self,
		x: torch.Tensor,
		x_pad: torch.Tensor,

		y: torch.Tensor,
		y_pad: torch.Tensor,
	) -> any:
		y, y_pad, ratio = self.transformer(x, x_pad, y, y_pad)
		y = self.classifier(y)
		return y, y_pad, ratio


def main():
	args = argparse.ArgumentParser()
	args.add_argument("config", type=Path)
	args.add_argument("output", type=Path)
	args = args.parse_args()

	if args.output.exists():
		print("Output file already exists. Aborting.")
		exit(1)

	config = load_config(args.config)

	model = MaskModel(
		model=config.model,
		tokenizer=config.tokenizer,
	)

	log = MaskModelLog(
		directory=args.output,
	)

	checkpoint = MaskModelCheckpoint(
		directory=args.output,
	)

	dataset = prepare_dataset(
		dataset=config.dataset.path,
		tokenizer=config.tokenizer.path,
		tokenized_col=config.dataset.text_col,
	)

	batch_kwargs = {
		"pad_index": config.tokenizer.pad,
		"mask_index": config.tokenizer.mask,
		"causal_mask": True, # TODO: False?
		"mask_prob": 0.15,
		"ignore_index": -100,
	}

	train_loader = data.DataLoader(
		dataset=dataset["train"],
		batch_size=config.train.batch,
		shuffle=True,
		collate_fn=partial(MaskModelBatch, **batch_kwargs),
	)

	test_loader = data.DataLoader(
		dataset=dataset["test"],
		batch_size=config.test.batch,
		shuffle=False,
		collate_fn=partial(MaskModelBatch, **batch_kwargs),
	)

	criterion = nn.CrossEntropyLoss()

	optimizer = optim.AdamW(
		params=model.parameters(),
		lr=config.train.lr
	)

	scheduler = get_scheduler(
		name=config.train.scheduler,
		optimizer=optimizer,
		num_warmup_steps=int(config.train.warmup * len(train_loader)),
		num_training_steps=config.train.epochs * len(train_loader),
	)

	train(
		epochs=config.train.epochs,
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
