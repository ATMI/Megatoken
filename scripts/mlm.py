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

from pipeline.batch import Batch
from pipeline.checkpoint import Checkpoint
from pipeline.log import Log
from pipeline.step import Step, StepResult
from pipeline.train import train
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

	# TODO: make wider masks
	# TODO: make mode in masking
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
			"x": self.masked_x,
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

	def condition(self, step: Step) -> bool:
		cond = False

		# Keyboard Interruption
		if step.is_abort:
			cond = True

		# Save each N steps
		if step.idx % 1000 == 0:
			cond = True

		# Conditions based on previous checkpoints
		if self.prev_loss is not None:
			# Save if loss dropped greatly after last checkpoint
			loss_vel = self.prev_loss - step.result.loss
			if loss_vel >= self.tol: #0.05
				cond = True

			# Save each T minutes
			if self.timestamp - time.time() >= self.time_interval * 60:
				cond = True


		if cond:
			self.timestamp = time.time()
			self.prev_loss = step.result.loss


		return step.is_last or cond


class MaskModelLog(Log):

	def info(self, step: Step) -> Dict[str, any]:
		acc, acc_ = self.accuracy(step.result)
		ppl = self.perplexity(step.result.loss)
		loss_ = self.topk_loss(step.result.loss)

		# TODO: Fix ratios
		logs = {
			"loss": step.result.loss,
			"loss@K": loss_,
			"acc": acc,
			"acc@K": acc_,
			"ratio": step.result.pred.ratios.mean(),
			"PPL": ppl,
		}

		return logs

	def accuracy(self, result: StepResult) -> Tuple[float, float]:
		out = result.pred
		#TODO: fix access for label and ignore_index
		label = result.batch.label
		ignore_index = result.batch.ignore_index

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


def main():
	args = argparse.ArgumentParser()
	args.add_argument("config", type=Path, required=True)
	args.add_argument("output", type=Path, required=True)
	args.add_argument("tokenizer", type=str, default="google-bert/bert-base-uncased")
	args = args.parse_args()

	if args.output.exists():
		print("Output file already exists. Aborting.")
		exit(1)

	config = load_config(args.config)

	dataset = {}
	model = nn.Module()

	tokenizer_config = getattr(config.Tokenizer, args.tokenizer)
	#TODO provide other values in config file!
	batch_kwargs = {
		"pad_index": tokenizer_config.pad,
		"mask_index": tokenizer_config.mask,
		"causal_mask": True,
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
	optimizer = optim.AdamW(model.parameters(), lr=config.lr)
	scheduler = get_scheduler(
		name=config.scheduler,
		optimizer=optimizer,
		num_warmup_steps=int(config.train.warmup * len(train_loader)),
		num_training_steps=config.epochs * len(train_loader),
	)

	log = MaskModelLog(args.output)
	checkpoint = MaskModelCheckpoint(args.output)

	train(
		epochs=config.epochs,
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
