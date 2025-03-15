import argparse
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


class MLMBatch(Batch):

	def __init__(
		self,
		batch,
		pad: int,
		mask_token_id: int,
		causal_mask: bool,
		mask_prob: float = 0.15,
		ignore_index: int = -100,
	):
		self.x, self.pad_mask = self.collate(batch)

		self.pad_index = pad
		self.mask_token_id = mask_token_id
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

		mask = torch.zeros_like(self.x, dtype=torch.bool)

		for i in range(self.x.size(0)):
			candidates = torch.where(~self.pad_mask[i])[0]
			perm = torch.randperm(x_lens[i])
			selected = candidates[perm[:num_to_mask[i]]]
			mask[i, selected] = True

		masked_input = self.x.clone()
		masked_input[mask] = self.mask_token_id

		labels = torch.full_like(self.x, fill_value=self.ignore_index)  # -100 = ignore index for CrossEntropyLoss
		labels[mask] = self.x[mask]

		return masked_input, labels

	def x(self) -> Dict[str, any]:
		self.masked_x, self.label = self.masking()

		attn_mask = None
		if self.causal_mask:
			attn_mask = nn.Transformer.generate_square_subsequent_mask(self.x.size(1), dtype=torch.bool)

		inputs = {
			"x": self.masked_x,
			"pad": self.pad_mask,
			"attn": attn_mask
		}

		return inputs

	def y(self) -> torch.Tensor:
		"""
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

class MLMLog(Log):

	def info(self, step: Step) -> Dict[str, any]:
		acc, acc_ = self.accuracy(step.result)
		ppl = self.perplexity(step.result.loss)
		loss_ = self.topk_loss(step.result.loss)

		# TODO: Any other log values?
		logs = {
			"loss": step.result.loss,
			"loss@K": loss_,
			"acc": acc,
			"acc@K": acc_,
			"ratio": step.result.pred.ratio,
			"PPL": ppl,
		}

		return logs

	def accuracy(self, result: StepResult) -> Tuple[float, float]:
		out = result.pred
		label = result.batch.label  # TODO: access labels with shapes | reshape label here (bad idea)
		ignore_index = result.batch.ignore_index  # TODO: access ignore index

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

	def perplexity(self, loss: torch.Tensor) -> float:
		return torch.exp(loss).item()


def main():
	args = argparse.ArgumentParser()
	args.add_argument("config", type=Path, required=True)
	args.add_argument("output", type=Path, required=True)
	args = args.parse_args()

	if args.output.exists():
		print("Output file already exists. Aborting.")
		exit(1)

	config = load_config(args.config)

	dataset = {}
	model = nn.Module()

	train_loader = data.DataLoader(
		dataset=dataset["train"],
		batch_size=config.train.batch,
		shuffle=True,
		collate_fn=MLMBatch,
	)

	test_loader = data.DataLoader(
		dataset=dataset["test"],
		batch_size=config.test.batch,
		shuffle=False,
		collate_fn=MLMBatch,
	)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=config.lr)
	scheduler = get_scheduler(
		name=config.scheduler,
		optimizer=optimizer,
		num_warmup_steps=int(config.train.warmup * len(train_loader)),
		num_training_steps=config.epochs * len(train_loader),
	)

	log = MLMLog(args.output)
	checkpoint = MLMCheckpoint(args.output)

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
