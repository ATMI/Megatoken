import signal

import torch
from torch import optim, Tensor
from torch.nn import functional as fn
from torch.utils import data
from tqdm import tqdm

from .batch import AutoEncoderBatch
from .checkpoint import AutoEncoderCheckpoint
from .dataset import AutoEncoderDataset
from .log import AutoEncoderLog
from .model import AutoEncoder, AutoEncoderConfig
from ..util.metric import accuracy
from ..util.prepare import prepare_random, prepare_device
from ..util.prompt import prompt


def interrupt(_, __):
	if not prompt("Interrupt? "):
		return
	exit(0)


def main():
	epoch_num = 1
	warmup = 0

	prepare_random()
	device = prepare_device()

	model_name = "google/flan-t5-small"
	config = AutoEncoderConfig.from_pretrained(model_name)

	model = AutoEncoder.from_pretrained(model_name, config=config)
	model = model.train().to(device)

	# model_init = "checkpoint/a80cc2759968a882eb92e8e8ae1a9cac7bb191c0/autoencoder_00.pth"
	# model_init = torch.load(model_init, map_location=device, weights_only=True)
	# model.load_state_dict(model_init["model"])

	dataset = AutoEncoderDataset(
		name="Yelp/yelp_review_full",
		version=None,
		split="train",
		text_column="text",
		model_name=model_name,
		ign_token=model.ign_token,
	)
	dataloader = data.DataLoader(
		dataset=dataset,
		batch_size=16,
		shuffle=True,
		collate_fn=AutoEncoderBatch.collate_fn(
			pad_token=model.pad_token,
			ign_token=model.ign_token,
		),
	)

	optimizer = optim.Adam(
		params=model.parameters(),
		lr=3e-4,
	)
	scheduler = optim.lr_scheduler.StepLR(
		optimizer=optimizer,
		step_size=1,
		gamma=0.5,
	)

	log = AutoEncoderLog(
		file="autoencoder.log",
		rolling_n=100,
	)
	checkpoint = AutoEncoderCheckpoint(
		model=model,
		optimizer=optimizer,
		scheduler=scheduler,
		dataloader=dataloader,
		freq_step=2000,
		limit=5,
	)

	signal.signal(signal.SIGINT, interrupt)
	for epoch in range(epoch_num):
		bar = tqdm(
			iterable=dataloader,
			leave=True,
			desc=f"Epoch {epoch + 1}/{epoch_num}",
		)

		for step, batch in enumerate(dataloader):
			optimizer.zero_grad()

			batch = batch.to(device)
			output = model(
				input_ids=batch.encoder_input_ids,
				attention_mask=batch.pad_mask,
				decoder_input_ids=batch.decoder_input_ids,
				decoder_attention_mask=None,
			)

			prune_lengths = output.prune_probs.sum(dim=2)
			prune_ratios = torch.roll(prune_lengths, 1)
			prune_ratios[:, 0] = batch.lengths
			prune_ratios = prune_lengths / prune_ratios

			loss_vol: Tensor = (prune_ratios ** 2).mean()
			loss_cls: Tensor = fn.cross_entropy(
				input=output.logits.flatten(0, 1),
				target=batch.labels.flatten(),
				ignore_index=model.ign_token
			)

			if step < warmup and epoch < 1:
				loss = loss_cls
			else:
				loss = loss_cls + 3 * loss_vol

			loss.backward()
			optimizer.step()

			loss_vol = loss_vol.item()
			loss_cls = loss_cls.item()
			loss = loss.item()

			acc = accuracy(output.logits, batch.labels, model.ign_token)
			abs_comp = prune_lengths[:, -1] / batch.lengths
			abs_comp = abs_comp.mean().item()
			rel_comp = prune_ratios.mean(dim=0).tolist()

			postfix = log(acc, abs_comp, rel_comp, loss, loss_cls, loss_vol)
			bar.set_postfix(postfix)
			bar.update(1)

			checkpoint(epoch, step)
		bar.close()
		scheduler.step()


if __name__ == "__main__":
	main()
