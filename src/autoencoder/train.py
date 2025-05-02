import signal

from torch import optim
from torch.nn import functional as fn
from torch.utils import data
from tqdm import tqdm
from transformers import T5ForConditionalGeneration

from .batch import AutoEncoderBatch
from .checkpoint import AutoEncoderCheckpoint
from .dataset import AutoEncoderDataset
from .log import AutoEncoderLog
from ..util.metric import accuracy
from ..util.prepare import prepare_random, prepare_device
from ..util.prompt import prompt


def interrupt(_, __):
	if not prompt("Interrupt? "):
		return
	exit(0)


def main():
	epoch_num = 2
	warmup = 500

	prepare_random()
	device = prepare_device()

	model_name = "google/flan-t5-small"
	model = T5ForConditionalGeneration.from_pretrained(model_name)
	model = model.to(device).train()

	dataset = AutoEncoderDataset(
		split="train",
		model_name=model_name,
		ign_token=-100,
	)
	dataloader = data.DataLoader(
		dataset=dataset,
		batch_size=12,
		shuffle=True,
		collate_fn=AutoEncoderBatch.collate_fn(
			pad_token=model.config.pad_token_id,
			ign_token=-100,
		),
	)

	optimizer = optim.AdamW(
		params=model.parameters(),
		lr=2e-5,
		weight_decay=0.01,
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
		freq_step=5000,
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

			# prune_lengths = output.prune_probs.sum(dim=2)
			# prune_ratios = torch.roll(prune_lengths, 1)
			# prune_ratios[:, 0] = batch.lengths
			# prune_ratios = prune_lengths / prune_ratios  # .detach()
			# loss_vol = (prune_ratios ** 2).mean()

			loss = fn.cross_entropy(
				input=output.logits.flatten(0, 1),
				target=batch.labels.flatten(),
				ignore_index=-100,
			)

			# if step < warmup and epoch < 1:
			# 	loss = loss_cls
			# else:
			# 	loss = loss_cls + 3 * loss_vol

			loss.backward()
			optimizer.step()

			# loss_vol = loss_vol.item()
			# loss_cls = loss_cls.item()
			loss = loss.item()

			acc = accuracy(output.logits, batch.labels, -100)
			# abs_comp = prune_lengths[:, -1] / batch.lengths
			# abs_comp = abs_comp.mean().item()
			# rel_comp = prune_ratios.mean(dim=0).tolist()

			# postfix = log(acc, abs_comp, rel_comp, loss, loss_cls, loss_vol)
			postfix = log(acc, loss)
			bar.set_postfix(postfix)
			bar.update(1)

			checkpoint(epoch, step)
		bar.close()
		scheduler.step()


if __name__ == "__main__":
	main()
