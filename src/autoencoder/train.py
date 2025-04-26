import signal

from torch import optim
from torch.nn import functional as fn
from torch.utils import data
from tqdm import tqdm

from .checkpoint import AutoEncoderCheckpoint
from .dataset import AutoEncoderDataset
from .batch import AutoEncoderBatch
from .model import AutoEncoder
from .log import AutoEncoderLog

from ..util.metric import accuracy
from ..util.prepare import prepare_random, prepare_device
from ..util.prompt import prompt


def interrupt(_, __):
	if not prompt("Interrupt? "):
		return
	exit(0)


def main():
	prepare_random()
	device = prepare_device()

	epoch_num = 2
	warmup = 500

	model = AutoEncoder(
		name="google/flan-t5-small",
		bias=5,
		temperature=0.1,
	).to(device)
	dataset = AutoEncoderDataset(
		name="abisee/cnn_dailymail",
		version="3.0.0",
		split="train",
		text_column="article",
		tokenizer=model.name,
	)
	dataloader = data.DataLoader(
		dataset=dataset,
		batch_size=20,
		shuffle=True,
		collate_fn=AutoEncoderBatch.collate_fn(
			pad_token=model.pad_token,
			ign_token=-100,
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
		freq_step=5000,
		limit=2,
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
			memory, logits = model.forward(
				memory_tokens=batch.tokens,
				memory_eos_mask=batch.eos_mask,
				memory_pad_mask=batch.pad_mask,
				memory_attn_mask=None,
				memory_attn_scores=False,

				input_tokens=batch.tokens,
				input_pad_mask=batch.pad_mask,
				input_attn_mask=batch.decoder_mask,
			)

			loss_vol = (memory.rel_ratios ** 2).mean()
			loss_cls = fn.cross_entropy(
				logits.flatten(0, 1),
				batch.labels.flatten()
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

			acc = accuracy(logits, batch.labels, batch.ignore_token)
			abs_comp = memory.abs_ratios.mean().item()
			rel_comp = memory.rel_ratios.mean(dim=1).tolist()

			postfix = log(acc, abs_comp, rel_comp, loss, loss_cls, loss_vol)
			bar.set_postfix(postfix)
			bar.update(1)

			checkpoint(epoch, step)
		bar.close()
		scheduler.step()


if __name__ == "__main__":
	main()
