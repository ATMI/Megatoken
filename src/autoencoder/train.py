import json
import signal

import torch
from torch import optim
from torch.nn import functional as fn
from torch.utils import data
from tqdm import tqdm

from .config import Config
from .batch import Batch
from .model import Model

from ..util import prepare
from ..util.metric import RollingMean, accuracy
from ..util.prompt import prompt


def main():
	prepare.rnd(Config.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dataset = prepare.dataset()
	dataloader = data.DataLoader(
		dataset=dataset["train"],
		batch_size=Config.batch_size,
		shuffle=True,
		collate_fn=Batch.collate,
	)

	model = Model(Config.model, Config.bias, Config.temperature)
	model = model.to(device)

	optimizer = optim.Adam(model.parameters(), Config.lr)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.step, gamma=Config.gamma)

	# init = torch.load("0.pth", map_location=device, weights_only=True)
	# model.load_state_dict(init["model"])
	# optimizer.load_state_dict(init["optimizer"])

	step_num = len(dataloader)
	log = open("log.json", "w")
	rolling = RollingMean(Config.rolling_n)

	def finish():
		log.close()
		exit(0)

	def interrupt(_, __):
		if prompt("Checkpoint? "):
			checkpoint("interrupt.pth")
		if not prompt("Interrupt? "):
			return
		finish()

	signal.signal(signal.SIGINT, interrupt)
	for epoch in range(Config.epoch_num):
		bar = tqdm(
			total=step_num,
			leave=True,
			desc=f"Epoch {epoch + 1}/{Config.epoch_num}",
		)

		def checkpoint(path):
			state = {
				"epoch": epoch,
				"model": model.state_dict(),
				"optimizer": optimizer.state_dict(),
				"scheduler": scheduler.state_dict(),
			}
			torch.save(state, path)

		for step, batch in enumerate(dataloader):
			optimizer.zero_grad()

			batch = batch.to(device)
			memory, logits = model.forward(
				memory_tokens=batch.inputs,
				memory_eos_mask=batch.eos_mask,
				memory_pad_mask=batch.pad_mask,
				memory_attn_mask=None,
				memory_attn_scores=False,

				input_tokens=batch.inputs,
				input_pad_mask=batch.pad_mask,
				input_attn_mask=batch.decoder_mask,
			)

			loss_vol = (memory.rel_ratios ** 2).mean()
			loss_cls = fn.cross_entropy(logits.flatten(0, 1), batch.labels.flatten())

			if step < Config.warmup and epoch < 1:
				loss = loss_cls
			else:
				loss = loss_cls + 3 * loss_vol

			loss.backward()
			optimizer.step()
			# torch.cuda.empty_cache()

			acc = accuracy(logits, batch.labels, Config.ignore_token) * 100
			comp = memory.abs_ratios.mean().item()
			ratios = memory.rel_ratios.mean(dim=1).tolist()
			loss_vol = loss_vol.item()
			loss_cls = loss_cls.item()
			loss = loss.item()

			log_ = json.dumps({"acc": acc, "los": loss, "cls": loss_cls, "vol": loss_vol, "rat": ratios, "comp": comp})
			log.write(log_ + "\n")
			log.flush()

			acc, comp, ratios, loss, loss_vol, loss_cls = rolling(acc, comp, ratios, loss, loss_vol, loss_cls)
			bar.set_postfix(
				acc=f"{acc:.2f}",
				comp=f"{comp:.3f}",
				los=f"{loss:.3f}",
				cls=f"{loss_cls:.3f}",
				vol=f"{loss_vol:.3f}",
			)
			bar.update(1)

			if (step + 1) % (step_num // 10) == 0:
				checkpoint(f"{epoch}_{step}.pth")

		checkpoint(f"{epoch}.pth")
		scheduler.step()
		bar.close()
	finish()


if __name__ == "__main__":
	main()
