import json
import signal

import torch
from torch import optim
from torch.nn import functional as fn
from tqdm import tqdm

import prepare
from config import Config
from metric import RollingMean, accuracy
from prompt import prompt


def main():
	torch.manual_seed(Config.seed)
	torch.cuda.manual_seed(Config.seed)
	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_loader, test_loader = prepare.dataloaders()

	model = prepare.model()
	model = model.to(device)
	optimizer = optim.Adam(model.parameters(), Config.lr)

	# init = torch.load("9749.pth", map_location=device, weights_only=True)
	# model.load_state_dict(init["model"])
	# optimizer.load_state_dict(init["optimizer"])

	step_num = len(train_loader)
	bar = tqdm(total=step_num)

	log_file = open("log.json", "w")
	rolling = RollingMean(Config.rolling_n)

	def finish():
		log_file.close()
		bar.close()
		exit(0)

	def checkpoint(path):
		state = {
			"model": model.state_dict(),
			"optimizer": optimizer.state_dict(),
		}
		torch.save(state, path)

	def interrupt(_, __):
		if prompt("Checkpoint? "):
			checkpoint("checkpoint.pth")
		if not prompt("Interrupt? "):
			return
		finish()

	signal.signal(signal.SIGINT, interrupt)
	for step, batch in enumerate(train_loader):
		optimizer.zero_grad()

		batch = batch.to(device)
		result = model.forward(
			memory_tokens=batch.inputs,
			memory_eos_mask=batch.eos_mask,
			memory_pad_mask=batch.pad_mask,
			memory_attn_mask=None,

			input_tokens=batch.inputs,
			input_pad_mask=batch.pad_mask,
			input_attn_mask=batch.decoder_mask,
		)

		input_lengths = batch.pad_mask.sum(dim=1, keepdim=True)
		input_lengths = torch.cat((input_lengths, result.volume[:, :-1]), dim=1)

		ratios = result.volume / input_lengths
		input_lengths = input_lengths[:, 0]

		loss_vol = ratios.mean()
		loss_cls = fn.cross_entropy(result.logits.flatten(0, 1), batch.labels.flatten())

		if step > Config.warmup:
			loss = loss_cls + 3 * loss_vol
		else:
			loss = loss_cls

		loss.backward()
		optimizer.step()
		# torch.cuda.empty_cache()

		acc = accuracy(result.logits, batch.labels) * 100
		comp = (result.volume[:, -1] / input_lengths).mean().item()
		ratios = ratios.mean(dim=0).tolist()
		loss_vol = loss_vol.item()
		loss_cls = loss_cls.item()
		loss = loss.item()

		log = {
			"acc": acc,
			"los": loss,
			"cls": loss_cls,
			"vol": loss_vol,
			"rat": ratios,
			"comp": comp,
		}
		log_file.write(json.dumps(log) + "\n")
		log_file.flush()

		acc, comp, ratios, loss, loss_vol, loss_cls = rolling(acc, comp, ratios, loss, loss_vol, loss_cls)
		bar.set_postfix(
			acc=f"{acc:.2f}",
			comp=f"{comp:.3f}",
			los=f"{loss:.3f}",
			cls=f"{loss_cls:.3f}",
			vol=f"{loss_vol:.3f}",
		)
		bar.update(1)

		if (step + 1) % (step_num // 10) == 0 or (step + 1) == step_num:
			checkpoint(f"{step}.pth")
	finish()


if __name__ == "__main__":
	main()
