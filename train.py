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

	# init = torch.load("checkpoint/64f160e780f1bd4bc7b9ff6a0ec60c8c20b02a55/32499.pth", map_location=device, weights_only=True)
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
			memory_pad_mask=batch.pad_mask,
			memory_attn_mask=None,

			input_tokens=batch.inputs,
			input_pad_mask=batch.pad_mask,
			input_attn_mask=batch.decoder_mask,
		)

		loss_volume = result.volume.mean()
		loss_class = fn.cross_entropy(result.logits.flatten(0, 1), batch.labels.flatten())

		if step > 1000:
			loss = loss_class + 3 * loss_volume
		else:
			loss = loss_class

		loss.backward()
		optimizer.step()
		torch.cuda.empty_cache()

		acc = accuracy(result.logits, batch.labels) * 100
		ratio = result.volume[:, -1].mean().item() * 100
		loss_volume = loss_volume.item()
		loss_class = loss_class.item()
		loss = loss.item()

		acc_, ratio_, loss_, loss_volume_, loss_class_ = rolling(acc, ratio, loss, loss_volume, loss_class)

		log = {
			"acc": acc, "acc~": acc_,
			"los": loss, "los~": loss_,
			"cls": loss_class, "cls~": loss_class_,
			"vol": loss_volume, "vol~": loss_volume_,
			"rat": ratio, "rat~": ratio_,
		}
		log_file.write(json.dumps(log) + "\n")
		log_file.flush()

		bar.set_postfix(
			acc=f"{acc_:.2f}",
			los=f"{loss_:.3f}",
			cls=f"{loss_class_:.3f}",
			vol=f"{loss_volume_:.3f}",
			rat=f"{ratio_:.2f}",
		)
		bar.update(1)

		if (step + 1) % (step_num // 10) == 0 or (step + 1) == step_num:
			checkpoint(f"{step}.pth")
	finish()


if __name__ == "__main__":
	main()
