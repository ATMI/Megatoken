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
		if not prompt("Interrupt? "):
			return
		if prompt("Checkpoint? "):
			checkpoint("checkpoint.pth")
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
			input_attn_mask=None,
		)

		input_length = batch.pad_mask.sum(dim=1)
		input_length = input_length.unsqueeze(1)

		loss_volume = ((result.volume / input_length) ** 2).mean()
		loss_class = fn.cross_entropy(result.logits.flatten(0, 1), batch.labels.flatten())
		loss = loss_class + loss_volume

		loss.backward()
		optimizer.step()
		torch.cuda.empty_cache()

		acc = accuracy(result.logits, batch.labels) * 100
		loss_volume = loss_volume.item()
		loss_class = loss_class.item()
		loss = loss.item()

		acc_, loss_, loss_volume_, loss_class_ = rolling(acc, loss, loss_volume, loss_class)

		log = {
			"acc": acc, "acc~": acc_,
			"loss": loss, "loss~": loss_,
			"class": loss_class, "class~": loss_class_,
			"volume": loss_volume, "valve~": loss_volume_,
		}
		log_file.write(json.dumps(log) + "\n")
		log_file.flush()

		postfix = {
			"acc": f"{acc_:.2f}",
			"loss": f"{loss_:.3f}",
			"class": f"{loss_class_:.3f}",
			"volume": f"{loss_volume_:.3f}",
		}
		bar.set_postfix(**postfix)
		bar.update(1)

		if (step + 1) % (step_num // 10) == 0 or (step + 1) == step_num:
			checkpoint(f"{step}.pth")
	finish()


if __name__ == "__main__":
	main()
