import json
import signal

import torch
from torch import optim
from torch.nn import functional as fn
from tqdm import tqdm

import prepare
from config import Config
from metric import RollingMean, accuracy
from model import Model
from prompt import prompt


def main():
	torch.manual_seed(Config.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_loader, test_loader = prepare.dataloaders()

	model = Model()
	model = model.to(device)
	optimizer = optim.Adam(model.parameters(), Config.lr)

	init = torch.load("checkpoint/1/20312.pth", map_location=device, weights_only=True)
	model.load_state_dict(init["model"])
	optimizer.load_state_dict(init["optimizer"])

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

	def interrupt(sig, frame):
		if not prompt("Interrupt? "):
			return
		if prompt("Checkpoint? "):
			checkpoint("checkpoint.pth")
		finish()

	signal.signal(signal.SIGINT, interrupt)
	for step, batch in enumerate(train_loader):
		optimizer.zero_grad()

		batch = batch.to(device)
		result = model(batch.sparse, batch.pad_mask)

		input_lengths = batch.pad_mask.size(1) - batch.pad_mask.sum(dim=1)
		input_lengths = input_lengths.unsqueeze(1)

		valve_loss = ((result.lengths / input_lengths) ** 2).mean()
		class_loss = fn.cross_entropy(result.logits.flatten(0, 1), batch.labels.flatten())
		loss = class_loss + valve_loss

		loss.backward()
		optimizer.step()
		torch.cuda.empty_cache()

		acc = accuracy(result.logits, batch.labels) * 100
		valve_loss = valve_loss.item()
		class_loss = class_loss.item()
		loss = loss.item()

		acc_, loss_, valve_, cls_ = rolling(acc, loss, valve_loss, class_loss)

		log = {
			"acc": f"{acc:.2f}",
			"acc~": f"{acc_:.2f}",
			"loss": f"{loss:.3f}",
			"loss~": f"{loss_:.3f}",
			"valve": f"{valve_loss:.3f}",
			"valve~": f"{valve_:.3f}",
			"cls": f"{class_loss:.3f}",
			"cls~": f"{cls_:.3f}",
		}
		log_file.write(json.dumps(log) + "\n")
		log_file.flush()

		postfix = {
			"acc": f"{acc_:.2f}",
			"loss": f"{loss_:.3f}",
			"valve": f"{valve_:.3f}",
			"cls": f"{cls_:.3f}",
		}
		bar.set_postfix(**postfix)
		bar.update(1)

		if (step + 1) % (step_num // 5) == 0 or (step + 1) == step_num:
			checkpoint(f"{step}.pth")
	finish()


if __name__ == "__main__":
	main()
