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
		dense, filled = model.forward(
			batch.tokens, batch.pad_mask,
			batch.sparse, batch.pad_mask,
		)

		tokens_count = batch.pad_mask.numel() - batch.pad_mask.sum()
		valve_loss = dense.pressure.mean() / tokens_count
		cls_loss = fn.cross_entropy(filled.flatten(0, 1), batch.labels.flatten())
		loss = cls_loss + valve_loss

		loss.backward()
		optimizer.step()
		torch.cuda.empty_cache()

		acc = accuracy(filled, batch.labels) * 100
		valve_loss = valve_loss.item()
		cls_loss = cls_loss.item()
		loss = loss.item()

		acc_, loss_, valve_, cls_ = rolling(acc, loss, valve_loss, cls_loss)

		log = {
			"acc": f"{acc:.2f}",
			"acc~": f"{acc_:.2f}",
			"loss": f"{loss:.3f}",
			"loss~": f"{loss_:.3f}",
			"valve": f"{valve_loss:.3f}",
			"valve~": f"{valve_:.3f}",
			"cls": f"{cls_loss:.3f}",
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

		if step % (step_num // 5) == 0 or (step + 1) == step_num:
			checkpoint(f"{step}.pth")
	finish()


if __name__ == "__main__":
	main()
