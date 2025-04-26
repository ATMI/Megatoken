import json

import datasets
import torch
from torch import optim
from torch.nn import functional as fn
from torch.utils import data
from tqdm import tqdm

from .dataset import Dataset
from .batch import Batch
from ..autoencoder.config import Config
from ..autoencoder.autoencoder import AutoEncoder
from ..util.metric import RollingMean, accuracy
from ..util import prepare


def main():
	prepare.prepare_random(Config.seed)

	dataset = datasets.load_from_disk("dataset")
	dataset = dataset["train"]
	# dataset = dataset.filter(lambda row: len(row["source"]) < Config.max_length)

	dataset = Dataset("memory", dataset)
	dataloader = data.DataLoader(
		dataset,
		batch_size=Config.batch_size,
		collate_fn=Batch.collate,
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = AutoEncoder(Config.model, Config.bias, Config.temperature)
	model = model.to(device)

	params = set()
	del model.t5.encoder
	for name, param in model.named_parameters():
		if name.startswith("t5.shared."):
			param.requires_grad = False
			continue
		params.add(name)

	init = torch.load("1.pth", map_location=device, weights_only=True)
	model.load_state_dict(init["model"], strict=False)

	optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), Config.lr)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.step, gamma=Config.gamma)

	log = open("log.json", "w")
	rolling = RollingMean(Config.rolling_n)

	for epoch in range(Config.epoch_num):
		bar = tqdm(dataloader, f"Epoch {epoch + 1}/{Config.epoch_num}", leave=True)

		for batch in bar:
			optimizer.zero_grad()

			batch = batch.to(device)
			logits = model.decode(
				memory=batch.memory,
				tokens=batch.target,
				pad_mask=batch.target_mask,
				attn_mask=None,
			)

			loss = fn.cross_entropy(logits.flatten(0, 1), batch.labels.flatten())
			loss.backward()
			optimizer.step()

			acc = accuracy(logits, batch.labels, Config.ignore_token) * 100
			loss = loss.item()

			record = {"acc": acc, "loss": loss}
			log.write(json.dumps(record) + "\n")

			acc, loss = rolling(acc, loss)
			bar.set_postfix(acc=acc, loss=loss)

		checkpoint = {
			"model": model.state_dict(),
			"optimizer": optimizer.state_dict(),
			"scheduler": scheduler.state_dict(),
		}
		torch.save(checkpoint, f"summarizer_{epoch}.pth")

		scheduler.step()
		bar.close()

	log.close()


if __name__ == "__main__":
	main()
