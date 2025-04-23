import json

import torch
from torch import nn
from torch import optim
from torch.utils import data
from tqdm import tqdm

from .batch import Batch
from .dataset import Dataset
from .model import Classifier
from .config import Config

from ..util import prepare, metric
from ..util.metric import RollingMean


def main():
	prepare.rnd(Config.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dataset = Dataset("train")
	dataloader = data.DataLoader(dataset, Config.batch_size, collate_fn=Batch.collate)

	model = Classifier()
	model = model.to(device)

	optimizer = optim.Adam(model.parameters(), Config.lr)
	criterion = nn.BCEWithLogitsLoss()

	log_file = open("cls-log.json", "w")

	bar = tqdm(dataloader)
	rolling = RollingMean(Config.rolling_n)

	for batch in bar:
		optimizer.zero_grad()

		batch = batch.to(device)
		logits = model.forward(batch.embeds, batch.indices)

		loss = criterion(logits, batch.labels.float())
		loss.backward()
		optimizer.step()

		conf = metric.confusion(logits, batch.labels)
		a, p, r = rolling(conf.accuracy, conf.precision, conf.recall)

		bar.set_postfix(a=a, p=p, r=r)
		log = {
			"acc": a,
			"loss": loss.item(),
			"precision": p,
			"recall": r,
		}
		log_file.write(json.dumps(log) + "\n")
		log_file.flush()

	torch.save(model.state_dict(), f"classifier-{i}.pth")

	log_file.close()

if __name__ == "__main__":
	main()
