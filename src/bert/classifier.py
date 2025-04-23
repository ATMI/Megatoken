import json

import torch
from torch import nn
from torch import optim
from torch.utils import data
from tqdm import tqdm

from src.bert.dataset import Dataset
from src.bert.config import Config

from src.util import prepare, metric
from src.util.metric import RollingMean


class Classifier(nn.Module):
	def __init__(self):
		super().__init__()

		self.mlm = nn.Sequential(
			nn.Linear(768, 128),
			nn.ReLU(),
			nn.Linear(128, 1),
		)

	def forward(self, x):
		x = self.mlm(x)
		return x.squeeze()


def collate_fn(batch):
	embeds, labels = zip(*batch)
	embeds = torch.stack(embeds)
	labels = torch.tensor(labels)
	return embeds, labels




def val(
		model,
		dataloader,
		e
):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	bar = tqdm(dataloader, leave=False)
	rolling = RollingMean(Config.rolling_n)

	ac = []
	pr = []
	rc = []

	model.eval()
	for x, y in bar:
		x, y = x.to(device), y.to(device)
		with torch.no_grad():
			logits = model.forward(x)

		conf = metric.confusion(logits, y)
		a, p, r = rolling(conf.accuracy, conf.precision, conf.recall)
		ac.append(a)
		pr.append(p)
		rc.append(r)
		bar.set_postfix(a=a, p=p, r=r)
	print(f"Val {e}\t Acc:{sum(ac) / len(ac)}, Precision:{sum(pr) / len(pr)}, Recall:{sum(rc) / len(rc)}\n")


def epoch(
		model,
		criterion,
		dataloader,
		optimizer,
		e,
		log_file
):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	bar = tqdm(dataloader, leave=False)
	rolling = RollingMean(Config.rolling_n)

	ac = []
	pr = []
	rc = []

	model.train()
	for x, y in bar:
		x, y = x.to(device), y.to(device)
		optimizer.zero_grad()

		logits = model.forward(x)

		loss = criterion(logits, y.float())
		loss.backward()
		optimizer.step()

		conf = metric.confusion(logits, y)
		a, p, r = rolling(conf.accuracy, conf.precision, conf.recall)

		ac.append(a)
		pr.append(p)
		rc.append(r)
		bar.set_postfix(a=a, p=p, r=r)

		log = {
			"acc": a,
			"loss": loss.item(),
			"precision": p,
			"recall": r,
		}

		log_file.write(json.dumps(log) + "\n")
	print(f"Epoch {e}\t Acc:{sum(ac)/len(ac)}, Precision:{sum(pr)/len(pr)}, Recall:{sum(rc)/len(rc)}")

def main():
	prepare.rnd(Config.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dataset = Dataset("train")
	dataloader = data.DataLoader(dataset, Config.batch_size, collate_fn=collate_fn)

	dataset = Dataset("test")
	test_dl = data.DataLoader(dataset, Config.batch_size, collate_fn=collate_fn)

	model = Classifier()
	model = model.to(device)

	optimizer = optim.Adam(model.parameters(), Config.lr)
	criterion = nn.BCEWithLogitsLoss()

	log_file = open("bert-logs", "w")

	try:
		for i in range(10):
			epoch(model, criterion, dataloader, optimizer, i, log_file)
			val(model, test_dl, i)
			torch.save(model.state_dict(), f"bert-classifier-{i}.pth")
	except KeyboardInterrupt:
		print("Stop")
	finally:
		log_file.close()

if __name__ == "__main__":
	main()
