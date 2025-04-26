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
	prepare.prepare_random(Config.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dataset = Dataset("train")
	dataloader = data.DataLoader(dataset, Config.batch_size, collate_fn=Batch.collate)

	model = Classifier()
	model = model.to(device)

	optimizer = optim.Adam(model.parameters(), Config.lr)
	criterion = nn.BCEWithLogitsLoss()
	rolling = RollingMean(Config.rolling_n)

	for epoch in range(Config.epoch_num):
		bar = tqdm(
			dataloader,
			leave=True,
			desc=f"Epoch {epoch + 1}/{Config.epoch_num}"
		)
		for batch in bar:
			optimizer.zero_grad()

			batch = batch.to(device)
			logits = model.forward(batch.embeds, batch.indices)

			loss = criterion(logits, batch.target.float())
			loss.backward()
			optimizer.step()

			conf = metric.confusion(logits, batch.target)
			a, p, r = rolling(conf.accuracy, conf.precision, conf.recall)

			bar.set_postfix(a=a, p=p, r=r)
		bar.close()
		torch.save(model.state_dict(), f"{epoch}_classifier.pth")


if __name__ == "__main__":
	main()
