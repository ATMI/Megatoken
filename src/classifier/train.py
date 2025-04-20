import torch
from torch import nn
from torch import optim
from torch.utils import data
from tqdm import tqdm

from .batch import Batch
from .dataset import Dataset
from .model import Classifier
from .config import Config

from ..util import prepare
from ..util.metric import RollingMean


def main():
	prepare.rnd(Config.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dataset = Dataset()
	dataloader = data.DataLoader(dataset, Config.batch_size, collate_fn=Batch.collate)

	model = Classifier()
	model = model.to(device)

	optimizer = optim.Adam(model.parameters(), Config.lr)
	criterion = nn.BCEWithLogitsLoss()

	step_num = len(dataloader)
	bar = tqdm(total=step_num)
	rolling = RollingMean(Config.rolling_n)

	for batch in dataloader:
		optimizer.zero_grad()

		batch = batch.to(device)
		pred = model.forward(batch.embeds, batch.indices)

		loss = criterion(pred, batch.labels.float())
		loss.backward()
		optimizer.step()

		pred = (pred > 0)
		a = (pred == batch.labels).float().mean().item()
		tp = ((pred == 1) & (batch.labels == 1)).sum().item()
		fp = ((pred == 1) & (batch.labels == 0)).sum().item()
		fn = ((pred == 0) & (batch.labels == 1)).sum().item()

		p = tp / (tp + fp) if tp + fp > 0 else 0.0
		r = tp / (tp + fn) if tp + fn > 0 else 0.0
		a, p, r = rolling(a, p, r)
		bar.set_postfix(a=a, p=p, r=r)
		bar.update(1)

	torch.save(model.state_dict(), "classifier.pth")


if __name__ == "__main__":
	main()
