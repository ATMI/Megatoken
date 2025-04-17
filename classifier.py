from dataclasses import dataclass
from typing import Tuple, List

import torch
from torch import nn, Tensor
from torch import optim
from torch.utils import data
from tqdm import tqdm

import binfile
import prepare
from config import Config
from metric import RollingMean


class Dataset(data.Dataset):
	def __init__(self):
		super(Dataset, self).__init__()

		self.embed = binfile.Reader("embeds")
		self.ds = prepare.dataset()["train"]

	def __len__(self):
		return len(self.ds)

	def __getitem__(self, idx):
		sample = self.ds[idx]
		embed = self.embed.by_id(sample["id"])
		label = sample["label"] > 2
		return embed, label


class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(512, 128),
			nn.ReLU(),
			nn.Linear(128, 1),
		)

	def forward(self, embeds: Tensor, indices: Tensor) -> Tensor:
		probs = self.mlp(embeds).squeeze(1)
		zeros = torch.zeros(indices[-1] + 1, device=embeds.device)
		probs = zeros.scatter_add(0, indices, probs)
		return probs


@dataclass
class Batch:
	embeds: Tensor
	labels: Tensor
	indices: Tensor

	def to(self, device) -> "Batch":
		return Batch(
			embeds=self.embeds.to(device),
			labels=self.labels.to(device),
			indices=self.indices.to(device),
		)

	@staticmethod
	def collate_fn(batch: List[Tuple[Tensor, bool]]) -> "Batch":
		embeds = []
		labels = []
		indices = []

		for i, sample in enumerate(batch):
			embed, label = sample
			index = torch.tensor([i] * len(embed), dtype=torch.long)

			embeds.append(embed)
			labels.append(label)
			indices.append(index)

		labels = torch.tensor(labels)
		embeds = torch.cat(embeds, dim=0)
		indices = torch.cat(indices, dim=0)

		return Batch(
			embeds=embeds,
			labels=labels,
			indices=indices,
		)


def main():
	prepare.rnd()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dataset = Dataset()
	dataloader = data.DataLoader(dataset, Config.batch_size, collate_fn=Batch.collate_fn)

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


if __name__ == "__main__":
	main()
