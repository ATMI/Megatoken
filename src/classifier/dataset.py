from torch.utils import data

from ..util.tensorfile import TensorReader
from ..util import prepare


class Dataset(data.Dataset):
	def __init__(self, split: str):
		super(Dataset, self).__init__()

		self.embed = TensorReader(f"embeds.cot.{split}")
		self.ds = prepare.dataset()[split]

	def __len__(self):
		return len(self.ds)

	def __getitem__(self, idx):
		sample = self.ds[idx]
		embed = self.embed.by_id(sample["id"])
		label = sample["label"] > 2
		return embed, label
