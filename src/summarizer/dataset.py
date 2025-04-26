from torch.utils import data

from ..util.tensorfile import TensorReader


class Dataset(data.Dataset):
	def __init__(self, memory: str, dataset):
		self.memory = TensorReader(memory)
		self.labels = dataset

	def __len__(self) -> int:
		return len(self.labels)

	def __getitem__(self, idx):
		sample = self.labels[idx]
		memory = self.memory.by_id(sample["id"])
		target = sample["target"]
		text = sample["highlights"]
		return memory, target, text
