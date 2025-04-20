from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor


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
	def collate(batch: List[Tuple[Tensor, bool]]) -> "Batch":
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
