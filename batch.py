from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils import rnn

from config import Config


@dataclass
class Batch:
	tokens: Tensor
	sparse: Tensor
	labels: Tensor
	pad_mask: Tensor

	def to(self, device) -> "Batch":
		return Batch(
			self.tokens.to(device),
			self.sparse.to(device),
			self.labels.to(device),
			self.pad_mask.to(device),
		)

	@staticmethod
	def collate(batch) -> "Batch":
		batch = [row["tokens"] for row in batch]
		tokens = [torch.tensor(row) for row in batch]
		tokens = rnn.pad_sequence(tokens, batch_first=True, padding_value=Config.pad_token)

		sparse = tokens.clone()
		labels = torch.full_like(tokens, Config.ignore_token)
		pad_mask = torch.ones_like(tokens, dtype=torch.bool)

		for i, row in enumerate(batch):
			length = len(row)
			pad_mask[i, :length] = False

			sparsity = max(1, int(length * Config.sparsity))
			indices = torch.randperm(length)
			indices = indices[:sparsity]

			sparse[i, indices] = Config.mask_token
			labels[i, indices] = tokens[i, indices]

		return Batch(tokens, sparse, labels, pad_mask)
