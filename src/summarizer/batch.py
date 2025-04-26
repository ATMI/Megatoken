from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ..autoencoder.config import Config
from ..autoencoder.autoencoder import AutoEncoder


@dataclass
class Batch:
	memory: AutoEncoder.Memory
	target: Tensor
	target_mask: Tensor
	labels: Tensor
	text: List[str]

	def to(self, device) -> "Batch":
		return Batch(
			memory=AutoEncoder.Memory(
				embeds=self.memory.embeds.to(device),
				pad_mask=self.memory.pad_mask.to(device),

				kv_dim=None,
				gate_masks=None,
				attn_scores=None,
			),
			target=self.target.to(device),
			target_mask=self.target_mask.to(device),
			labels=self.labels.to(device),
			text=self.text,
		)

	@staticmethod
	def collate(
		batch: List[Tuple[Tensor, Tensor, str]],
	) -> "Batch":
		batch_size = len(batch)
		memory, labels, text = tuple(map(list, zip(*batch)))

		target = [torch.tensor([Config.pad_token] + label[:-1]) for label in labels]
		labels = [torch.tensor(label) for label in labels]

		labels_tensor = pad_sequence(labels, batch_first=True, padding_value=Config.ignore_token)
		target_tensor = pad_sequence(target, batch_first=True, padding_value=Config.pad_token)
		memory_tensor = pad_sequence(memory, batch_first=True, padding_value=0.0)

		memory_mask = torch.ones((batch_size, memory_tensor.size(1)), dtype=torch.bool)
		target_mask = torch.ones((batch_size, target_tensor.size(1)), dtype=torch.bool)

		for row in range(batch_size):
			memory_mask[row, len(memory[row]):] = 0
			target_mask[row, len(target[row]):] = 0

		memory = AutoEncoder.Memory(
			embeds=memory_tensor,
			pad_mask=memory_mask,

			kv_dim=None,
			gate_masks=None,
			attn_scores=None,
		)

		return Batch(
			memory=memory,
			target=target_tensor,
			target_mask=target_mask,
			labels=labels_tensor,
			text=text,
		)
