from dataclasses import dataclass

import torch
from torch import nn, Tensor

import prepare
from config import Config


class Model(nn.Module):
	@dataclass
	class Outputs:
		logits: Tensor
		lengths: Tensor

	def __init__(self):
		super().__init__()
		self.embedding = prepare.embedding()
		self.encoder = prepare.encoder()
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(Config.model_dim, Config.vocab_size)
		)

	def forward(
		self,
		tokens: Tensor,
		pad_mask: Tensor,
	) -> Outputs:
		batch = tokens.size(0)
		length = tokens.size(1)
		device = tokens.device
		attn_mask = torch.zeros((batch, length, length), device=device)

		embeds = self.embedding(tokens)
		result = self.encoder(embeds, pad_mask, attn_mask)
		logits = self.classifier(result.embeds)

		return Model.Outputs(
			logits=logits,
			lengths=result.lengths,
		)
