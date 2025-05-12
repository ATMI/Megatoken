from typing import Tuple

import torch
from torch import nn, Tensor, LongTensor, BoolTensor

from ..autoencoder.model import AutoEncoderConfig, AutoEncoder


class Encoder(nn.Module):
	def __init__(
		self,
		checkpoint: str,
		device: torch.device,
	):
		super(Encoder, self).__init__()

		name = "google/flan-t5-small"
		encoder_config = AutoEncoderConfig.from_pretrained(name)
		encoder = AutoEncoder(encoder_config)
		pad_token = encoder.pad_token

		checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
		checkpoint = checkpoint["model"]
		encoder.load_state_dict(checkpoint)
		encoder = encoder.encoder.eval().to(device)

		self.name = name
		self.pad_token = pad_token
		self.encoder = encoder

	@torch.no_grad()
	def forward(
		self,
		input_ids: LongTensor,
		attention_mask: BoolTensor,
	) -> Tuple[Tensor, LongTensor]:
		output = self.encoder(
			input_ids=input_ids,
			attention_mask=attention_mask,
		)

		mask = (output.prune_masks[:, -1] > -1) & attention_mask
		embeds = output.last_hidden_state[mask]
		indices = mask.nonzero(as_tuple=True)[0]

		return embeds, indices
