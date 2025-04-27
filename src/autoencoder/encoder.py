from dataclasses import dataclass
from typing import Tuple
import math

import torch

from torch import nn
from torch.nn import functional as fn
from torch import Tensor, FloatTensor
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Stack


class PruneLayer(nn.Module):
	def __init__(
		self,
		bias: float,
		temperature: float,
	):
		super(PruneLayer, self).__init__()
		self.bias = bias
		self.temperature = temperature

	def forward(
		self,
		embeds: Tensor,
	) -> Tuple[Tensor]:
		gates = embeds[:, :, 0]

		if self.training:
			gumbels = -torch.empty_like(gates, memory_format=torch.legacy_contiguous_format)
			gumbels = gumbels.exponential_().log()
			gates = gates + gumbels

		gates = (gates + self.bias) / self.temperature
		gates = fn.logsigmoid(gates)

		if not self.training:
			gates = torch.where(gates > -1, gates, -torch.inf)

		return gates


class Encoder(T5Stack):
	@dataclass
	class Output(BaseModelOutputWithPastAndCrossAttentions):
		input_embeds: FloatTensor = None
		prune_masks: FloatTensor = None
		prune_probs: FloatTensor = None

	def __init__(self, config, embed_tokens=None):
		super().__init__(config, embed_tokens)
		self.prune_layers = None

	def init(self):
		bias = self.config.prune_bias
		temperature = self.config.prune_temperature

		self.prune_layers = nn.ModuleList(
			PruneLayer(bias, temperature)
			for _ in range(len(self.block))
		)

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		**kwargs,
		# encoder_hidden_states=None,
		# encoder_attention_mask=None,
		# inputs_embeds=None,
		# head_mask=None,
		# cross_attn_head_mask=None,
		# past_key_values=None,
		# use_cache=None,
		# output_attentions=None,
		# output_hidden_states=None,
		# return_dict=None,
		# cache_position=None
	) -> Output:
		device = input_ids.device
		batch_size = input_ids.size(0)
		input_length = input_ids.size(1)

		batch_indices = torch.arange(batch_size, device=device)
		token_indices = torch.arange(input_length, device=device)

		padding_mask = attention_mask.bool()
		input_lengths = padding_mask.sum(dim=1)

		attention_mask = torch.where(padding_mask, 0, -torch.inf)
		attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

		layer_num = len(self.block)
		prune_masks = (batch_size, layer_num, input_length)
		prune_masks = torch.zeros(prune_masks, device=device)
		prune_probs = torch.zeros((batch_size, layer_num, input_length), device=device)
		prune_keep = (torch.rand(batch_size, device=device) * input_lengths).long()

		# Kinda strange variable with cache disabled,
		# but it's used to calculate the position bias
		# in HF spaghetti. Trust me :)
		embeds = self.embed_tokens(input_ids)
		embeds = self.dropout(embeds)

		for i, encoder_layer in enumerate(self.block):
			embeds[:, :, 0] = 0.0
			embeds, attention_mask = encoder_layer(
				hidden_states=embeds,
				cache_position=token_indices,
				attention_mask=attention_mask,
				position_bias=attention_mask if i > 0 else None,
				output_attentions=False,
			)

			prune_layer = self.prune_layers[i]
			prune_mask = prune_layer(embeds=embeds)
			prune_mask = prune_mask + prune_masks[:, i - 1] if i else prune_mask
			prune_mask[batch_indices, prune_keep] = 0.0
			prune_masks[:, i] = prune_mask

			prune_prob = (prune_mask / math.sqrt(self.config.d_kv)).exp()
			prune_prob = (prune_prob * padding_mask)
			prune_probs[:, i] = prune_prob

			prune_mask = prune_mask.unsqueeze(1) + prune_mask.unsqueeze(2)
			attention_mask = attention_mask + prune_mask.unsqueeze(1)
			attention_mask[:, :, token_indices, token_indices] = 0.0

		embeds = self.final_layer_norm(embeds)
		embeds = self.dropout(embeds)

		output = Encoder.Output(
			last_hidden_state=embeds,
			prune_masks=prune_masks,
			prune_probs=prune_probs,
		)

		return output
