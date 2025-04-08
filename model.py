from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as fn
from transformers import T5ForConditionalGeneration


class Gate(nn.Module):
	def __init__(
		self,
		bias: float,
		temperature: float,
	):
		super(Gate, self).__init__()
		self.bias = bias
		self.temperature = temperature

	def forward(
		self,
		embeds: Tensor,
	) -> Tuple[Tensor]:
		gates = embeds[:, :, 0]

		gumbels = -torch.empty_like(gates, memory_format=torch.legacy_contiguous_format).exponential_().log()
		gumbels = (gates + gumbels + self.bias) / self.temperature
		gumbels = fn.logsigmoid(gumbels)

		return gumbels


class Model(nn.Module):
	@dataclass
	class Outputs:
		logits: Tensor
		memory: Tensor
		volume: Tensor

	@dataclass
	class Memory:
		pad_mask: Tensor
		gate_mask: Tensor

		embeds: Tensor
		volume: Tensor

	def __init__(
		self,
		name: str,
		bias: float,
		temperature: float,
	):
		super(Model, self).__init__()
		self.t5 = T5ForConditionalGeneration.from_pretrained(name)
		self.gates = nn.ModuleList(
			Gate(bias, temperature)
			for _ in range(len(self.t5.encoder.block) // 2)
		)

	def encode(
		self,
		tokens: Tensor,
		pad_mask: Tensor | None,
		attn_mask: Tensor | None,
	) -> Memory:
		device = tokens.device
		batch_size = tokens.size(0)
		input_length = tokens.size(1)

		if attn_mask is None:
			mask_size = (batch_size, input_length, input_length)
			attn_mask = torch.zeros(mask_size, device=device)

		if pad_mask is not None:
			mask = torch.where(pad_mask, 0, -torch.inf)
			attn_mask = attn_mask + mask.unsqueeze(1)

		# Adding dimension per attention head for HF compatibility
		attn_mask = attn_mask.unsqueeze(1)
		gate_mask = torch.zeros(input_length, device=device)
		diag_mask = 1 - torch.eye(input_length, device=device)
		diag_mask = diag_mask * pad_mask.unsqueeze(2)
		volume = torch.zeros((batch_size, len(self.gates)), device=device)

		# Kinda strange variable with cache disabled,
		# but it's used to calculate the position bias
		# in HF spaghetti. Trust me :)
		cache_position = torch.arange(input_length, device=device)
		embeds = self.t5.encoder.embed_tokens(tokens)
		embeds = self.t5.encoder.dropout(embeds)

		heads_num = self.t5.config.num_heads
		input_volume = pad_mask.sum(dim=1) * heads_num
		indices = torch.arange(input_length, device=device)

		for i, encoder_layer in enumerate(self.t5.encoder.block):
			j = i // 2

			embeds[:, :, 0] = 0.0
			embeds, attn_mask, attn_scores = encoder_layer(
				hidden_states=embeds,
				cache_position=cache_position,
				attention_mask=attn_mask,
				position_bias=attn_mask if i > 0 else None,
				output_attentions=True,
				# position_bias=None,
			)

			if i % 2 == 0:
				gate_layer = self.gates[j]
				gate = gate_layer(embeds=embeds)
				gate_mask = gate_mask + gate

				gate = gate.unsqueeze(1) + gate.unsqueeze(2)
				attn_mask = attn_mask + gate.unsqueeze(1)
				attn_mask[:, :, indices, indices] = 0.0
			else:
				ratios = attn_scores.sum(dim=1)
				ratios = ratios * diag_mask
				ratios = ratios.sum(dim=(1, 2))
				ratios = ratios / input_volume
				volume[:, j] = ratios

		embeds = self.t5.encoder.final_layer_norm(embeds)
		embeds = self.t5.encoder.dropout(embeds)

		return Model.Memory(
			pad_mask=pad_mask,
			gate_mask=gate_mask,

			embeds=embeds,
			volume=volume,
		)

	def decode(
		self,
		memory: Memory,
		tokens: Tensor,
		pad_mask: Tensor | None,
		attn_mask: Tensor | None,
	):
		# Code here is similar to encode in some parts,
		# maybe it worth to do something with it or not :)
		device = tokens.device
		batch_size = tokens.size(0)
		input_length = tokens.size(1)

		if attn_mask is None:
			mask_size = (batch_size, input_length, input_length)
			attn_mask = torch.full(mask_size, -torch.inf, device=device)
			attn_mask = attn_mask.triu(diagonal=1)

		if pad_mask is not None:
			mask = torch.where(pad_mask, 0, -torch.inf)
			attn_mask = attn_mask + mask.unsqueeze(1)

		self_attn_mask = attn_mask.unsqueeze(1)
		cross_attn_mask = torch.where(memory.pad_mask, 0, -torch.inf)
		cross_attn_mask = cross_attn_mask + memory.gate_mask
		cross_attn_mask = cross_attn_mask[:, None, None, :]
		# cross_attn_mask = cross_attn_mask.repeat(1, 1, input_length, 1)

		cache_position = torch.arange(input_length, device=device)
		input_embeds = self.t5.decoder.embed_tokens(tokens)
		input_embeds = self.t5.decoder.dropout(input_embeds)

		for i, decoder_layer in enumerate(self.t5.decoder.block):
			self_attn, cross_attn, fc = decoder_layer.layer

			input_embeds, _, self_attn_mask = self_attn(
				hidden_states=input_embeds,
				cache_position=cache_position,
				attention_mask=self_attn_mask,
				position_bias=self_attn_mask if i > 0 else None,
				# position_bias=None,
			)

			input_embeds, _, cross_attn_mask = cross_attn(
				hidden_states=input_embeds,
				key_value_states=memory.embeds,
				cache_position=cache_position,
				attention_mask=cross_attn_mask,
				position_bias=cross_attn_mask if i > 0 else None,
				# position_bias=None,
			)

			input_embeds = fc(input_embeds)

		input_embeds = self.t5.decoder.final_layer_norm(input_embeds)
		input_embeds = self.t5.decoder.dropout(input_embeds)

		if self.t5.config.tie_word_embeddings:
			input_embeds = input_embeds * (self.t5.model_dim ** -0.5)

		logits = self.t5.lm_head(input_embeds)
		return logits

	def forward(
		self,
		memory_tokens: Tensor,
		input_tokens: Tensor,

		memory_pad_mask: Tensor | None,
		input_pad_mask: Tensor | None,

		memory_attn_mask: Tensor | None = None,
		input_attn_mask: Tensor | None = None,
	) -> "Outputs":
		# Encode
		memory = self.encode(
			tokens=memory_tokens,
			pad_mask=memory_pad_mask,
			attn_mask=memory_attn_mask,
		)

		# Decode
		logits = self.decode(
			memory=memory,
			tokens=input_tokens,
			pad_mask=input_pad_mask,
			attn_mask=input_attn_mask,
		)

		return Model.Outputs(
			logits=logits,
			memory=memory.embeds,
			volume=memory.volume,
		)
