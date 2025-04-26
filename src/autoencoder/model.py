import math
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

		if self.training:
			gumbels = -torch.empty_like(gates, memory_format=torch.legacy_contiguous_format)
			gumbels = gumbels.exponential_().log()
			gates = gates + gumbels

		gates = (gates + self.bias) / self.temperature
		gates = fn.logsigmoid(gates)

		if not self.training:
			gates = torch.where(gates > -1, gates, -torch.inf)

		return gates


class AutoEncoder(nn.Module):
	@dataclass
	class Memory:
		kv_dim: int | None

		embeds: Tensor
		pad_mask: Tensor

		gate_masks: Tensor | None
		attn_scores: Tensor | None

		def gate_to_length(self, mask: Tensor) -> Tensor:
			length = (mask / math.sqrt(self.kv_dim)).exp()
			length = (length * self.pad_mask).sum(dim=-1)
			return length

		def __post_init__(self):
			# TODO: fix kolhoz
			if self.gate_masks is None:
				return

			after_gates = self.gate_masks
			prior_gates = torch.zeros_like(after_gates[0]).unsqueeze(0)
			prior_gates = torch.cat((prior_gates, after_gates[:-1]), dim=0)

			prior_lengths = self.gate_to_length(prior_gates)
			after_lengths = self.gate_to_length(after_gates)

			self.lengths = after_lengths
			self.rel_ratios = after_lengths / prior_lengths
			self.abs_ratios = after_lengths[-1] / prior_lengths[0]

		@property
		def gate_mask(self) -> Tensor:
			if self.gate_masks is None:
				return None
			return self.gate_masks[-1]

	def __init__(
		self,
		name: str,
		bias: float,
		temperature: float,
	):
		super(AutoEncoder, self).__init__()
		self.t5 = T5ForConditionalGeneration.from_pretrained(name)

		self.head_num = self.t5.config.num_heads
		self.attn_num = len(self.t5.encoder.block)
		self.gate_num = self.attn_num

		self.gates = nn.ModuleList(
			Gate(bias, temperature) for _ in range(self.gate_num)
		)

	def encode(
		self,
		tokens: Tensor,
		lengths: Tensor,
	) -> Memory:
		device = tokens.device
		batch_size = tokens.size(0)
		max_length = tokens.size(1)

		indices = torch.arange(max_length, device=device)
		attn_mask = indices[None, :]
		attn_mask = attn_mask < lengths

		gate_masks = (self.gate_num, batch_size, max_length)
		gate_masks = torch.zeros(gate_masks, device=device)

		# Kinda strange variable with cache disabled,
		# but it's used to calculate the position bias
		# in HF spaghetti. Trust me :)
		embeds = self.t5.encoder.embed_tokens(tokens)
		embeds = self.t5.encoder.dropout(embeds)

		for i, encoder_layer in enumerate(self.t5.encoder.block):
			embeds[:, :, 0] = 0.0
			embeds, attn_mask = encoder_layer(
				hidden_states=embeds,
				cache_position=indices,
				attention_mask=attn_mask,
				position_bias=attn_mask if i > 0 else None,
				output_attentions=False,
			)

			gate_layer = self.gates[i]
			gate = gate_layer(embeds=embeds)
			gate_masks[i] = gate if i == 0 else gate + gate_masks[i - 1]

			gate = gate.unsqueeze(1) + gate.unsqueeze(2)
			attn_mask = attn_mask + gate.unsqueeze(1)
			attn_mask[:, :, indices, indices] = 0.0

		embeds = self.t5.encoder.final_layer_norm(embeds)
		embeds = self.t5.encoder.dropout(embeds)

		return AutoEncoder.Memory(
			kv_dim=self.t5.config.d_kv,

			embeds=embeds,
			pad_mask=pad_mask,

			gate_masks=gate_masks,
			attn_scores=attn_scores,
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

		# TODO: normal fix
		if memory.gate_mask is not None:
			cross_attn_mask = cross_attn_mask + memory.gate_mask
		cross_attn_mask = cross_attn_mask[:, None, None, :]

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
		memory_eos_mask: Tensor,
		memory_pad_mask: Tensor | None,
		memory_attn_mask: Tensor | None,
		memory_attn_scores: bool,

		input_tokens: Tensor,
		input_pad_mask: Tensor | None,
		input_attn_mask: Tensor | None,
	) -> Tuple[Memory, Tensor]:
		# Encode
		memory = self.encode(
			tokens=memory_tokens,
			eos_mask=memory_eos_mask,
			pad_mask=memory_pad_mask,
			attn_mask=memory_attn_mask,
			attn_scores=memory_attn_scores,
		)

		# Decode
		logits = self.decode(
			memory=memory,
			tokens=input_tokens,
			pad_mask=input_pad_mask,
			attn_mask=input_attn_mask,
		)

		return memory, logits
