import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as fn
from transformers import T5ForConditionalGeneration, AutoTokenizer


class Prune(nn.Module):
	def __init__(
		self,
		bias: float,
		temperature: float,
	):
		super(Prune, self).__init__()
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
		embeds: Tensor
		prune_masks: Tensor
		prune_probs: Tensor

	@property
	def name(self) -> str:
		return self.t5.config.name_or_path

	@property
	def kv_dim(self) -> int:
		return self.t5.config.d_kv

	@property
	def max_length(self) -> int:
		return self.t5.config.n_positions

	@property
	def head_num(self) -> int:
		return self.t5.config.num_heads

	@property
	def encoder_num(self) -> int:
		return len(self.t5.encoder.block)

	@property
	def pad_token(self) -> int:
		return self.tokenizer.pad_token_id

	@property
	def ign_token(self) -> int:
		return -100

	@staticmethod
	def decoder_mask(max_length: int, visibility: int) -> Tensor:
		mask = torch.full((max_length, max_length), -torch.inf)
		for i in range(max_length):
			mask[i, max(0, i - visibility): i + 1] = 0
		return mask

	@staticmethod
	def pad_mask(
		indices: int | Tensor,
		lengths: Tensor,
		real: bool,
	) -> Tensor:
		if isinstance(indices, int):
			indices = torch.arange(indices, device=lengths.device)
		mask = indices.unsqueeze(0) < lengths.unsqueeze(1)
		mask = torch.where(mask, 0, -torch.inf) if real else mask
		return mask

	def __init__(
		self,
		name: str,
		visibility: int,
		bias: float,
		temperature: float,
	):
		super(AutoEncoder, self).__init__()
		self.tokenizer = AutoTokenizer.from_pretrained(name)
		self.t5 = T5ForConditionalGeneration.from_pretrained(name)

		self.prune_layers = nn.ModuleList(Prune(bias, temperature) for _ in range(self.encoder_num))
		self.decoder_mask = AutoEncoder.decoder_mask(self.max_length, visibility)

	def encode(
		self,
		src_tokens: Tensor,
		src_lengths: Tensor,
	) -> Memory:
		device = src_tokens.device
		batch_size = src_tokens.size(0)
		max_length = src_tokens.size(1)

		batch_indices = torch.arange(batch_size, device=device)
		token_indices = torch.arange(max_length, device=device)

		pad_mask = AutoEncoder.pad_mask(token_indices, src_lengths, False)
		attn_mask = torch.where(pad_mask, 0, -torch.inf)
		attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)

		prune_masks = (batch_size, self.encoder_num, max_length)
		prune_masks = torch.zeros(prune_masks, device=device)
		prune_probs = torch.zeros((batch_size, self.encoder_num, max_length), device=device)
		prune_keep = (torch.rand(batch_size, device=device) * src_lengths).long()

		# Kinda strange variable with cache disabled,
		# but it's used to calculate the position bias
		# in HF spaghetti. Trust me :)
		embeds = self.t5.encoder.embed_tokens(src_tokens)
		embeds = self.t5.encoder.dropout(embeds)

		for i, encoder_layer in enumerate(self.t5.encoder.block):
			embeds[:, :, 0] = 0.0
			embeds, attn_mask = encoder_layer(
				hidden_states=embeds,
				cache_position=token_indices,
				attention_mask=attn_mask,
				position_bias=attn_mask if i > 0 else None,
				output_attentions=False,
			)

			prune_layer = self.prune_layers[i]
			prune_mask = prune_layer(embeds=embeds)
			prune_mask = prune_mask + prune_masks[:, i - 1] if i else prune_mask
			prune_mask[batch_indices, prune_keep] = 0.0
			prune_masks[:, i] = prune_mask

			prune_prob = (prune_mask / math.sqrt(self.kv_dim)).exp()
			prune_prob = (prune_prob * pad_mask)
			prune_probs[:, i] = prune_prob

			prune_mask = prune_mask.unsqueeze(1) + prune_mask.unsqueeze(2)
			attn_mask = attn_mask + prune_mask.unsqueeze(1)
			attn_mask[:, :, token_indices, token_indices] = 0.0

		embeds = self.t5.encoder.final_layer_norm(embeds)
		embeds = self.t5.encoder.dropout(embeds)

		return AutoEncoder.Memory(
			embeds=embeds,
			prune_masks=prune_masks,
			prune_probs=prune_probs,
		)

	def decode(
		self,
		mem_embeds: Tensor,
		mem_lengths: Tensor,
		mem_mask: Tensor | None,

		tgt_tokens: Tensor,
		tgt_lengths: Tensor,
	):
		# Code here is similar to encode in some parts,
		# maybe it worth to do something with it or not :)
		device = mem_embeds.device
		batch_size = mem_embeds.size(0)

		mem_indices = torch.arange(mem_embeds.size(1), device=device)
		tgt_indices = torch.arange(tgt_tokens.size(1), device=device)

		tgt_attn_mask = AutoEncoder.pad_mask(tgt_indices, mem_lengths, True)
		mem_attn_mask = AutoEncoder.pad_mask(mem_indices, mem_lengths, True)

		mem_attn_mask = mem_attn_mask + mem_mask
		mem_attn_mask = mem_attn_mask.unsqueeze(1).unsqueeze(1)

		if self.decoder_mask.device != device:
			self.decoder_mask = self.decoder_mask.to(device)

		decoder_mask = self.decoder_mask.unsqueeze(0)
		tgt_attn_mask = tgt_attn_mask.unsqueeze(1)
		tgt_attn_mask = tgt_attn_mask + decoder_mask
		for i in range(batch_size):
			tgt_attn_mask[i, tgt_lengths[i]:, 0] = 0
		tgt_attn_mask = tgt_attn_mask.unsqueeze(1)

		tgt_embeds = self.t5.decoder.embed_tokens(tgt_tokens)
		tgt_embeds = self.t5.decoder.dropout(tgt_embeds)

		for i, decoder_layer in enumerate(self.t5.decoder.block):
			self_attn, cross_attn, fc = decoder_layer.layer

			tgt_embeds, _, tgt_attn_mask = self_attn(
				hidden_states=tgt_embeds,
				cache_position=tgt_indices,
				attention_mask=tgt_attn_mask,
				position_bias=tgt_attn_mask if i > 0 else None,
			)

			tgt_embeds, _, mem_attn_mask = cross_attn(
				hidden_states=tgt_embeds,
				key_value_states=mem_embeds,
				cache_position=mem_indices,
				attention_mask=mem_attn_mask,
				position_bias=mem_attn_mask if i > 0 else None,
			)

			tgt_embeds = fc(tgt_embeds)

		tgt_embeds = self.t5.decoder.final_layer_norm(tgt_embeds)
		tgt_embeds = self.t5.decoder.dropout(tgt_embeds)

		if self.t5.config.tie_word_embeddings:
			tgt_embeds = tgt_embeds * (self.t5.model_dim ** -0.5)

		logits = self.t5.lm_head(tgt_embeds)
		return logits

	def forward(
		self,
		src_tokens: Tensor,
		src_lengths: Tensor,

		tgt_tokens: Tensor,
		tgt_lengths: Tensor,
	) -> Tuple[Memory, Tensor]:
		# Encode
		mem = self.encode(
			src_tokens=src_tokens,
			src_lengths=src_lengths,
		)

		# Decode
		tgt_logits = self.decode(
			mem_embeds=mem.embeds,
			mem_lengths=src_lengths,
			mem_mask=mem.prune_masks[:, -1],

			tgt_tokens=tgt_tokens,
			tgt_lengths=tgt_lengths,
		)

		return mem, tgt_logits
