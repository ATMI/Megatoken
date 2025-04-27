import math
from dataclasses import dataclass
from typing import Tuple

import torch
from joblib import Memory
from torch import nn, Tensor
from torch.nn import functional as fn
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.utils import ModelOutput


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
		input_embeds: Tensor
		prune_masks: Tensor
		prune_probs: Tensor

	@dataclass
	class Output(Seq2SeqLMOutput):
		prune_probs: Tensor = None

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

	def __init__(
		self,
		name: str,
		bias: float,
		temperature: float,
	):
		super(AutoEncoder, self).__init__()
		self.tokenizer = AutoTokenizer.from_pretrained(name)
		self.t5 = T5ForConditionalGeneration.from_pretrained(name)
		self.prune_layers = nn.ModuleList(Prune(bias, temperature) for _ in range(self.encoder_num))

	def encode(
		self,
		input_ids: Tensor,  # (batch_size, seq_len)
		attention_mask: Tensor,  # (batch_size, seq_len)
	) -> Memory:
		device = input_ids.device
		batch_size = input_ids.size(0)
		input_length = input_ids.size(1)

		batch_indices = torch.arange(batch_size, device=device)
		token_indices = torch.arange(input_length, device=device)

		padding_mask = attention_mask
		input_lengths = padding_mask.sum(dim=1)

		attention_mask = torch.where(padding_mask, 0, -torch.inf)
		attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

		prune_masks = (batch_size, self.encoder_num, input_length)
		prune_masks = torch.zeros(prune_masks, device=device)
		prune_probs = torch.zeros((batch_size, self.encoder_num, input_length), device=device)
		prune_keep = (torch.rand(batch_size, device=device) * input_lengths).long()

		# Kinda strange variable with cache disabled,
		# but it's used to calculate the position bias
		# in HF spaghetti. Trust me :)
		input_embeds = self.t5.encoder.embed_tokens(input_ids)
		input_embeds = self.t5.encoder.dropout(input_embeds)

		for i, encoder_layer in enumerate(self.t5.encoder.block):
			input_embeds[:, :, 0] = 0.0
			input_embeds, attention_mask = encoder_layer(
				hidden_states=input_embeds,
				cache_position=token_indices,
				attention_mask=attention_mask,
				position_bias=attention_mask if i > 0 else None,
				output_attentions=False,
			)

			prune_layer = self.prune_layers[i]
			prune_mask = prune_layer(embeds=input_embeds)
			prune_mask = prune_mask + prune_masks[:, i - 1] if i else prune_mask
			prune_mask[batch_indices, prune_keep] = 0.0
			prune_masks[:, i] = prune_mask

			prune_prob = (prune_mask / math.sqrt(self.kv_dim)).exp()
			prune_prob = (prune_prob * padding_mask)
			prune_probs[:, i] = prune_prob

			prune_mask = prune_mask.unsqueeze(1) + prune_mask.unsqueeze(2)
			attention_mask = attention_mask + prune_mask.unsqueeze(1)
			attention_mask[:, :, token_indices, token_indices] = 0.0

		input_embeds = self.t5.encoder.final_layer_norm(input_embeds)
		input_embeds = self.t5.encoder.dropout(input_embeds)

		return AutoEncoder.Memory(
			input_embeds=input_embeds,
			prune_masks=prune_masks,
			prune_probs=prune_probs,
		)

	def decode(
		self,
		input_embeds: Tensor,
		attention_mask: Tensor,
		decoder_input_ids: Tensor,
		decoder_attention_mask: Tensor,
	):
		# Code here is similar to encode in some parts,
		# maybe it worth to do something with it or not :)
		device = input_embeds.device
		input_length = input_embeds.size(1)
		output_length = decoder_input_ids.size(1)

		input_indices = torch.arange(input_length, device=device)
		output_indices = torch.arange(output_length, device=device)

		attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
		decoder_attention_mask = decoder_attention_mask.unsqueeze(1)
		decoder_attention_mask = torch.where(decoder_attention_mask, 0, -torch.inf)

		output_embeds = self.t5.decoder.embed_tokens(decoder_input_ids)
		output_embeds = self.t5.decoder.dropout(output_embeds)

		for i, decoder_layer in enumerate(self.t5.decoder.block):
			self_attn, cross_attn, fc = decoder_layer.layer

			output_embeds, _, decoder_attention_mask = self_attn(
				hidden_states=output_embeds,
				cache_position=output_indices,
				attention_mask=decoder_attention_mask,
				position_bias=decoder_attention_mask if i > 0 else None,
			)

			output_embeds, _, attention_mask = cross_attn(
				hidden_states=output_embeds,
				key_value_states=input_embeds,
				cache_position=input_indices,
				attention_mask=attention_mask,
				position_bias=attention_mask if i > 0 else None,
			)

			output_embeds = fc(output_embeds)

		output_embeds = self.t5.decoder.final_layer_norm(output_embeds)
		output_embeds = self.t5.decoder.dropout(output_embeds)

		if self.t5.config.tie_word_embeddings:
			output_embeds = output_embeds * (self.t5.model_dim ** -0.5)

		logits = self.t5.lm_head(output_embeds)
		return logits

	def forward(
		self,
		input_ids: Tensor,
		attention_mask: Tensor,
		decoder_input_ids: Tensor,
		decoder_attention_mask: Tensor,
	) -> Output:
		# Encode
		memory = self.encode(
			input_ids=input_ids,
			attention_mask=attention_mask,
		)

		attention_mask = torch.where(attention_mask, 0, -torch.inf)
		attention_mask = memory.prune_masks[:, -1] + attention_mask

		# Decode
		logits = self.decode(
			input_embeds=memory.input_embeds,
			attention_mask=attention_mask,

			decoder_input_ids=decoder_input_ids,
			decoder_attention_mask=decoder_attention_mask,
		)

		output = AutoEncoder.Output(
			logits=logits,
			prune_probs=memory.prune_probs,
			past_key_values=memory.input_embeds,
		)

		return output
