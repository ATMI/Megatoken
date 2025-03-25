from typing import Tuple

import torch
from torch import nn, Tensor
from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer


class Gate(nn.Module):
	def __init__(self, bias: float, temperature: float):
		super(Gate, self).__init__()
		self.bias = bias
		self.temperature = temperature

	def forward(
		self,
		embeds: Tensor,
	) -> Tuple[Tensor, Tensor]:
		gates = (embeds[:, :, 0] + self.bias) / self.temperature
		gates = gates.sigmoid()
		gates = gates.clamp(min=1e-10, max=1)
		embeds = embeds * gates.unsqueeze(2)
		return embeds, gates


class Model(nn.Module):
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
			for _ in self.t5.encoder.block
		)

	def encode(
		self,
		tokens: Tensor,
		attn_mask: Tensor,
	) -> Tuple[Tensor, Tensor]:
		device = tokens.device
		seq_length = tokens.size(1)

		embeds = self.t5.encoder.embed_tokens(tokens)
		embeds = self.t5.encoder.dropout(embeds)

		gate_mask = torch.zeros(seq_length, device=device)
		diag_indices = torch.arange(seq_length, device=device)

		# Adding dimension per attention head for HF compatibility
		attn_mask = attn_mask.unsqueeze(1)

		# Kinda strange variable with cache disabled,
		# but it's used to calculate the position bias
		# in HF spaghetti. Trust me :)
		cache_position = torch.arange(seq_length, device=device)
		position_bias = None

		for encoder_layer, gate_layer in zip(self.t5.encoder.block, self.gates):
			embeds, position_bias = encoder_layer(
				hidden_states=embeds,
				attention_mask=attn_mask,
				position_bias=position_bias,
				cache_position=cache_position,
			)

			embeds, gates = gate_layer(embeds)
			gates = gates.log()
			gate_mask = gate_mask + gates

			gates = gates.unsqueeze(2).repeat(1, 1, seq_length)
			gates[:, diag_indices, diag_indices] = 0
			gates = gates.unsqueeze(1)

			attn_mask = attn_mask + gates

		embeds = self.t5.encoder.final_layer_norm(embeds)
		embeds = self.t5.encoder.dropout(embeds)

		return embeds, gate_mask

	def decode(
		self,
		input_tokens: Tensor,
		memory_embeds: Tensor,

		self_attn_mask: Tensor,
		cross_attn_mask: Tensor,
	):
		# Code here is similar to encode in some parts,
		# maybe it worth to do something with it or not :)
		input_length = input_tokens.size(1)
		input_embeds = self.t5.decoder.embed_tokens(input_tokens)
		input_embeds = self.t5.decoder.dropout(input_embeds)

		self_attn_mask = self_attn_mask.unsqueeze(1)
		cross_attn_mask = cross_attn_mask.unsqueeze(1)

		self_position_bias = None
		cross_position_bias = None
		cache_position = torch.arange(input_length, device=input_embeds.device)

		for i, decoder_layer in enumerate(self.t5.decoder.block):
			self_attn, cross_attn, fc = decoder_layer.layer

			input_embeds, _, self_position_bias = self_attn(
				hidden_states=input_embeds,
				attention_mask=self_attn_mask,
				position_bias=self_position_bias,
				cache_position=cache_position,
			)

			input_embeds, _, cross_position_bias = cross_attn(
				hidden_states=input_embeds,
				key_value_states=memory_embeds,
				attention_mask=cross_attn_mask,
				position_bias=cross_position_bias,
				cache_position=cache_position,
			)

			input_embeds = fc(input_embeds)

		input_embeds = self.t5.decoder.final_layer_norm(input_embeds)
		input_embeds = self.t5.decoder.dropout(input_embeds)

		if self.t5.config.tie_word_embeddings:
			input_embeds = input_embeds * (self.t5.model_dim ** -0.5)

		logits = self.t5.lm_head(input_embeds)
		return logits

	@staticmethod
	def pad_attention_mask(
		attn_mask: Tensor | None,
		pad_mask: Tensor,
	) -> Tensor:
		pad_mask = torch.where(pad_mask > 0, 0, -torch.inf)
		pad_mask = pad_mask.unsqueeze(1)

		if attn_mask is None:
			pad_mask = pad_mask.repeat(1, pad_mask.size(2), 1)
			attn_mask = pad_mask
		else:
			attn_mask = attn_mask + pad_mask

		return attn_mask

	def forward(
		self,
		memory_tokens: Tensor,
		memory_pad_mask: Tensor | None,
		memory_attn_mask: Tensor | None,

		input_tokens: Tensor,
		input_pad_mask: Tensor | None,
		input_attn_mask: Tensor | None,
	) -> Tensor:
		# Encode
		if memory_pad_mask is None:
			memory_pad_mask = torch.ones_like(memory_tokens)

		memory_attn_mask = self.pad_attention_mask(
			attn_mask=memory_attn_mask,
			pad_mask=memory_pad_mask,
		)

		memory_embeds, memory_gate_mask = self.encode(
			attn_mask=memory_attn_mask,
			tokens=memory_tokens,
		)

		# Decode
		if input_pad_mask is None:
			input_pad_mask = torch.ones_like(input_tokens)

		self_attn_mask = self.pad_attention_mask(
			attn_mask=input_attn_mask,
			pad_mask=input_pad_mask,
		)

		cross_attn_mask = self.pad_attention_mask(
			attn_mask=memory_gate_mask.unsqueeze(1),
			pad_mask=memory_pad_mask,
		)

		logits = self.decode(
			input_tokens=input_tokens,
			memory_embeds=memory_embeds,

			self_attn_mask=self_attn_mask,
			cross_attn_mask=cross_attn_mask,
		)

		return logits


def main():
	model_name = "google/flan-t5-small"
	device = torch.device("cuda")

	custom_model = Model(name=model_name, bias=5, temperature=0.1).to(device)
	original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

	# Fix the seed
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)

	tokenizer = AutoTokenizer.from_pretrained(model_name)

	input_text = ["This is a test. Asdasd", "How are you?"]  # Sample inputs
	inputs = tokenizer(input_text, return_tensors="pt", padding=True)

	input_ids = inputs['input_ids'].to(device)
	pad_mask = inputs['attention_mask'].to(device)

	decoder_input_ids = torch.full((input_ids.shape[0], 1), 234, dtype=torch.long).to(device)
	original_output = original_model(
		input_ids=input_ids,
		attention_mask=pad_mask,
		decoder_input_ids=decoder_input_ids
	)
	custom_output = custom_model.forward(
		memory_tokens=input_ids,
		memory_pad_mask=pad_mask,
		memory_attn_mask=None,

		input_tokens=decoder_input_ids,
		input_attn_mask=None,
		input_pad_mask=None,
	)

	print("---final---")
	print("Custom Model Output:", custom_output)
	print("Original Model Output:", original_output.logits)


if __name__ == "__main__":
	main()
