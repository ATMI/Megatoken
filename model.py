from typing import Tuple

import torch
from torch import nn, Tensor
from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer


class Model(nn.Module):
	def __init__(self, config):
		super(Model, self).__init__()
		self.t5 = T5ForConditionalGeneration.from_pretrained(config)

	def encode(
		self,
		tokens: Tensor,
		attn_mask: Tensor,
	) -> Tuple[Tensor, Tensor]:
		batch_size = tokens.size(0)
		seq_length = tokens.size(1)

		embeds = self.t5.encoder.embed_tokens(tokens)
		embeds = self.t5.encoder.dropout(embeds)

		if attn_mask is None:
			attn_mask = torch.zeros(
				size=(batch_size, 1, seq_length, seq_length),
				device=embeds.device,
				dtype=embeds.dtype,
			)
		else:
			attn_mask = attn_mask[:, None, :]
			attn_mask = attn_mask.to(dtype=embeds.dtype)

		# Kinda strange variable with cache disabled,
		# but it's used to calculate the position bias
		# in HF spaghetti. Trust me :)
		cache_position = torch.arange(seq_length, device=embeds.device)
		position_bias = None

		for i, encoder_layer in enumerate(self.t5.encoder.block):
			embeds, position_bias = encoder_layer(
				hidden_states=embeds,
				attention_mask=attn_mask,
				position_bias=position_bias,
				cache_position=cache_position,
			)

		embeds = self.t5.encoder.final_layer_norm(embeds)
		embeds = self.t5.encoder.dropout(embeds)

		attn_mask = attn_mask[:, :, 0, :].unsqueeze(2)
		return embeds, attn_mask

	def decode(
		self,
		input_tokens: Tensor,
		memory_embeds: Tensor,

		self_attn_mask: Tensor,
		cross_attn_mask: Tensor,
	):
		input_length = input_tokens.size(1)
		input_embeds = self.t5.decoder.embed_tokens(input_tokens)
		input_embeds = self.t5.decoder.dropout(input_embeds)

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

	def forward(
		self,
		memory_tokens: Tensor,
		memory_attn_mask: Tensor | None,

		input_tokens: Tensor | None,
		input_attn_mask: Tensor | None,
	) -> Tensor:
		memory_embeds, memory_attn_mask = self.encode(
			tokens=memory_tokens,
			attn_mask=memory_attn_mask,
		)

		logits = self.decode(
			input_tokens=input_tokens,
			memory_embeds=memory_embeds,

			self_attn_mask=input_attn_mask,
			cross_attn_mask=memory_attn_mask,
		)

		return logits


def pad_attention_mask(pad: Tensor) -> Tensor:
	mask = torch.where(pad > 0, 0, -torch.inf)
	mask = mask.unsqueeze(1)
	mask = mask.repeat(1, pad.size(1), 1)
	return mask


def main():
	model_name = "google/flan-t5-small"
	device = torch.device("cuda")

	custom_model = Model(model_name).to(device)
	original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

	# Fix the seed
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)

	tokenizer = AutoTokenizer.from_pretrained(model_name)

	input_text = ["This is a test.", "How are you?"]  # Sample inputs
	inputs = tokenizer(input_text, return_tensors="pt", padding=True)

	input_ids = inputs['input_ids'].to(device)
	pad_mask = inputs['attention_mask'].to(device)

	decoder_input_ids = torch.full((input_ids.shape[0], 3), tokenizer.pad_token_id, dtype=torch.long).to(device)
	original_output = original_model(
		input_ids=input_ids,
		attention_mask=pad_mask,
		decoder_input_ids=decoder_input_ids
	)
	custom_output = custom_model.forward(
		memory_tokens=input_ids,
		memory_attn_mask=pad_attention_mask(pad_mask),

		input_tokens=decoder_input_ids,
		input_attn_mask=None,
	)

	print("---final---")
	print("Custom Model Output:", custom_output)
	print("Original Model Output:", original_output.logits)


if __name__ == "__main__":
	main()
