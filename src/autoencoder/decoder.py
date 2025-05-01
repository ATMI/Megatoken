import torch
from transformers.models.t5.modeling_t5 import T5Stack


class Decoder(T5Stack):
	def __init__(self, config, embed_tokens=None):
		super().__init__(config, embed_tokens)
		self.visibility_mask = None

	def init(self):
		max_length = self.config.n_positions
		visibility = self.config.decoder_visibility

		if visibility > 0:
			mask = torch.full((max_length, max_length), -torch.inf)
			for i in range(max_length):
				start = max(0, i - visibility)
				mask[i, start:i + 1] = 0
		else:
			mask = torch.triu(torch.ones(max_length, max_length), diagonal=1)
			mask = mask.masked_fill(mask == 1, -torch.inf)

		self.register_buffer("decoder_indices", torch.arange(max_length, dtype=torch.long), False)
		self.register_buffer("encoder_indices", torch.arange(max_length, dtype=torch.long), False)
		self.register_buffer("decoder_attn_mask", mask, False)

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		encoder_hidden_states=None,
		encoder_attention_mask=None,
		**kwargs,
	):
		encoder_embeds = encoder_hidden_states
		encoder_embeds_length = encoder_embeds.size(1)
		decoder_embeds_length = input_ids.size(1)

		encoder_embeds_indices = self.encoder_indices[:encoder_embeds_length]
		decoder_embeds_indices = self.decoder_indices[:decoder_embeds_length]

		encoder_attn_mask = encoder_attention_mask[:, None, None, :]
		decoder_attn_mask = self.decoder_attn_mask[None, None, :decoder_embeds_length, :decoder_embeds_length]

		decoder_embeds = self.embed_tokens(input_ids)
		# decoder_embeds = self.dropout(decoder_embeds)

		for i, decoder_layer in enumerate(self.block):
			self_attn, cross_attn, fc = decoder_layer.layer

			decoder_embeds, _, decoder_attn_mask = self_attn(
				hidden_states=decoder_embeds,
				cache_position=decoder_embeds_indices,
				attention_mask=decoder_attn_mask,
				position_bias=decoder_attn_mask if i > 0 else None,
			)

			decoder_embeds, _, encoder_attn_mask = cross_attn(
				hidden_states=decoder_embeds,
				key_value_states=encoder_embeds,
				cache_position=encoder_embeds_indices,
				attention_mask=encoder_attn_mask,
				position_bias=encoder_attn_mask if i > 0 else None,
			)

			decoder_embeds = fc(decoder_embeds)

		decoder_embeds = self.final_layer_norm(decoder_embeds)
		decoder_embeds = self.dropout(decoder_embeds)

		return decoder_embeds
