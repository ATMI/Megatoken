import torch
from transformers.models.t5.modeling_t5 import T5Stack


class Decoder(T5Stack):
	def __init__(self, config, embed_tokens=None):
		super().__init__(config, embed_tokens)
		self.visibility_mask = None

	def init(self):
		visibility = self.config.decoder_visibility
		if visibility <= 0:
			self.visibility_mask = None
			return

		max_length = self.config.n_positions
		visibility_mask = torch.full((max_length, max_length), -torch.inf)

		for i in range(max_length):
			start = max(0, i - visibility)
			visibility_mask[i, start:i + 1] = 0

		self.visibility_mask = visibility_mask

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		encoder_hidden_states=None,
		encoder_attention_mask=None,
		**kwargs,
		# inputs_embeds=None,
		# head_mask=None,
		# cross_attn_head_mask=None,
		# past_key_values=None,
		# use_cache=None,
		# output_attentions=None,
		# output_hidden_states=None,
		# return_dict=None,
		# cache_position=None
	):
		device = input_ids.device
		encoder_embeds = encoder_hidden_states

		encoder_embeds_length = encoder_embeds.size(1)
		decoder_embeds_length = input_ids.size(1)

		encoder_embeds_indices = torch.arange(encoder_embeds_length, device=device)
		decoder_embeds_indices = torch.arange(decoder_embeds_length, device=device)

		encoder_attn_mask = encoder_attention_mask
		encoder_attn_mask = encoder_attn_mask.unsqueeze(1)
		encoder_attn_mask = encoder_attn_mask.unsqueeze(1)

		decoder_attn_mask = attention_mask
		if decoder_attn_mask is None and self.visibility_mask is not None:
			if self.visibility_mask.device != device:
				self.visibility_mask = self.visibility_mask.to(device)

			decoder_attn_mask = self.visibility_mask[:decoder_embeds_length, :decoder_embeds_length]
			decoder_attn_mask = decoder_attn_mask.unsqueeze(0)
			decoder_attn_mask = decoder_attn_mask.unsqueeze(0)

		decoder_embeds = self.embed_tokens(input_ids)
		decoder_embeds = self.dropout(decoder_embeds)

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
