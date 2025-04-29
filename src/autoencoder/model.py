from dataclasses import dataclass
from typing import Optional

import torch
from torch import LongTensor, FloatTensor, BoolTensor
from transformers import T5ForConditionalGeneration, T5Config, AutoModel, AutoConfig
from transformers.modeling_outputs import Seq2SeqLMOutput

from .decoder import Decoder
from .encoder import Encoder


class AutoEncoderConfig(T5Config):
	model_type = "t5-autoencoder"
	decoder_visibility = 5
	prune_temperature = 0.1
	prune_bias = 5
	ign_token_id = -100

	def __init__(self, **kwargs):
		super(AutoEncoderConfig, self).__init__(**kwargs)


class AutoEncoder(T5ForConditionalGeneration):
	config_class = AutoEncoderConfig

	@dataclass
	class Output(Seq2SeqLMOutput):
		prune_probs: FloatTensor = None

	@property
	def eos_token(self) -> int:
		return self.config.eos_token_id

	@property
	def pad_token(self) -> int:
		return self.config.pad_token_id

	@property
	def ign_token(self) -> int:
		return self.config.ign_token_id

	def __init__(self, config):
		super().__init__(config)

		self.encoder.__class__ = Encoder
		self.decoder.__class__ = Decoder

		self.encoder.init()
		self.decoder.init()

	def forward(
		self,
		input_ids: Optional[LongTensor] = None,
		attention_mask: Optional[FloatTensor] = None,

		decoder_input_ids: Optional[LongTensor] = None,
		decoder_attention_mask: Optional[BoolTensor] = None,

		encoder_outputs: Optional[Encoder.Output] = None,
		**kwargs,
	) -> Output:
		# Encode
		if encoder_outputs is None:
			encoder_outputs = self.encoder(
				input_ids=input_ids,
				attention_mask=attention_mask,
			)

		encoder_embeds = encoder_outputs.last_hidden_state
		encoder_attn_mask = torch.zeros_like(attention_mask, dtype=torch.float)
		encoder_attn_mask.masked_fill_(attention_mask.eq(0), -torch.inf)
		encoder_attn_mask = encoder_attn_mask + encoder_outputs.prune_masks[:, -1]
		decoder_attn_mask = decoder_attention_mask

		# Decode
		decoder_embeds = self.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attn_mask,

			encoder_hidden_states=encoder_embeds,
			encoder_attention_mask=encoder_attn_mask,
		)

		if self.config.tie_word_embeddings:
			decoder_embeds = decoder_embeds * (self.model_dim ** -0.5)
		logits = self.lm_head(decoder_embeds)

		output = AutoEncoder.Output(
			logits=logits,
			prune_probs=encoder_outputs.prune_probs,
			past_key_values=encoder_outputs.past_key_values,
		)

		return output


AutoConfig.register("t5-autoencoder", AutoEncoderConfig)
AutoModel.register(AutoEncoderConfig, AutoEncoder)
