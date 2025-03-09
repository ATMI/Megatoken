from typing import Type

import torch
from torch import nn
import torch.nn.functional as F

from Megatoken.encoder import SoftGate
from Megatoken.utils.config import load_config
from Megatoken.model.transformer_components import RotaryEncoderLayer, RotaryDecoderLayer, AbsolutePositionalEncoding


class coBERT(nn.Module):
	def __init__(
			self,
			cfg,
			c_gate: Type,
			vocab_size: int,
			pad_idx: int,
	):
		"""

		:param cfg: Config file
		:param c_gate: Compression Gate
		:param vocab_size: Size of the vocabulary
		:param pad_idx: Index of padding token
		"""
		super(coBERT, self).__init__()
		self.cfg = cfg
		self.max_seq_len = cfg.max_seq_len

		self.pos_enc = AbsolutePositionalEncoding(cfg.embed_dim)

		# FIXME passing uninitialized class might be bad practice
		self.gate = c_gate(
			embed_dim=cfg.embed_dim
		)

		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=cfg.embed_dim,
			padding_idx=pad_idx,
		)

		self.encoder = nn.TransformerEncoder(
			encoder_layer=nn.TransformerEncoderLayer(
				d_model=cfg.embed_dim,
				nhead=cfg.encoder.n_head,
				dim_feedforward=cfg.encoder.dim_fc,
				dropout=cfg.encoder.dropout,
				batch_first=cfg.batch_first,
				activation=F.gelu,
			),
			num_layers=cfg.encoder.num_layers,
		)

		self.decoder = nn.TransformerDecoder(
			decoder_layer=nn.TransformerDecoderLayer(
				d_model=cfg.embed_dim,
				nhead=cfg.decoder.n_head,
				dim_feedforward=cfg.decoder.dim_fc,
				dropout=cfg.encoder.dropout,
				batch_first=cfg.batch_first,
				activation=F.gelu,
			),
			num_layers=cfg.decoder.num_layers,
		)

	def forward(
			self,
			seq,
			seq_pad_mask=None,
			tgt_mask=None,
	):
		r"""

		:param seq: Input Sequence
		:param seq_pad_mask: Padding mask of initial sequence
		:param tgt_mask: Attention mask of target sequence in decoder (self-attention)
		:return:
		"""

		_pos_seq = self.pos_enc(seq)

		enc_seq = self.encoder(
			_pos_seq,
			src_key_padding_mask=seq_pad_mask,
		)

		comp_seq, comp_pad_mask = self.gate(enc_seq, seq_pad_mask)

		# TODO read again about causal mask
		out = self.decoder(
			seq,
			comp_seq,
			tgt_mask=tgt_mask,
			tgt_key_padding_mask=seq_pad_mask,
			memory_key_padding_mask=comp_pad_mask,
			tgt_is_causal=True,
		)

		return out



if __name__ == "__main__":
	cfg = load_config("../configs/test.yaml")

	model = coBERT(
		cfg=cfg,
		c_gate=SoftGate,
		vocab_size=100,
		pad_idx=1,
	)

	x = torch.rand(10, 32, cfg.embed_dim)
	padding_mask = torch.randint(0, 2, (10, 32)).bool()
	mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), dtype=torch.bool)
	out = model.forward(x, tgt_mask=mask, seq_pad_mask=padding_mask)
	print(out.shape)

