import torch
from torch import nn

from .transformer_components import RotaryEncoderLayer, RotaryDecoderLayer, precompute_freqs_cis


class coBERT(nn.Module):
	def __init__(
			self,
			cfg,
			c_gate: nn.Module,
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

		# FIXME passing uninitialized class might be bad practice
		self.gate = c_gate

		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=cfg.embed_dim,
			padding_idx=pad_idx,
		)

		self.encoder = nn.TransformerEncoder(
			encoder_layer=RotaryEncoderLayer(
				d_model=cfg.embed_dim,
				nhead=cfg.encoder.n_head,
				dim_feedforward=cfg.encoder.dim_fc,
				dropout=cfg.encoder.dropout,
				batch_first=cfg.batch_first,
			),
			num_layers=cfg.encoder.num_layers,
		)

		self.decoder = nn.TransformerDecoder(
			decoder_layer=RotaryDecoderLayer(
				d_model=cfg.embed_dim,
				nhead=cfg.decoder.n_head,
				dim_feedforward=cfg.decoder.dim_fc,
				dropout=cfg.encoder.dropout,
				batch_first=cfg.batch_first,
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
		x = self.encoder(
			seq,
			src_key_padding_mask=seq_pad_mask,
		)

		seq_pad_mask = seq_pad_mask.t()

		comp_seq, comp_pad_mask = self.gate(x, seq_pad_mask)

		seq_pad_mask = seq_pad_mask.t()

		# TODO read again about causal mask
		out = self.decoder(
			seq,
			comp_seq,
			tgt_mask=tgt_mask,
			tgt_key_padding_mask=seq_pad_mask,
			memory_key_padding_mask=comp_pad_mask,
			# tgt_is_causal=True,
		)

		return out
