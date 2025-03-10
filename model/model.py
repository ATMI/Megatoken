from typing import Type, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .encoder_gates import SoftGate
from .transformer_components import GatedEncoder, GatedEncoderLayer, AbsolutePositionalEncoding

from utils.config import load_config


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

		self.embeddings = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=cfg.embed_dim,
			padding_idx=pad_idx,
		)
		self.gate = c_gate(
			cfg.embed_dim
		)

		self.encoder = GatedEncoder(
			encoder_layer=GatedEncoderLayer(
				d_model=cfg.embed_dim,
				nhead=cfg.encoder.n_head,
				comp_gate=self.gate,
				dim_feedforward=cfg.encoder.dim_fc,
				dropout=cfg.encoder.dropout,
				activation=F.gelu,
				batch_first=cfg.batch_first,
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

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(cfg.embed_dim, vocab_size),
		)

	def forward(
			self,
			seq,
			seq_pad_mask=None,
			seq_mask=None,
	) -> Tuple[
		torch.Tensor,
		float
	]:
		r"""

		:param seq: Input Sequence
		:param seq_pad_mask: Padding mask of initial sequence
		:param seq_mask: Attention mask of target sequence in decoder (self-attention)
		:return: Tuple | (pred_seq, compression_ratio)
		"""
		x = self.embeddings(seq)
		_pos_seq = self.pos_enc(x)

		comp_seq, comp_pad_mask, ratio = self.encoder(
			_pos_seq,
			src_key_padding_mask=seq_pad_mask,
		)

		out = self.decoder(
			_pos_seq,
			comp_seq,
			tgt_mask=seq_mask,
			tgt_key_padding_mask=seq_pad_mask,
			memory_key_padding_mask=comp_pad_mask,
			tgt_is_causal=True,
		)

		out = self.classifier(out)
		return out, ratio


if __name__ == "__main__":
	cfg = load_config("../config/test.yaml")

	model = coBERT(
		cfg=cfg,
		c_gate=SoftGate,
		vocab_size=100,
		pad_idx=1,
	)

	x = torch.randint(low=0, high=100, size=(10, 32))
	padding_mask = torch.randint(0, 2, (10, 32)).bool()
	mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), dtype=torch.bool)

	out, ratio = model.forward(x, seq_mask=mask, seq_pad_mask=padding_mask)
	print(out.shape)
