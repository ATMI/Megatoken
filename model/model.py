from torch import nn
from .encoder import RotaryEncoder


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
		:param c_gate:
		:param vocab_size:
		:param pad_idx:
		"""
		super(coBERT, self).__init__()

		self.max_seq_len = self.config.max_seq_len

		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=cfg.embed_dim,
			padding_idx=pad_idx,
		)

		self.encoder = RotaryEncoder(
			encoder_layer=nn.TransformerEncoderLayer(
				d_model=cfg.embed_dim,
				nhead=cfg.encoder.n_head,
				dim_feedforward=cfg.encoder.dim_ff,
				dropout=cfg.encoder.dropout,
			),
			num_layers=cfg.encoder.num_layers,
		)

		self.decoder = nn.TransformerDecoder(
			decoder_layer=nn.TransformerDecoderLayer(
				d_model=cfg.embed_dim,
				nhead=cfg.decoder.n_head,
				dim_feedforward=cfg.decoder.dim_ff,
				dropout=cfg.encoder.dropout,
			),
			num_layers=cfg.decoder.num_layers,
		)

		# FIXME passing uninitialized class might be bad practice
		self.gate = c_gate()

	def forward(
			self,
			seq,
			seq_pad_mask=None,
			tgt_mask=None,
	):
		r"""

		:param seq: Input Sequence
		:param seq_pad_mask: Padding mask of initial sequence
		:param tgt_mask: Attention mask of target sequence (self-attention)
		:return:
		"""
		x = self.encoder(
			seq,
			src_key_padding_mask=seq_pad_mask,
		)

		comp_seq, comp_pad_mask = self.gate(x)

		out = self.decoder(
			seq,
			comp_seq,
			tgt_mask=tgt_mask,
			tgt_key_padding_mask=seq_pad_mask,
			memory_key_padding_mask=comp_pad_mask,
		)

		return out
