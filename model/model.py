from torch import nn
from encoder import Encoder

# TODO Rotary Embeds

class coBERT(nn.Module):
	def __init__(
			self,
			cfg,
			cgate: nn.Module=None
	):
		r'''

		:param cfg: Config file
		:param cgate: Compression gate Module
		'''
		super(coBERT, self).__init__()
		self.config = cfg

		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=embed_dim,
			padding_idx=pad_idx,
		)

		self.encoder = Encoder(
			encoder_layer=nn.TransformerEncoderLayer(
				d_model=cfg.encoder.dim_size,
				nhead=cfg.encoder.n_head,
				dim_feedforward=cfg.encoder.dim_ff,
				dropout=cfg.encoder.dropout,
			),
			num_layers=cfg.encoder.num_layers,
		)

		self.decoder = nn.TransformerDecoder(
			decoder_layer=nn.TransformerDecoderLayer(
				d_model=cfg.decoder.dim_size,
				nhead=cfg.decoder.n_head,
				dim_feedforward=cfg.decoder.dim_ff,
				dropout=cfg.encoder.dropout,
			),
			num_layers=cfg.decoder.num_layers,
		)

		self.gate = cgate


	def forward(
			self,
			seq,
			seq_mask=None,
			seq_pad_mask=None,
			tgt_mask=None,
	):
		r'''

		:param seq: Input Sequence
		:param seq_mask: Attention mask of initial sequence for encoder
		:param seq_pad_mask: Padding mask of initial sequence
		:param tgt_mask: ???
		:return:
		'''
		# TODO do we pass attention mask for encoder?
		x = self.encoder(
			seq,
			mask=seq_mask,
			src_key_padding_mask=seq_pad_mask,
		)

		comp_seq, comp_mask, comp_pad_mask = self.gate(x)

		out = self.decoder(
			seq,
			comp_seq,
			tgt_mask=tgt_mask,
			memory_mask=comp_mask,
			tgt_key_padding_mask=seq_pad_mask,
			memory_key_padding_mask=comp_pad_mask,
		)
		return out


