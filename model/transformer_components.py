import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class AbsolutePositionalEncoding(nn.Module):
	def __init__(
			self,
			d_model: int,
			dropout: float = 0.1,
			max_len: int = 5000
	):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(1, max_len, d_model)
		pe[0, :, 0::2] = torch.sin(position * div_term)
		pe[0, :, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x: Tensor) -> Tensor:
		"""
		Arguments:
			x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
		"""
		x = x + self.pe[:, :x.size(1)]
		return self.dropout(x)



# FIXME: change batch-size dim
# Rotary embeds from LLama-2 (https://github.com/meta-llama/llama/blob/main/llama/model.py#L80)
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
	"""
	Reshape frequency tensor for broadcasting it with another tensor.

	This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
	for the purpose of broadcasting the frequency tensor during element-wise operations.

	Args:
		freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
		x (torch.Tensor): Target tensor for broadcasting compatibility.

	Returns:
		torch.Tensor: Reshaped frequency tensor.

	Raises:
		AssertionError: If the frequency tensor doesn't match the expected shape.
		AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
	"""
	ndim = x.ndim
	assert 1 < ndim
	assert freqs_cis.shape == (x.shape[1], x.shape[-1])
	shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
	return freqs_cis.view(*shape)


def apply_rotary_emb(
		xq: torch.Tensor,
		xk: torch.Tensor,
		freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Apply rotary embeddings to input tensors using the given frequency tensor.

	This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
	frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
	is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
	returned as real tensors.

	Args:
		xq (torch.Tensor): Query tensor to apply rotary embeddings.
		xk (torch.Tensor): Key tensor to apply rotary embeddings.
		freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

	Returns:
		Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
	"""
	xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
	xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
	freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
	xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
	xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
	return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
	"""
	Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

	This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
	and the end index 'end'. The 'theta' parameter scales the frequencies.
	The returned tensor contains complex values in complex64 data type.

	Args:
		dim (int): Dimension of the frequency tensor.
		end (int): End index for precomputing frequencies.
		theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

	Returns:
		torch.Tensor: Precomputed frequency tensor with complex exponentials.
	"""
	freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
	t = torch.arange(end, device=freqs.device)  # type: ignore
	freqs = torch.outer(t, freqs).float()  # type: ignore
	freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
	return freqs_cis


class RotaryEncoderLayer(nn.TransformerEncoderLayer):
	def __init__(
			self,
			d_model: int,
			nhead: int,
			dim_feedforward: int = 2048,
			dropout: float = 0.1,
			batch_first: bool = False,
	) -> None:
		super(RotaryEncoderLayer, self).__init__(
			d_model,
			nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=batch_first,
			activation=F.gelu,
		)
		# Limiting the context length to precompute the rotary embeddings
		token_limit = 1024
		self.pos_encodings = precompute_freqs_cis(d_model // nhead, token_limit)

	def _sa_block(
			self,
			x: Tensor,
			attn_mask: Optional[Tensor],
			key_padding_mask: Optional[Tensor],
			is_causal: bool = False,
	) -> Tensor:
		seq_len = x.size(1)  # FIXME: Works only for batch_first=True
		freqs_cis = self.pos_encodings[:seq_len]
		qx, kx = apply_rotary_emb(x, x, freqs_cis)

		x = self.self_attn(
			qx,
			kx,
			x,
			attn_mask=attn_mask,
			key_padding_mask=key_padding_mask,
			need_weights=False,
			is_causal=is_causal,
		)[0]
		return self.dropout1(x)


class RotaryDecoderLayer(nn.TransformerDecoderLayer):
	def __init__(
			self,
			d_model: int,
			nhead: int,
			dim_feedforward: int = 2048,
			dropout: float = 0.1,
			batch_first: bool = False,
	):
		super(RotaryDecoderLayer, self).__init__(
			d_model,
			nhead,
			dim_feedforward =dim_feedforward,
			dropout=dropout,
			batch_first=batch_first,
			activation=F.gelu,
		)
		# Limiting the context length to precompute the rotary embeddings
		token_limit = 4096
		self.pos_encodings = precompute_freqs_cis(d_model // nhead, token_limit)

	def _sa_block(
			self,
			x: Tensor,
			attn_mask: Optional[Tensor],
			key_padding_mask: Optional[Tensor],
			is_causal: bool = False,
	) -> Tensor:
		# qx, kx = apply_rotary_emb(x, x, self.pos_encodings)
		# vx = x

		x = self.self_attn(
			x,
			x,
			x,
			attn_mask=attn_mask,
			key_padding_mask=key_padding_mask,
			is_causal=is_causal,
			need_weights=False,
		)[0]
		return self.dropout1(x)
	#
	# def _mha_block(
	# 		self,
	# 		x: Tensor,
	# 		mem: Tensor,
	# 		attn_mask: Optional[Tensor],
	# 		key_padding_mask: Optional[Tensor],
	# 		is_causal: bool = False,
	# ) -> Tensor:
	# 	x = self.multihead_attn(
	# 		x,
	# 		mem,
	# 		mem,
	# 		attn_mask=attn_mask,
	# 		key_padding_mask=key_padding_mask,
	# 		is_causal=is_causal,
	# 		need_weights=False,
	# 	)[0]
	# 	return self.dropout2(x)
