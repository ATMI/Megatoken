import copy
import math
from typing import Optional, Tuple, Union, Callable

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


class GatedEncoder(nn.Module):
	def __init__(
			self,
			encoder_layer: "GatedEncoderLayer",
			num_layers: int,
			norm: Optional[nn.Module] = None,
			mask_check: bool = True,
	):
		super().__init__()

		self.norm = norm
		self.num_layers = num_layers

		self.layers = nn.ModuleList([
			copy.deepcopy(encoder_layer)
			for _ in range(num_layers)
		])

	def compression_ratio(
			self,
			src_pad: Tensor,
			comp_pad: Tensor
	) -> float:
		"""
		Calculates the compression ratio of gate.
		:param src_pad: Pad mask of initial sequence
		:param comp_pad: Pad mask of compressed sequence
		:return: Compression ratio
		"""
		src_len = src_pad.size(1) - src_pad.sum(dim=1)
		comp_len = comp_pad.size(1) - comp_pad.sum(dim=1)
		ratio = (comp_len / src_len).mean(dim=0).item()
		return ratio

	def forward(
			self,
			src: Tensor,
			mask: Optional[Tensor] = None,
			src_key_padding_mask: Optional[Tensor] = None,
			is_causal: Optional[bool] = None,
	) -> Tuple[Tensor, Tensor, Tensor]:
		"""

		:param src: Input sequence
		:param mask: Attention mask
		:param src_key_padding_mask: Padding mask
		:param is_causal: Hint to apply attention mask
		:return: Tuple | (Compressed_Seq, Pad_Mask, Comp. ratios of N layers)
		"""
		output = src
		pad_mask = src_key_padding_mask

		# Tensor with mean compression ratio for each layer
		comp_ratios = torch.zeros(self.num_layers, dtype=torch.float)

		for l_num, layer in enumerate(self.layers):
			comp_out, comp_pad_mask = layer(
				output,
				src_mask=mask,
				is_causal=is_causal,
				src_key_padding_mask=pad_mask,
			)

			ratio = self.compression_ratio(pad_mask, comp_pad_mask)
			comp_ratios[l_num] = ratio

			pad_mask = comp_pad_mask
			output = comp_out

		if self.norm is not None:
			output = self.norm(output)

		return output, pad_mask, comp_ratios






class GatedEncoderLayer(nn.TransformerEncoderLayer):
	def __init__(
			self,
			d_model: int,
			nhead: int,
			comp_gate,
			dim_feedforward: int = 2048,
			dropout: float = 0.1,
			activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
			batch_first: bool = True,
	):
		super().__init__(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			activation=activation,
			batch_first=batch_first,
		)
		self.gate_layer = comp_gate

	def forward(
			self,
			src: Tensor,
			src_mask: Optional[Tensor] = None,
			src_key_padding_mask: Optional[Tensor] = None,
			is_causal: bool = False,
	) -> Tuple[Tensor, Tensor]:
		"""
		Forward pass. Calling default TransformerEncoderLayer forward pass
		and apply compression gate after.
		:param src: Input sequence
		:param src_mask: Attention mask
		:param src_key_padding_mask: Padding mask
		:param is_causal: Hint to apply attention mask
		:return: Tuple | (Compressed_Seq, Pad_Mask)
		"""
		enc_out = super().forward(
			src=src,
			src_mask=src_mask,
			src_key_padding_mask=src_key_padding_mask,
			is_causal=is_causal,
		)

		out, pad_mask = self.gate_layer(enc_out, src_key_padding_mask)

		return out, pad_mask












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
