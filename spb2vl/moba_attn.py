# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RotaryEmbedding#, RMSNorm
from fla.ops.utils.index import prepare_lens_from_mask
from transformers.integrations import use_kernel_forward_from_hub

from moba import moba_attn_varlen

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [batch, seq, heads, head_dim]
    # cos, sin: [batch, seq, head_dim]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

@use_kernel_forward_from_hub("RMSNorm")
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class MoBAAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        head_dim: int = 128,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10000.,
        moba_chunk_size: int = 1024,
        moba_topk: int = 4,
        max_position_embeddings: int | None = None,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = head_dim
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.moba_chunk_size = moba_chunk_size
        self.moba_topk = moba_topk
        self.layer_idx = layer_idx
        self.max_position_embeddings = max_position_embeddings

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        self.q_proj = nn.Linear(self.hidden_size, self.q_dim, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.q_dim, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()
        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
            # moba_attn_varlen only supports bfloat16 or fp16
            q, k = q.to(torch.bfloat16), k.to(torch.bfloat16)
        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens')

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        # q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos=cos, sin=sin)
            
        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
        
        k = torch.repeat_interleave(k, self.num_kv_groups, dim=2)
        v = torch.repeat_interleave(v, self.num_kv_groups, dim=2)

        kv_len = k.size(1)
        if q_len == kv_len:
            # Contains at least one padding token in the sequence
            if attention_mask is not None:
                if q.shape[1] == 1 and self.window_size is not None:
                    attention_mask = attention_mask[:, -self.window_size:]
                q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v), attention_mask, q_len)
                cu_seqlens_q, cu_seqlens_k = cu_seqlens
                max_seqlen_q, max_seqlen_k = max_seq_lens
                o = moba_attn_varlen(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens=cu_seqlens_k,
                    max_seqlen=max_seqlen_k,
                    moba_chunk_size=self.moba_chunk_size,
                    moba_topk=self.moba_topk,
                )
                o = pad_input(o, indices_q, batch_size, q_len)
            elif cu_seqlens is not None:
                o = moba_attn_varlen(
                    q=q.squeeze(0),
                    k=k.squeeze(0),
                    v=v.squeeze(0),
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    moba_chunk_size=self.moba_chunk_size,
                    moba_topk=self.moba_topk,
                ).unsqueeze(0)
            else:
                cu_seqlens_k = torch.cumsum(
                    torch.tensor([0] + [kv_len] * batch_size, device=q.device),
                    dim=0,
                    dtype=torch.int32,
                )
                q, k, v = [rearrange(src, 'b l h d -> (b l) h d').contiguous() for src in [q, k, v]]
                o = moba_attn_varlen(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens=cu_seqlens_k,
                    max_seqlen=kv_len,
                    moba_chunk_size=self.moba_chunk_size,
                    moba_topk=self.moba_topk,
                )
                o = rearrange(o, "(b l) h d -> b l h d", b=batch_size).contiguous()

        else:
            if attention_mask is not None:
                if q.shape[1] == 1 and self.window_size is not None:
                    attention_mask = attention_mask[:, -self.window_size:]
                q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v), attention_mask, q_len)
                cu_seqlens_q, cu_seqlens_k = cu_seqlens
                max_seqlen_q, max_seqlen_k = max_seq_lens
                o = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
                )
                o = pad_input(o, indices_q, batch_size, q_len)
            elif cu_seqlens is not None:
                o = flash_attn_varlen_func(
                    q.squeeze(0), k.squeeze(0), v.squeeze(0),
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
                ).unsqueeze(0)
            else:
                o = flash_attn_func(
                    q, k, v,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
                )

        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values
