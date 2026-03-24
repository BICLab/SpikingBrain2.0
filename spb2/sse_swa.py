# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import pad_input, unpad_input
from fla.ops.utils.index import prepare_lens_from_mask
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input

from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution, RotaryEmbedding
from fla.ops.gla import chunk_gla, fused_recurrent_gla
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.sse import prepare_sample_relpos_global_index_flat, softmax_and_mask


if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None


def sort_along_l(q, k, v, gk, beta, e, cu_seqlens, K, emulq, emulk):
    _, L, H, D = q.shape
    N = e.size(-1)
    S = len(cu_seqlens) - 1

    e = F.softmax(e, dim=-1, dtype=torch.float)
    topk_value, topk_expert = torch.topk(e, k=K, dim=2)  # [1, L, K]
    topk_value, e = topk_value.to(q.dtype), e.to(q.dtype)
    mask_w = torch.zeros_like(e, dtype=torch.bool).scatter_(dim=-1, index=topk_expert, src=torch.ones_like(topk_expert, dtype=torch.bool))
    experts_flat = topk_expert.reshape(L * K)  # [L*K]
    values_flat  = topk_value.reshape(L * K)   # [L*K]

    sample_idx_flat, relpos_flat, global_idx_flat, lengths = prepare_sample_relpos_global_index_flat(cu_seqlens, K)  # ([L*K] * 3, S)
    assert sample_idx_flat.dtype == torch.long and relpos_flat.dtype == torch.long and global_idx_flat.dtype == torch.long

    bits_pos = int(lengths.max().item()).bit_length()
    bits_exp = int((N - 1)).bit_length()
    shift_exp  = bits_pos
    shift_samp = bits_pos + bits_exp

    ## sort by (sample_idx <- expert_idx <- relpos_in_sample)
    key = (sample_idx_flat << shift_samp) | (experts_flat << shift_exp) | relpos_flat
    order = torch.argsort(key, stable=False)
    experts_sorted = experts_flat.take(order)
    sample_sorted  = sample_idx_flat.take(order)
    global_sorted  = global_idx_flat.take(order)   # gather index
    values_sorted  = values_flat.take(order)       # sorted eta
    # pos_sorted   = relpos_flat.take(order)

    ## x: [1, L, H, D] -> y: [1, L*K, H, D]
    index4gather = global_sorted[None, :, None, None].expand(1, L * K, H, D)
    if beta is None:
        q, k, v, gk = [torch.gather(x, dim=1, index=index4gather) for x in (q, k, v, gk)]  # GLA
    else:
        q, k, v = [torch.gather(x, dim=1, index=index4gather) for x in (q, k, v)]          # GDN
        gk, beta = [torch.gather(x, dim=1, index=index4gather[..., 0]) for x in (gk, beta)] 
    if emulq:
        q = q * values_sorted[None, :, None, None]
    if emulk:
        k = k * values_sorted[None, :, None, None]

    ## calculate offsets (new cu_seqlens)
    pair_id = sample_sorted * N + experts_sorted  # [L*K]
    counts = torch.bincount(pair_id, minlength=S * N)  # [S*N]
    state_sizes = counts.view(S, N)
    offsets = torch.zeros(1 + S * N, dtype=torch.long, device=q.device)
    offsets[1:] = counts.cumsum(dim=0)
    offsets = torch.unique(offsets)
    
    return q, k, v, gk, beta, e, mask_w, offsets, state_sizes, global_sorted


class SSEGLAH(nn.Module):
    """
    The layer implementaion for [SSE: Scaling Linear Attention with Sparse State Expansion](https://arxiv.org/pdf/2507.16577).

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 2.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA is applied if `num_v_heads` > `num_heads`. Default: `None`.
        mode (str, Optional):
            Which GLA kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_output_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `False`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        num_sparse_partition (int, optional):
            Number of state partitions. Default: 4.
        num_writer (int, optional):
            Top-k write size (number of writers). Default: 1.
        num_reader (int, optional):
            Top-k read size (number of readers). Default: 1.
        sse_implementation (str, optional):
            SSE implementation to use. One of `"varlen"` or `"mask"`. Default: `"varlen"`.
        use_q_softmax (bool, optional):
            Whether to apply softmax to the query. Default: `False`.
        use_k_softmax (bool, optional):
            Whether to apply softmax to the key. Default: `True`.
        emulq (bool, optional):
            Whether to use a read gate operating on the state output (Q). Default: `True`.
        emulk (bool, optional):
            Whether to use a write gate operating on the state input (KV). Default: `True`.
        gate_logit_normalizer (int, Optional):
            The normalizer for the gate logits, appied after `logsigmoid`. Default: 16.
        gate_low_rank_dim (int, Optional):
            The low rank dim for the gate projection. Default: 16.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1.,
        head_dim: int = 256,
        num_heads: int = 6,
        mode: str = 'chunk',
        use_output_gate: bool = True,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        num_sparse_partition: int = 4,
        num_writer: int = 1,
        num_reader: int = 1,
        sse_implementation: str = "varlen",
        sse_qk_relu: bool = False,
        use_q_softmax: bool = False,
        use_k_softmax: bool = True,
        emulq: bool = True,
        emulk: bool = True,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        qkv_bias: bool = False,
        # === swa configs ===
        swa_num_kv_heads: int | None = None,
        swa_qk_norm: bool = False,
        swa_dropout: float = 0.5,
        window_size: int | None = None,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int | None = None,
        # ===================
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> SSEGLAH:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.layer_idx = layer_idx

        assert num_reader < num_sparse_partition and num_writer < num_sparse_partition, \
            "num_reader and num_writer must be less than num_sparse_partition."
        assert sse_implementation in ["mask", "varlen"], \
            f"Unknown SSE implementation {sse_implementation}"

        self.num_sparse_partition = num_sparse_partition
        self.num_writer = num_writer
        self.num_reader = num_reader
        self.sse_implementation = {
            "mask": self.sse_linear_attention_mask,
            "varlen": self.sse_linear_attention_varlen,
        }[sse_implementation]

        self.use_output_gate = use_output_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.use_q_softmax = use_q_softmax
        self.use_k_softmax = use_k_softmax
        self.sse_qk_relu = sse_qk_relu
        self.emulq = emulq
        self.emulk = emulk

        # mha for sse, gqa for swa
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.sse_num_kv_heads = num_heads
        if swa_num_kv_heads is None:
            self.swa_num_kv_heads = self.num_heads
        else:
            self.swa_num_kv_heads = swa_num_kv_heads
        self.swa_num_kv_groups = num_heads // self.swa_num_kv_heads

        self.sse_head_k_dim = head_dim
        self.sse_head_v_dim = int(self.head_dim * self.expand_v)
        self.sse_key_dim = int(self.sse_num_kv_heads * self.sse_head_k_dim)
        self.sse_value_dim = int(self.sse_num_kv_heads * self.sse_head_v_dim)

        self.swa_q_dim = self.num_heads * self.head_dim
        self.swa_kv_dim = int(self.swa_num_kv_heads * self.head_dim)
        self.swa_qk_norm = swa_qk_norm
        self.qkv_bias = qkv_bias

        self.swa_dropout = swa_dropout
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # Consistency check: Ensure expand_v produces integer values
        assert self.expand_v == 1.0
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        self.sse_q_proj = nn.Linear(hidden_size, self.sse_key_dim, bias=self.qkv_bias)
        self.sse_k_proj = nn.Linear(hidden_size, self.sse_key_dim, bias=self.qkv_bias)
        self.sse_v_proj = nn.Linear(hidden_size, self.sse_value_dim, bias=self.qkv_bias)
        self.lora_q_proj = nn.Sequential(nn.Linear(hidden_size, self.sse_head_v_dim, bias=False),
                                         nn.Linear(self.sse_head_v_dim, self.sse_key_dim, bias=False))
        self.lora_k_proj = nn.Sequential(nn.Linear(hidden_size, self.sse_head_v_dim, bias=False),
                                         nn.Linear(self.sse_head_v_dim, self.sse_key_dim, bias=False))

        self.gate_logit_normalizer = gate_logit_normalizer
        self.sse_gk_proj = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
                                     nn.Linear(gate_low_rank_dim, self.sse_key_dim, bias=True))
                                     for _ in range(2)])

        self.sse_e_proj = nn.Linear(hidden_size, self.num_sparse_partition, bias=False)

        self.swa_q_proj = nn.Linear(hidden_size, self.swa_q_dim, bias=self.qkv_bias)
        self.swa_k_proj = nn.Linear(hidden_size, self.swa_kv_dim, bias=self.qkv_bias)
        self.swa_v_proj = nn.Linear(hidden_size, self.swa_kv_dim, bias=self.qkv_bias)

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d_shared = ShortConvolution(
                hidden_size=self.sse_key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )
            self.k_conv1d_shared = ShortConvolution(
                hidden_size=self.sse_key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )

        if use_output_gate:
            self.sse_g_proj = nn.Sequential(nn.Linear(hidden_size, self.sse_head_v_dim, bias=False),
                                        nn.Linear(self.sse_head_v_dim, self.sse_value_dim, bias=False))
            self.sse_o_norm = FusedRMSNormGated(self.sse_head_v_dim, eps=norm_eps)
        else:
            self.sse_o_norm = RMSNorm(self.sse_head_v_dim, eps=norm_eps)

        self.sse_o_proj = nn.Linear(self.sse_value_dim, self.hidden_size, bias=False)
        self.swa_o_proj = nn.Linear(self.swa_q_dim, self.hidden_size, bias=False)

        self.sse_merge_norm = RMSNorm(self.hidden_size, eps=norm_eps)
        self.swa_merge_norm = RMSNorm(self.hidden_size, eps=norm_eps)

        if swa_qk_norm:
            self.swa_q_norm = RMSNorm(self.head_dim, eps=norm_eps)
            self.swa_k_norm = RMSNorm(self.head_dim, eps=norm_eps)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)
    
    def sse_linear_attention_varlen(self, q1, q2, k1, k2, v, gk1, gk2, eta, recurrent_state=None, use_cache=False, cu_seqlens=None):
        """
        q1: [bsz, qlen, nhead, head_dim]
        q2: [bsz, qlen, nhead, head_dim]
        k1: [bsz, klen, nhead, head_dim]
        k2: [bsz, klen, nhead, head_dim]
        v: [bsz, klen, nhead, head_dim]
        gk1: [bsz, klen, nhead, head_dim]
        gk2: [bsz, klen, nhead, head_dim]
        eta: [bsz, klen, num_sparse_partition]
        """
        assert self.num_writer == self.num_reader, "varlen only support num_writer == num_reader"
        bsz, q_len, nhead, _ = q1.shape
        # change to inference mode.
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        # if self.training:
        #     assert mode == 'chunk', "Only chunk mode is supported in training."

        v1 = v
        v2 = v
        if cu_seqlens is None:
            cu_seqlens = torch.arange(0, (bsz + 1) * q_len, q_len, dtype=torch.int32, device=q1.device)
            q1, k1, gk1, v1 = [rearrange(src, 'b l h d -> 1 (b l) h d').contiguous() for src in [q1, k1, gk1, v]]
            q2, k2, gk2, v2 = [rearrange(src, 'b l h d -> 1 (b l) h d').contiguous() for src in [q2, k2, gk2, v]]
        S = len(cu_seqlens) - 1

        if use_cache:
            recurrent_state1 = recurrent_state[:S] if recurrent_state is not None else \
                torch.zeros(S, self.num_heads, self.head_dim, self.head_dim).to(torch.float32).to(v.device)
            recurrent_state2 = recurrent_state[S:] if recurrent_state is not None else \
                torch.zeros(S*self.num_sparse_partition, self.num_heads, self.head_dim, self.head_dim).to(torch.float32).to(v.device)

        q2, k2, v2, gk2, _, eta, mask, offsets, state_sizes, global_sorted = sort_along_l(q2, k2, v2, gk2, None, eta, cu_seqlens, self.num_writer, self.emulq, self.emulk)

        aux_loss = torch.zeros(()).to(eta)
        if self.training:
            p = torch.mean(eta.float(), dim=(0, 1))
            f = torch.mean(mask.float(), dim=(0, 1))
            aux_loss = torch.sum(p * f) * self.num_sparse_partition / self.num_writer
        # print(f"layer {self.layer_idx}, aux_loss {aux_loss}")

        q, k, gk, v = [torch.cat(pair, dim=1) for pair in zip((q1, k1, gk1, v1), (q2, k2, gk2, v2))]
        offsets = torch.cat([cu_seqlens.to(offsets), offsets[1:] + cu_seqlens[-1]])
        
        recurrent_state_rec = None
        if use_cache:
            state_id = torch.nonzero(state_sizes.flatten(), as_tuple=True)[0].cpu()
            recurrent_state_rec = torch.cat((recurrent_state1, recurrent_state2[state_id]), dim=0)

        if mode == 'fused_recurrent':
            o, recurrent_state_rec = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=gk,
                initial_state=recurrent_state_rec,
                output_final_state=use_cache,
                cu_seqlens=offsets,
            )
        elif mode == 'chunk':
            o, recurrent_state_rec = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state_rec,
                output_final_state=use_cache,
                cu_seqlens=offsets,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if recurrent_state_rec is not None:
            recurrent_state1 = recurrent_state_rec[:S]
            recurrent_state2[state_id] = recurrent_state_rec[S:]
            recurrent_state = torch.cat((recurrent_state1, recurrent_state2), dim=0)
        else:
            recurrent_state = None

        o1, o2 = o[:, :cu_seqlens[-1]], o[:, cu_seqlens[-1]:]
        o2_reduce = torch.zeros_like(o1)
        o2_reduce.index_add_(dim=1, index=global_sorted, source=o2)
        o = o1 + o2_reduce
        if bsz > 1:
            o = rearrange(o, "1 (b l) h d -> b l h d", b=bsz).contiguous()

        return o, recurrent_state, aux_loss
    
    def sse_linear_attention_mask(self, q1, q2, k1, k2, v, gk1, gk2, eta, recurrent_state=None, use_cache=False, cu_seqlens=None):
        """
        q1: [bsz, qlen, nhead, head_dim]
        q2: [bsz, qlen, nhead, head_dim]
        k1: [bsz, klen, nhead, head_dim]
        k2: [bsz, klen, nhead, head_dim]
        v: [bsz, klen, nhead, head_dim]
        gk1: [bsz, klen, nhead, head_dim]
        gk2: [bsz, klen, nhead, head_dim]
        eta: [bsz, klen, num_sparse_partition]
        """
        bsz, q_len, nhead, _ = q1.shape
        # change to inference mode.
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        # if self.training:
        #     assert mode == 'chunk', "Only chunk mode is supported in training."

        q2, k2, v2, gk2, eta, mask_w, mask_r = softmax_and_mask(q2, k2, v, gk2, eta, self.num_writer, self.num_reader)

        # writer-only auxloss
        aux_loss = torch.zeros(()).to(eta)
        if self.training:
            p = torch.mean(eta.float(), dim=(0, 1))
            f = torch.mean(mask_w.float(), dim=(0, 1))
            aux_loss = torch.sum(p * f) * self.num_sparse_partition / self.num_writer
        # print(f"layer {self.layer_idx}, aux_loss {aux_loss}")
        
        q, k, gk, v = [torch.cat(pair, dim=-2) for pair in zip((q1, k1, gk1, v), (q2, k2, gk2, v2))]

        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        o = rearrange(o, "b l (n h) d -> b l n h d", n=self.num_sparse_partition+1)
        o = o.sum(2)

        return o, recurrent_state, aux_loss

    def swa_softmax_attention(self, q, k, v, attention_mask, cu_seqlens, max_seqlen, batch_size, q_len):
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
        return o

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        hidden_states_ori = hidden_states

        # ==== Linear Attention Starts Here ====
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens')
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        
        q1 = self.sse_q_proj(hidden_states)
        k1 = self.sse_k_proj(hidden_states)
        q2 = q1 + self.lora_q_proj(hidden_states)
        k2 = k1 + self.lora_k_proj(hidden_states)
        v = self.sse_v_proj(hidden_states)

        gk1 = self.sse_gk_proj[0](hidden_states)
        gk2 = self.sse_gk_proj[1](hidden_states)
        
        eta = self.sse_e_proj(hidden_states)

        if self.use_short_conv:
            conv_state_q, conv_state_k = None, None
            if last_state is not None:
                conv_state_q, conv_state_k = last_state['conv_state']
            q1, conv_state_q = self.q_conv1d_shared(
                x=q1,
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k1, conv_state_k = self.k_conv1d_shared(
                x=k1,
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )

        q1, q2, k1, k2, gk1, gk2 = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.sse_head_k_dim), (q1, q2, k1, k2, gk1, gk2))
        v = rearrange(v, '... (h d) -> ... h d', d=self.sse_head_v_dim)

        if self.use_q_softmax:
            q1 = F.softmax(q1.float(), dim=-1).to(v)
            q2 = F.softmax(q2.float(), dim=-1).to(v)
        else:
            q1 = F.relu(q1) if self.sse_qk_relu else F.silu(q1)
            q2 = F.relu(q2) if self.sse_qk_relu else F.silu(q2)
        if self.use_k_softmax:
            k1 = F.softmax(k1.float(), dim=-1).to(v)
            k2 = F.softmax(k2.float(), dim=-1).to(v)
        else:
            k1 = F.relu(k1) if self.sse_qk_relu else F.silu(k1)
            k2 = F.relu(k2) if self.sse_qk_relu else F.silu(k2)
        v = F.silu(v)

        gk1 = F.logsigmoid(gk1) / self.gate_logit_normalizer
        gk2 = F.logsigmoid(gk2) / self.gate_logit_normalizer

        if self.sse_num_kv_heads > self.num_heads:
            q1, q2, k1, k2, gk1, gk2 = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.sse_num_kv_heads // self.num_heads), (q1, q2, k1, k2, gk1, gk2))

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        sse_o, recurrent_state, aux_loss = self.sse_implementation(
            q1,
            q2,
            k1,
            k2,
            v,
            gk1,
            gk2,
            eta,
            recurrent_state=recurrent_state,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
        )

        if self.use_output_gate:
            g = rearrange(self.sse_g_proj(hidden_states), '... (h d) -> ... h d', d=self.sse_head_v_dim)
            sse_o = self.sse_o_norm(sse_o, g)
        else:
            sse_o = self.sse_o_norm(sse_o)
        sse_o = rearrange(sse_o, 'b t h d -> b t (h d)')
        sse_o = self.sse_o_proj(sse_o)
        if attention_mask is not None:
            sse_o = pad_input(sse_o.squeeze(0), indices, batch_size, q_len)

        # ==== Window Attention Starts Here ====
        if self.training and torch.rand(()) > 1 - self.swa_dropout:
            o = self.sse_merge_norm(sse_o)
            return o, (None, aux_loss), past_key_values

        q = rearrange(self.swa_q_proj(hidden_states_ori), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.swa_k_proj(hidden_states_ori), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.swa_v_proj(hidden_states_ori), '... (h d) -> ... h d', d=self.head_dim)

        if self.swa_qk_norm:
            q, k = self.swa_q_norm(q), self.swa_k_norm(k)

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
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=kwargs.get('cu_seqlens'))

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        swa_o = self.swa_softmax_attention(
            q,
            k,
            v,
            attention_mask,
            cu_seqlens,
            max_seqlen,
            batch_size,
            q_len
        )
        swa_o = swa_o.reshape(batch_size, q_len, -1)
        swa_o = self.swa_o_proj(swa_o)

        o = (self.sse_merge_norm(sse_o) + self.swa_merge_norm(swa_o)) / 2

        return o, (None, aux_loss), past_key_values


class SSEGDNH(nn.Module):
    """
    The layer implementaion for [SSE: Scaling Linear Attention with Sparse State Expansion](https://arxiv.org/pdf/2507.16577).

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 2.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA is applied if `num_v_heads` > `num_heads`. Default: `None`.
        mode (str, Optional):
            Which Gated DeltaNet kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_output_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `False`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference: [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        num_sparse_partition (int, optional):
            Number of state partitions. Default: 4.
        num_writer (int, optional):
            Top-k write size (number of writers). Default: 1.
        num_reader (int, optional):
            Top-k read size (number of readers). Default: 1.
        sse_implementation (str, optional):
            SSE implementation to use. One of `"varlen"` or `"mask"`. Default: `"varlen"`.
        use_q_softmax (bool, optional):
            Whether to apply softmax to the query. Default: `False`.
        use_k_softmax (bool, optional):
            Whether to apply softmax to the key. Default: `True`.
        emulq (bool, optional):
            Whether to use a read gate operating on the state output (Q). Default: `True`.
        emulk (bool, optional):
            Whether to use a write gate operating on the state input (KV). Default: `True`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 256,
        num_heads: int = 6,
        mode: str = 'chunk',
        use_output_gate: bool = True,
        use_short_conv: bool = False,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        num_sparse_partition: int = 4,
        num_writer: int = 1,
        num_reader: int = 1,
        sse_implementation: str = "varlen",
        sse_qk_relu: bool = False,
        use_q_softmax: bool = False,
        use_k_softmax: bool = False,
        emulq: bool = True,
        emulk: bool = True,
        qkv_bias: bool = False,
        # === swa configs ===
        swa_num_kv_heads: int | None = None,
        swa_qk_norm: bool = False,
        swa_dropout: float = 0.5,
        window_size: int | None = None,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int | None = None,
        # ===================
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> SSEGDNH:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.layer_idx = layer_idx

        assert num_reader < num_sparse_partition and num_writer < num_sparse_partition, \
            "num_reader and num_writer must be less than num_sparse_partition."
        assert sse_implementation in ["mask", "varlen"], \
            f"Unknown SSE implementation {sse_implementation}"

        self.num_sparse_partition = num_sparse_partition
        self.num_writer = num_writer
        self.num_reader = num_reader
        self.sse_implementation = {
            "mask": self.sse_linear_attention_mask,
            "varlen": self.sse_linear_attention_varlen,
        }[sse_implementation]

        self.use_output_gate = use_output_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.sse_qk_relu = sse_qk_relu
        self.use_q_softmax = use_q_softmax
        self.use_k_softmax = use_k_softmax
        self.emulq = emulq
        self.emulk = emulk

        # mha for sse, gqa for swa
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.sse_num_kv_heads = num_heads
        if swa_num_kv_heads is None:
            self.swa_num_kv_heads = self.num_heads
        else:
            self.swa_num_kv_heads = swa_num_kv_heads
        self.swa_num_kv_groups = num_heads // self.swa_num_kv_heads

        self.sse_head_k_dim = head_dim
        self.sse_head_v_dim = int(self.head_dim * self.expand_v)
        self.sse_key_dim = int(self.sse_num_kv_heads * self.sse_head_k_dim)
        self.sse_value_dim = int(self.sse_num_kv_heads * self.sse_head_v_dim)

        self.swa_q_dim = self.num_heads * self.head_dim
        self.swa_kv_dim = int(self.swa_num_kv_heads * self.head_dim)
        self.swa_qk_norm = swa_qk_norm
        self.qkv_bias = qkv_bias

        self.swa_dropout = swa_dropout
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        
        # Consistency check: Ensure expand_v produces integer values
        assert self.expand_v == 1.0
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        self.sse_q_proj = nn.Linear(hidden_size, self.sse_key_dim, bias=False)
        self.sse_k_proj = nn.Linear(hidden_size, self.sse_key_dim, bias=False)
        self.sse_v_proj = nn.Linear(hidden_size, self.sse_value_dim, bias=False)
        self.lora_q_proj = nn.Sequential(nn.Linear(hidden_size, self.sse_head_v_dim, bias=False),
                                         nn.Linear(self.sse_head_v_dim, self.sse_key_dim, bias=False))
        self.lora_k_proj = nn.Sequential(nn.Linear(hidden_size, self.sse_head_v_dim, bias=False),
                                         nn.Linear(self.sse_head_v_dim, self.sse_key_dim, bias=False))
        
        self.sse_a_proj = nn.Linear(hidden_size, self.sse_num_kv_heads*2, bias=False)
        self.sse_b_proj = nn.Linear(hidden_size, self.sse_num_kv_heads*2, bias=False)

        A = torch.empty(self.sse_num_kv_heads*2, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.sse_num_kv_heads*2) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        self.sse_e_proj = nn.Linear(hidden_size, self.num_sparse_partition, bias=False)

        self.swa_q_proj = nn.Linear(hidden_size, self.swa_q_dim, bias=self.qkv_bias)
        self.swa_k_proj = nn.Linear(hidden_size, self.swa_kv_dim, bias=self.qkv_bias)
        self.swa_v_proj = nn.Linear(hidden_size, self.swa_kv_dim, bias=self.qkv_bias)

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d_shared = ShortConvolution(
                hidden_size=self.sse_key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )
            self.k_conv1d_shared = ShortConvolution(
                hidden_size=self.sse_key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )

        if use_output_gate:
            self.sse_g_proj = nn.Sequential(nn.Linear(hidden_size, self.sse_head_v_dim, bias=False),
                                        nn.Linear(self.sse_head_v_dim, self.sse_value_dim, bias=False))
            self.sse_o_norm = FusedRMSNormGated(self.sse_head_v_dim, eps=norm_eps)
        else:
            self.sse_o_norm = RMSNorm(self.sse_head_v_dim, eps=norm_eps)

        self.sse_o_proj = nn.Linear(self.sse_value_dim, self.hidden_size, bias=False)
        self.swa_o_proj = nn.Linear(self.swa_q_dim, self.hidden_size, bias=False)

        self.sse_merge_norm = RMSNorm(self.hidden_size, eps=norm_eps)
        self.swa_merge_norm = RMSNorm(self.hidden_size, eps=norm_eps)

        if swa_qk_norm:
            self.swa_q_norm = RMSNorm(self.head_dim, eps=norm_eps)
            self.swa_k_norm = RMSNorm(self.head_dim, eps=norm_eps)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)
    
    def sse_linear_attention_varlen(self, q1, q2, k1, k2, v, g1, g2, b1, b2, eta, recurrent_state=None, use_cache=False, cu_seqlens=None):
        """
        q1: [bsz, qlen, nhead, head_dim]
        q2: [bsz, qlen, nhead, head_dim]
        k1: [bsz, klen, nhead, head_dim]
        k2: [bsz, klen, nhead, head_dim]
        v: [bsz, klen, nhead, head_dim]
        g1: [bsz, klen, nhead]
        g2: [bsz, klen, nhead]
        b1: [bsz, klen, nhead]
        b2: [bsz, klen, nhead]
        eta: [bsz, klen, num_sparse_partition]
        """
        assert self.num_writer == self.num_reader, "varlen only support num_writer == num_reader"
        bsz, q_len, nhead, _ = q1.shape
        # change to inference mode.
        mode = 'fused_recurrent' if q_len // self.num_sparse_partition <= 64 else self.mode
        # if self.training:
        #     assert mode == 'chunk', "Only chunk mode is supported in training."

        v1 = v
        v2 = v
        if cu_seqlens is None:
            cu_seqlens = torch.arange(0, (bsz + 1) * q_len, q_len, dtype=torch.int32, device=q1.device)
            q1, k1, v1 = [rearrange(src, 'b l h d -> 1 (b l) h d').contiguous() for src in [q1, k1, v]]
            q2, k2, v2 = [rearrange(src, 'b l h d -> 1 (b l) h d').contiguous() for src in [q2, k2, v]]
            g1, g2, b1, b2 = [rearrange(src, 'b l h -> 1 (b l) h').contiguous() for src in [g1, g2, b1, b2]]
        S = len(cu_seqlens) - 1

        if use_cache:
            recurrent_state1 = recurrent_state[:S] if recurrent_state is not None else \
                torch.zeros(S, self.num_heads, self.head_dim, self.head_dim).to(torch.float32).to(v.device)
            recurrent_state2 = recurrent_state[S:] if recurrent_state is not None else \
                torch.zeros(S*self.num_sparse_partition, self.num_heads, self.head_dim, self.head_dim).to(torch.float32).to(v.device)

        q2, k2, v2, g2, b2, eta, mask, offsets, state_sizes, global_sorted = sort_along_l(q2, k2, v2, g2, b2, eta, cu_seqlens, self.num_writer, self.emulq, self.emulk)

        aux_loss = torch.zeros(()).to(eta)
        if self.training:
            p = torch.mean(eta.float(), dim=(0, 1))
            f = torch.mean(mask.float(), dim=(0, 1))
            aux_loss = torch.sum(p * f) * self.num_sparse_partition / self.num_writer
        # print(f"layer {self.layer_idx}, aux_loss {aux_loss}")

        q, k, g, b, v = [torch.cat(pair, dim=1) for pair in zip((q1, k1, g1, b1, v1), (q2, k2, g2, b2, v2))]
        offsets = torch.cat([cu_seqlens.to(offsets), offsets[1:] + cu_seqlens[-1]])
        
        recurrent_state_rec = None
        if use_cache:
            state_id = torch.nonzero(state_sizes.flatten(), as_tuple=True)[0].cpu()
            recurrent_state_rec = torch.cat((recurrent_state1, recurrent_state2[state_id]), dim=0)

        if mode == 'fused_recurrent':
            o, recurrent_state_rec = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=b,
                initial_state=recurrent_state_rec,
                output_final_state=use_cache,
                cu_seqlens=offsets,
                use_qk_l2norm_in_kernel=True,
            )
        elif mode == 'chunk':
            o, recurrent_state_rec = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=b,
                initial_state=recurrent_state_rec,
                output_final_state=use_cache,
                cu_seqlens=offsets,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if recurrent_state_rec is not None:
            recurrent_state1 = recurrent_state_rec[:S]
            recurrent_state2[state_id] = recurrent_state_rec[S:]
            recurrent_state = torch.cat((recurrent_state1, recurrent_state2), dim=0)
        else:
            recurrent_state = None

        o1, o2 = o[:, :cu_seqlens[-1]], o[:, cu_seqlens[-1]:]
        o2_reduce = torch.zeros_like(o1)
        o2_reduce.index_add_(dim=1, index=global_sorted, source=o2)
        o = o1 + o2_reduce
        if bsz > 1:
            o = rearrange(o, "1 (b l) h d -> b l h d", b=bsz).contiguous()

        return o, recurrent_state, aux_loss
    
    def sse_linear_attention_mask(self, q1, q2, k1, k2, v, g1, g2, b1, b2, eta, recurrent_state=None, use_cache=False, cu_seqlens=None):
        """
        q1: [bsz, qlen, nhead, head_dim]
        q2: [bsz, qlen, nhead, head_dim]
        k1: [bsz, klen, nhead, head_dim]
        k2: [bsz, klen, nhead, head_dim]
        v: [bsz, klen, nhead, head_dim]
        g1: [bsz, klen, nhead]
        g2: [bsz, klen, nhead]
        b1: [bsz, klen, nhead]
        b2: [bsz, klen, nhead]
        eta: [bsz, klen, num_sparse_partition]
        """
        bsz, q_len, nhead, _ = q1.shape
        # change to inference mode.
        mode = 'fused_recurrent' if q_len // self.num_sparse_partition <= 64 else self.mode
        # if self.training:
        #     assert mode == 'chunk', "Only chunk mode is supported in training."

        q2, k2, v2, _, eta, mask_w, mask_r = softmax_and_mask(q2, k2, v, v, eta, self.num_writer, self.num_reader)
        g2, b2 = [repeat(x, "b l h -> b l n h", n=self.num_sparse_partition) for x in (g2, b2)]
        mask_r = mask_r[..., None]
        g2, b2 = g2 * mask_r, b2 * mask_r
        g2, b2 = [rearrange(x, "b l n h -> b l (n h)") for x in (g2, b2)]

        # writer-only auxloss
        aux_loss = torch.zeros(()).to(eta)
        if self.training:
            p = torch.mean(eta.float(), dim=(0, 1))
            f = torch.mean(mask_w.float(), dim=(0, 1))
            aux_loss = torch.sum(p * f) * self.num_sparse_partition / self.num_writer
        # print(f"layer {self.layer_idx}, aux_loss {aux_loss}")
        
        q, k, g, b, v = [torch.cat(pair, dim=2) for pair in zip((q1, k1, g1, b1, v), (q2, k2, g2, b2, v2))]

        if mode == 'chunk':
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=b,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=b,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        o = rearrange(o, "b l (n h) d -> b l n h d", n=self.num_sparse_partition+1)
        o = o.sum(2)

        return o, recurrent_state, aux_loss

    def swa_softmax_attention(self, q, k, v, attention_mask, cu_seqlens, max_seqlen, batch_size, q_len):
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
        return o

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        hidden_states_ori = hidden_states

        # ==== Linear Attention Starts Here ====
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens')
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        q1 = self.sse_q_proj(hidden_states)
        k1 = self.sse_k_proj(hidden_states)
        q2 = q1 + self.lora_q_proj(hidden_states)
        k2 = k1 + self.lora_k_proj(hidden_states)
        v = self.sse_v_proj(hidden_states)

        b = self.sse_b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            b = b * 2.
        g = -self.A_log.float().exp() * F.softplus(self.sse_a_proj(hidden_states).float() + self.dt_bias)
        b1, b2 = torch.chunk(b, 2, dim=-1)
        g1, g2 = torch.chunk(g, 2, dim=-1)
        
        eta = self.sse_e_proj(hidden_states)

        if self.use_short_conv:
            conv_state_q, conv_state_k = None, None
            if last_state is not None:
                conv_state_q, conv_state_k = last_state['conv_state']
            q1, conv_state_q = self.q_conv1d_shared(
                x=q1,
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k1, conv_state_k = self.k_conv1d_shared(
                x=k1,
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )

        q1, q2, k1, k2 = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.sse_head_k_dim), (q1, q2, k1, k2))
        v = rearrange(v, '... (h d) -> ... h d', d=self.sse_head_v_dim)

        if self.use_q_softmax:
            q1 = F.softmax(q1.float(), dim=-1).to(v)
            q2 = F.softmax(q2.float(), dim=-1).to(v)
        else:
            q1 = F.relu(q1) if self.sse_qk_relu else F.silu(q1)
            q2 = F.relu(q2) if self.sse_qk_relu else F.silu(q2)
        if self.use_k_softmax:
            k1 = F.softmax(k1.float(), dim=-1).to(v)
            k2 = F.softmax(k2.float(), dim=-1).to(v)
        else:
            k1 = F.relu(k1) if self.sse_qk_relu else F.silu(k1)
            k2 = F.relu(k2) if self.sse_qk_relu else F.silu(k2)
        v = F.silu(v)

        if self.sse_num_kv_heads > self.num_heads:
            q1, q2, k1, k2 = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.sse_num_kv_heads // self.num_heads), (q1, q2, k1, k2))

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        sse_o, recurrent_state, aux_loss = self.sse_implementation(
            q1,
            q2,
            k1,
            k2,
            v,
            g1,
            g2,
            b1,
            b2,
            eta,
            recurrent_state=recurrent_state,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
        )

        if self.use_output_gate:
            g = rearrange(self.sse_g_proj(hidden_states), '... (h d) -> ... h d', d=self.sse_head_v_dim)
            sse_o = self.sse_o_norm(sse_o, g)
        else:
            sse_o = self.sse_o_norm(sse_o)
        sse_o = rearrange(sse_o, 'b t h d -> b t (h d)')
        sse_o = self.sse_o_proj(sse_o)
        if attention_mask is not None:
            sse_o = pad_input(sse_o.squeeze(0), indices, batch_size, q_len)

        # ==== Window Attention Starts Here ====
        if self.training and torch.rand(()) > 1 - self.swa_dropout:
            o = self.sse_merge_norm(sse_o)
            return o, (None, aux_loss), past_key_values

        q = rearrange(self.swa_q_proj(hidden_states_ori), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.swa_k_proj(hidden_states_ori), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.swa_v_proj(hidden_states_ori), '... (h d) -> ... h d', d=self.head_dim)

        if self.swa_qk_norm:
            q, k = self.swa_q_norm(q), self.swa_k_norm(k)

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
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=kwargs.get('cu_seqlens'))

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        swa_o = self.swa_softmax_attention(
            q,
            k,
            v,
            attention_mask,
            cu_seqlens,
            max_seqlen,
            batch_size,
            q_len
        )
        swa_o = swa_o.reshape(batch_size, q_len, -1)
        swa_o = self.swa_o_proj(swa_o)

        o = (self.sse_merge_norm(sse_o) + self.swa_merge_norm(swa_o)) / 2

        return o, (None, aux_loss), past_key_values
