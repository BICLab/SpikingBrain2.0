"""A clean version of efficient moba implementation with flash-attn"""

import torch

from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from functools import lru_cache
from einops import rearrange

from flash_moba import flash_moba_varlen_func


@lru_cache(maxsize=16)
def calc_chunks(cu_seqlen, moba_chunk_size):
    """calc chunks that needs moba attention"""

    # batch_sizes[batch_idx] = batch size ( seqlen ) of batch idx
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    # batch_num_chunk[batch_idx] = how many chunk in batch idx
    batch_num_chunk = (batch_sizes + (moba_chunk_size - 1)) // moba_chunk_size
    # cu_num_chunk[batch_idx] = first chunk id of this batch
    cu_num_chunk = torch.ones(
        batch_num_chunk.numel() + 1,
        device=cu_seqlen.device,
        dtype=batch_num_chunk.dtype,
    )
    cu_num_chunk[1:] = batch_num_chunk.cumsum(dim=0)
    # total chunk ( for all batch )
    num_chunk = cu_num_chunk[-1]
    # chunk_sizes[chunk_idx] = chunk_size of chunk idx
    chunk_sizes = torch.full(
        (num_chunk + 1,), moba_chunk_size, dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_sizes[0] = 0  # for calc cu chunk
    batch_last_chunk_size = batch_sizes - (batch_num_chunk - 1) * moba_chunk_size
    chunk_sizes[cu_num_chunk[1:]] = batch_last_chunk_size
    # cu_chunk[chunk_idx] = the start chunk offset of chunk idx
    cu_chunk = chunk_sizes.cumsum(dim=-1, dtype=torch.int32)
    # chunk_to_batch[chunk_idx] = batch idx of the chunk idx
    # from chunk idx -> batch idx
    chunk_to_batch = torch.zeros(
        (num_chunk,), dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_to_batch[cu_num_chunk[1:-1]] = 1 # cumsum 出来呈阶梯状，在每个batch的第一个chunk位置上加1，后续cumsum后就得到了chunk idx -> batch idx的映射
    chunk_to_batch = chunk_to_batch.cumsum(dim=0, dtype=torch.int32)

    """ filter chunks that need moba attn """

    # filter chunks ( remove last chunk of each batch )
    # filtered_chunk_indices: chunk index list that excludes the last chunk of each batch
    chunk_to_remove = cu_num_chunk[1:] - 1 # cumsum chunk idx - 1 is the last chunk idx of each batch
    chunk_to_remain = torch.ones(
        (num_chunk,), dtype=torch.bool, device=cu_seqlen.device
    )
    chunk_to_remain[chunk_to_remove] = False
    filtered_chunk_indices = chunk_to_remain.nonzero(as_tuple=True)[0]
    num_filtered_chunk = len(filtered_chunk_indices)

    return (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    )


class MixedAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
    ):
        ctx.max_seqlen = max_seqlen
        ctx.moba_chunk_size = moba_chunk_size
        ctx.softmax_scale = softmax_scale = q.shape[-1] ** (-0.5)

        # self attn
        self_attn_out_sh, self_attn_lse_hs, _, _ = (
            _flash_attn_varlen_forward(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self_attn_cu_seqlen,
                cu_seqlens_k=self_attn_cu_seqlen,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                causal=True,
                dropout_p=0.0,
            )
        )

        # moba attn
        moba_attn_out, moba_attn_lse_hs, _, _ = _flash_attn_varlen_forward(
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
        )

        # convert lse shape hs -> sh ( follow the legacy mix attn logic )
        self_attn_lse_sh = self_attn_lse_hs.t().contiguous()
        moba_attn_lse = moba_attn_lse_hs.t().contiguous()

        # output buffer [S, H, D], same shape as q
        output = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        # flatten vS & H for index ops
        output_2d = output.view(-1, q.shape[2])

        # calc mixed_lse
        # minus max lse to avoid exp explosion
        max_lse_1d = self_attn_lse_sh.view(-1)
        max_lse_1d = max_lse_1d.index_reduce(
            0, moba_q_sh_indices, moba_attn_lse.view(-1), "amax"
        )
        self_attn_lse_sh = self_attn_lse_sh - max_lse_1d.view_as(self_attn_lse_sh)
        moba_attn_lse = (
            moba_attn_lse.view(-1)
            .sub(max_lse_1d.index_select(0, moba_q_sh_indices))
            .reshape_as(moba_attn_lse)
        )

        mixed_attn_se_sh = self_attn_lse_sh.exp()
        moba_attn_se = moba_attn_lse.exp()

        mixed_attn_se_sh.view(-1).index_add_(
            0, moba_q_sh_indices, moba_attn_se.view(-1)
        )
        mixed_attn_lse_sh = mixed_attn_se_sh.log()

        # add attn output
        factor = (self_attn_lse_sh - mixed_attn_lse_sh).exp()  # [ vS, H ]
        self_attn_out_sh = self_attn_out_sh * factor.unsqueeze(-1)
        output_2d += self_attn_out_sh.reshape_as(output_2d)

        # add moba output
        mixed_attn_lse = (
            mixed_attn_lse_sh.view(-1)
            .index_select(0, moba_q_sh_indices)
            .view_as(moba_attn_lse)
        )
        factor = (moba_attn_lse - mixed_attn_lse).exp()  # [ vS, H ]
        moba_attn_out = moba_attn_out * factor.unsqueeze(-1)
        raw_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[-1])
        output_2d.index_add_(0, moba_q_sh_indices, raw_attn_out)
        output = output.to(q.dtype)
        # add back max lse
        # mixed_attn_lse_sh = mixed_attn_lse_sh + max_lse_1d.view_as(mixed_attn_se_sh)
        # ctx.save_for_backward(
        #     output,
        #     mixed_attn_lse_sh,
        #     q,
        #     k,
        #     v,
        #     self_attn_cu_seqlen,
        #     moba_q,
        #     moba_kv,
        #     moba_cu_seqlen_q,
        #     moba_cu_seqlen_kv,
        #     moba_q_sh_indices,
        # )

        return output

    # @staticmethod
    # def backward(ctx, d_output):

    #     max_seqlen = ctx.max_seqlen
    #     moba_chunk_size = ctx.moba_chunk_size
    #     softmax_scale = ctx.softmax_scale

    #     (
    #         output,
    #         mixed_attn_vlse_sh,
    #         q,
    #         k,
    #         v,
    #         self_attn_cu_seqlen,
    #         moba_q,
    #         moba_kv,
    #         moba_cu_seqlen_q,
    #         moba_cu_seqlen_kv,
    #         moba_q_sh_indices,
    #     ) = ctx.saved_tensors

    #     d_output = d_output.contiguous()

    #     dq, dk, dv, _ = _flash_attn_varlen_backward(
    #         dout=d_output,
    #         q=q,
    #         k=k,
    #         v=v,
    #         out=output,
    #         softmax_lse=mixed_attn_vlse_sh.t().contiguous(),
    #         dq=None,
    #         dk=None,
    #         dv=None,
    #         cu_seqlens_q=self_attn_cu_seqlen,
    #         cu_seqlens_k=self_attn_cu_seqlen,
    #         max_seqlen_q=max_seqlen,
    #         max_seqlen_k=max_seqlen,
    #         softmax_scale=softmax_scale,
    #         causal=True,
    #         dropout_p=0.0,
    #         window_size=(-1, -1),
    #         softcap=0.0,
    #         alibi_slopes=None,
    #         deterministic=True,
    #     )

    #     headdim = q.shape[-1]
    #     d_moba_output = (
    #         d_output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
    #     )
    #     moba_output = (
    #         output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
    #     )

    #     mixed_attn_vlse = (
    #         mixed_attn_vlse_sh.view(-1).index_select(0, moba_q_sh_indices).view(1, -1)
    #     )

    #     dmq, dmk, dmv, _ = _flash_attn_varlen_backward(
    #         dout=d_moba_output,
    #         q=moba_q,
    #         k=moba_kv[:, 0],
    #         v=moba_kv[:, 1],
    #         out=moba_output,
    #         softmax_lse=mixed_attn_vlse,
    #         dq=None,
    #         dk=None,
    #         dv=None,
    #         cu_seqlens_q=moba_cu_seqlen_q,
    #         cu_seqlens_k=moba_cu_seqlen_kv,
    #         max_seqlen_q=max_seqlen,
    #         max_seqlen_k=moba_chunk_size,
    #         softmax_scale=softmax_scale,
    #         causal=False,
    #         dropout_p=0.0,
    #         window_size=(-1, -1),
    #         softcap=0.0,
    #         alibi_slopes=None,
    #         deterministic=True,
    #     )

    #     dmkv = torch.stack((dmk, dmv), dim=1)
    #     return dq, dk, dv, None, dmq, dmkv, None, None, None, None, None


def moba_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
) -> torch.Tensor:
    """An efficient version of moba implementation with triton kernels and flash-attn, the core logic:
    1. Calculate the chunks and the number of chunks, n = floor(data_size / chunk_size)
       - tokens in the tail chunk are reserved for self attn
       - tokens in other chunks will be processed in later steps
    2. K in each chunk will calculate mean value as the representative k, and Q will attend to these representative
    k to get the gate logit, which will be used to select topk chunks
    3. Select the topk chunks and get the dense q for each kv chunk pair and do the varlen attention
    4. Combine the varlen attn and self attn results via online softmax to get the final result

    Args:
        q (torch.Tensor): [seqlen, head, head_dim]
        k (torch.Tensor): [seqlen, head, head_dim]
        v (torch.Tensor): [seqlen, head, head_dim]
        cu_seqlens (torch.Tensor): the cumulative sequence length tensor, same definition in flash attn
        max_seqlen (int): the max sequence length of the batch, same definition in flash attn

    Returns:
        attn_output (torch.Tensor): [seqlen, head, head_dim]
    """

    kv = torch.stack((k, v), dim=1)

    """ some basic variables """
    # qkv shape = [ S, H, D ]
    seqlen, num_head, head_dim = q.shape

    """ prepare chunk meta """
    (
        cu_chunk, # cumulative chunk size
        filtered_chunk_indices, # the chunk indices that are selected for moba attn, excluding the last chunk of each batch
        num_filtered_chunk,
        chunk_to_batch, # from chunk idx -> batch idx
    ) = calc_chunks(cu_seqlens, moba_chunk_size)

    # we will adjust selective topk to moba_topk - 1, as the last chunk is always chosen
    moba_topk = min(moba_topk - 1, num_filtered_chunk)
    need_moba_attn = moba_topk > 0
    
    # corner case: if no moba attn needed, just return self attn
    if not need_moba_attn:
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
        )

    self_attn_cu_seqlen = cu_chunk

    # filtered_kv is a dense matrix that only contains filtered chunk of kv
    filtered_kv_indices = torch.arange(
        0, moba_chunk_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_filtered_chunk, 1)
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None]
    filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1))

    """ calc key_gate_weight and gate """

    # key_gate_weight [ F_N_CHUNK, HEAD, HEAD_DIM ]
    key_gate_weight = (
        filtered_kv[:, 0]
        .view(num_filtered_chunk, moba_chunk_size, num_head, head_dim)
        .mean(dim=1)
        .float()
    )
    q = q.type(torch.float32)  # float logit on the fly for better gate logit perception
    key_gate_weight = key_gate_weight.type(
        torch.float32
    )  # float logit for better gate logit perception
    gate = torch.einsum(
        "nhd,shd->nhs", key_gate_weight, q
    )  # gate [ F_N_CHUNK, HEAD, SEQ ]
    key_gate_weight = key_gate_weight.type_as(k)
    q = q.type_as(k)

    # pose process gate, masking unchosen batch and apply causal mask to current chunk
    gate_seq_idx = torch.arange(0, seqlen, device=q.device, dtype=torch.int32)[
        None, :
    ].repeat(num_filtered_chunk, 1)
    chunk_end = cu_chunk[filtered_chunk_indices + 1]
    batch_end = cu_seqlens[chunk_to_batch[filtered_chunk_indices] + 1]
    gate_chunk_end_mask = gate_seq_idx < chunk_end[:, None]
    gate_batch_end_mask = gate_seq_idx >= batch_end[:, None]
    gate_inf_mask = gate_chunk_end_mask | gate_batch_end_mask
    gate.masked_fill_(gate_inf_mask.unsqueeze(1), -float("inf"))

    """ find moba q that needs moba attn """
    # find topk chunks
    # gate_mask [ N_CHUNK, HEAD, SEQ ], true indicates that needs attention
    _, gate_top_k_idx = torch.topk(gate, k=moba_topk, dim=0, largest=True, sorted=False)
    # apply causal mask
    gate_mask = torch.logical_not(gate.isinf())
    # select topk chunks
    gate_idx_mask = torch.zeros(gate_mask.shape, dtype=torch.bool, device=q.device)
    gate_idx_mask = gate_idx_mask.scatter_(dim=0, index=gate_top_k_idx, value=True)
    gate_mask = torch.logical_and(gate_mask, gate_idx_mask)

    # varlen trick: combining all q index that needs moba attn
    # the result will be like [ C0H0 ][ C0H1 ][ C0H2 ][ ... ][ CnHm ]
    moba_q_indices = gate_mask.reshape(gate_mask.shape[0], -1).nonzero(as_tuple=True)[
        -1
    ]  # [ HS indices ] * N
    # moba_seqlen_q indicates that how many q chunks are selected for each kv chunk - head
    moba_seqlen_q = gate_mask.sum(dim=-1).flatten()
    # select all q that needs moba attn based on the moba_q_indices
    moba_q = rearrange(q, "s h d -> ( h s ) d").index_select(
        0, moba_q_indices
    )  # [ selected_S, D ]
    moba_q = moba_q.unsqueeze(1)
    # moba_q_sh_indices represents the position in the origin q tensor of each q token inside moba_q
    moba_q_sh_indices = moba_q_indices % seqlen * num_head + moba_q_indices // seqlen

    """ prepare moba kv """
    # Since moba_q is organized as HS * N, we need to reorganize kv to adapt to q

    # cut off zero experts
    q_zero_mask = moba_seqlen_q == 0
    valid_expert_mask = ~q_zero_mask
    zero_expert_count = q_zero_mask.sum()
    # only keep the kv that has q select > 0
    if zero_expert_count > 0:
        moba_seqlen_q = moba_seqlen_q[valid_expert_mask]
    # moba cu_seqlen for flash attn
    moba_cu_seqlen_q = torch.cat(
        (
            torch.tensor([0], device=q.device, dtype=moba_seqlen_q.dtype),
            moba_seqlen_q.cumsum(dim=0),
        ),
        dim=0,
    ).to(torch.int32)
    moba_kv = rearrange(filtered_kv, "s x h d -> h s x d")
    moba_kv = moba_kv.split(moba_chunk_size, dim=1)
    moba_kv = torch.cat(moba_kv, dim=0)
    if zero_expert_count > 0:
        assert valid_expert_mask.sum() == moba_kv.shape[0] - zero_expert_count
        moba_kv = moba_kv[
            valid_expert_mask
        ]  # cut off zero Q expert from kv , or the grad may be nan
    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2)
    moba_cu_seqlen_kv = (
        torch.arange(
            0,
            num_filtered_chunk * num_head + 1 - zero_expert_count,
            dtype=torch.int32,
            device=q.device,
        )
        * moba_chunk_size
    )

    # Shape check
    assert (
        moba_cu_seqlen_kv.shape == moba_cu_seqlen_q.shape
    ), f"moba_cu_seqlen_kv.shape != moba_cu_seqlen_q.shape {moba_cu_seqlen_kv.shape} != {moba_cu_seqlen_q.shape}"

    # Wrapping up the flash attn call and online softmax dlse inside MixedAttention class
    return MixedAttention.apply(
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
    )


def _build_cu_seqlens_from_seq_lens(seq_lens: torch.Tensor) -> torch.Tensor:
    seq_lens_i32 = seq_lens.to(dtype=torch.int32)
    return torch.cat(
        (
            torch.zeros((1,), dtype=torch.int32, device=seq_lens.device),
            seq_lens_i32.cumsum(dim=0, dtype=torch.int32),
        ),
        dim=0,
    )


def _gather_paged_kv_from_cache(
    key_cache: torch.Tensor, # kv_cache: shape =[num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather paged KV cache into dense [total_k, H_kv, D] tensors."""
    device = key_cache.device
    block_size = key_cache.shape[1]
    num_seqs = seq_lens.shape[0]

    slot_indices: list[torch.Tensor] = []
    for seq_idx in range(num_seqs):
        kv_len = int(seq_lens[seq_idx].item())
        if kv_len <= 0:
            continue
        num_blocks = (kv_len + block_size - 1) // block_size
        seq_blocks = block_table[seq_idx, :num_blocks].to(dtype=torch.int64)
        pos = torch.arange(kv_len, device=device, dtype=torch.int64)
        block_idx = pos // block_size
        offset = pos % block_size
        slots = seq_blocks.index_select(0, block_idx) * block_size + offset
        slot_indices.append(slots)

    if len(slot_indices) == 0:
        k_out = torch.empty(
            (0, key_cache.shape[2], key_cache.shape[3]),
            dtype=out_dtype,
            device=device,
        )
        v_out = torch.empty_like(k_out)
        return k_out, v_out

    flat_slots = torch.cat(slot_indices, dim=0)
    flat_k = key_cache.reshape(-1, key_cache.shape[2], key_cache.shape[3])
    flat_v = value_cache.reshape(-1, value_cache.shape[2], value_cache.shape[3])
    k_dense = flat_k.index_select(0, flat_slots)
    v_dense = flat_v.index_select(0, flat_slots)

    # FP8 cache stores quantized values; dequantize with layer scale.
    if kv_cache_dtype.startswith("fp8"):
        k_dense = k_dense.float()
        v_dense = v_dense.float()
        if k_scale is not None:
            k_dense = k_dense * k_scale.float()
        if v_scale is not None:
            v_dense = v_dense * v_scale.float()

    return k_dense.to(dtype=out_dtype), v_dense.to(dtype=out_dtype)


def _pad_suffix_q_to_full_kv(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand suffix-only q to full-kv length by left-padding each sequence."""
    assert cu_seqlens_q.shape == cu_seqlens_k.shape
    num_seqs = cu_seqlens_q.shape[0] - 1

    q_full = torch.zeros(
        (int(cu_seqlens_k[-1].item()), q.shape[1], q.shape[2]),
        dtype=q.dtype,
        device=q.device,
    )
    out_indices: list[torch.Tensor] = []
    for seq_idx in range(num_seqs):
        q_s = int(cu_seqlens_q[seq_idx].item())
        q_e = int(cu_seqlens_q[seq_idx + 1].item())
        k_s = int(cu_seqlens_k[seq_idx].item())
        k_e = int(cu_seqlens_k[seq_idx + 1].item())
        q_len = q_e - q_s
        k_len = k_e - k_s
        assert q_len <= k_len, (
            f"q_len should be <= kv_len for each sequence, got q_len={q_len}, "
            f"kv_len={k_len} at seq_idx={seq_idx}"
        )
        if q_len == 0:
            continue
        start_in_k = k_e - q_len
        q_full[start_in_k:k_e] = q[q_s:q_e]
        out_indices.append(
            torch.arange(start_in_k, k_e, dtype=torch.int64, device=q.device)
        )

    if len(out_indices) == 0:
        return q_full, torch.empty((0,), dtype=torch.int64, device=q.device)
    return q_full, torch.cat(out_indices, dim=0)


def moba_attn_varlen_paged(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    moba_chunk_size: int,
    moba_topk: int,
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """MOBA attention for suffix-only queries over paged KV cache.

    This is designed for chunked prefill / multi-turn conversations where
    q only contains newly scheduled tokens while kv cache contains full history.
    """
    num_seqs = cu_seqlens_q.shape[0] - 1
    seq_lens_k = seqused_k[:num_seqs].to(dtype=torch.int32)
    cu_seqlens_k = _build_cu_seqlens_from_seq_lens(seq_lens_k)
    # to support chunked prefill.
    k_dense, v_dense = _gather_paged_kv_from_cache(
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table[:num_seqs],
        seq_lens=seq_lens_k,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
        out_dtype=q.dtype,
    )
    return flash_moba_varlen_func(
        q, 
        k_dense, 
        v_dense, 
        cu_seqlens_q, 
        cu_seqlens_k, 
        max_seqlen_q=max_seqlen_q, 
        max_seqlen_k=max_seqlen_k, 
        moba_chunk_size=moba_chunk_size,
        moba_topk=moba_topk,
        causal=True
    )
    # Expand KV heads to match Q heads if using GQA/MQA.
    # if q.shape[1] != k_dense.shape[1]:
    #     assert q.shape[1] % k_dense.shape[1] == 0, (
    #         f"head q {q.shape[1]} must be divisible by head kv "
    #         f"{k_dense.shape[1]} for MoBA attention"
    #     )
    #     kv_groups = q.shape[1] // k_dense.shape[1]
    #     k_dense = torch.repeat_interleave(k_dense, kv_groups, dim=1, output_size=q.shape[1])
    #     v_dense = torch.repeat_interleave(v_dense, kv_groups, dim=1, output_size=q.shape[1])

    # # Reuse existing MoBA implementation by padding per-sequence q on the left:
    # # [prefix(zeros), suffix(real q)] so q and kv share the same cu_seqlens.
    # q_full, out_indices = _pad_suffix_q_to_full_kv(
    #     q=q,
    #     cu_seqlens_q=cu_seqlens_q.to(dtype=torch.int32),
    #     cu_seqlens_k=cu_seqlens_k,
    # )

    # out_full = moba_attn_varlen(
    #     q=q_full,
    #     k=k_dense,
    #     v=v_dense,
    #     cu_seqlens=cu_seqlens_k,
    #     max_seqlen=max_seqlen_k,
    #     moba_chunk_size=moba_chunk_size,
    #     moba_topk=moba_topk,
    # )
    # return out_full.index_select(0, out_indices)