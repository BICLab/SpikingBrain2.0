# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

# SSE 可以直接用 GDN backend, 不用融合了，这样一来，
# MoBA backend 也可以直接复用 FlashAttn backend, super().forward
# 当且仅当 is_moba = True 的时候才跑自己的 forward

from dataclasses import dataclass

import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
)
from vllm.attention.utils.fa_utils import (
    is_flash_attn_varlen_func_available,
)

if is_flash_attn_varlen_func_available():
    from vllm.attention.utils.fa_utils import (
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )
from vllm.logger import init_logger

from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)

logger = init_logger(__name__)

from .moba_attn_ops import moba_attn_varlen_paged
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend, 
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)

def chk(name, x, prefix="", show=False):
    if x is None: return
    if prefix != "":
        name = prefix + "." + name
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).sum().item()
        max = torch.abs(x).max().item()
        min = torch.abs(x).min().item()
        print(f"[BAD] {name}: nonfinite={bad}, dtype={x.dtype}, shape={tuple(x.shape)}, max={max}, min={min}")
        
        raise RuntimeError(f"nonfinite in {name}")
    if show:
        max = torch.abs(x).max().item()
        min = torch.abs(x).min().item()
        print(f"[GOOD] {name}: dtype={x.dtype}, shape={tuple(x.shape)}, max={max}, min={min}")

# from vllm.attention.backends.registry import AttentionBackendEnum, register_backend
# @register_backend(AttentionBackendEnum.FLASH_ATTN)
class MobaSseFlashAttentionBackend(FlashAttentionBackend):

    # @staticmethod
    # def get_name() -> str:
    #     return "MOBA_SSE_FLASH_ATTN"
    
    @staticmethod
    def get_impl_cls() -> type["SseMobaFlashAttentionImpl"]:
        return SseMobaFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["SseMobaFlashAttentionMetadataBuilder"]:
        return SseMobaFlashAttentionMetadataBuilder


class SseMobaFlashAttentionMetadata(FlashAttentionMetadata):
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int


class SseMobaFlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder, AttentionMetadataBuilder[SseMobaFlashAttentionMetadata]):
    
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    reorder_batch_threshold: int = 1

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SseMobaFlashAttentionMetadata:
        
        metadata: SseMobaFlashAttentionMetadata = FlashAttentionMetadataBuilder.build(
            self,
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )
        
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )
        # Update metadata with MoBA-specific fields
        metadata.num_decodes = num_decodes
        metadata.num_decode_tokens = num_decode_tokens
        metadata.num_prefills = num_prefills
        metadata.num_prefill_tokens = num_prefill_tokens
        # print(f"Built SseMobaFlashAttentionMetadata: num_decodes={num_decodes}, num_decode_tokens={num_decode_tokens}, num_prefills={num_prefills}, num_prefill_tokens={num_prefill_tokens}")

        return metadata
    
class SseMobaFlashAttentionImpl(FlashAttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        is_moba: bool = False,
        moba_topk: int | None = None,
        moba_chunk_size: int | None = None,
    ):
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
        )
        self.is_moba = is_moba
        self.moba_topk = moba_topk
        self.moba_chunk_size = moba_chunk_size
        
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SseMobaFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # print(f"Running SseMobaFlashAttentionImpl forward with is_moba={self.is_moba}, type of attn_metadata: {type(attn_metadata)}")
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
        
        if not self.is_moba:
            return super().forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        else: # MoBA attention
            # num of decode seqs and tokens
            num_decodes = attn_metadata.num_decodes
            num_decode_tokens = attn_metadata.num_decode_tokens
            num_prefills = attn_metadata.num_prefills
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            
            cu_seqlens_q = attn_metadata.query_start_loc
            max_seqlen_q = attn_metadata.max_query_len
            
            key_cache, value_cache = kv_cache.unbind(0)
            # key and value may be None in the case of cross attention. They are
            # calculated once based on the output from the encoder and then cached
            # in KV cache.
            if (
                self.kv_sharing_target_layer_name is None
                and key is not None
                and value is not None
            ):
                # Reshape the input keys and values and store them in the cache.
                # Skip this if sharing KV cache with an earlier attention layer.
                # NOTE(woosuk): Here, key and value are padded while slot_mapping is
                # not padded. However, we don't need to do key[:num_actual_tokens]
                # and value[:num_actual_tokens] because the reshape_and_cache_flash
                # op uses the slot_mapping's shape to determine the number of
                # actual tokens.
                reshape_and_cache_flash(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )

            if self.kv_cache_dtype.startswith("fp8"):
                # queries are quantized in the attention layer
                dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    self.kv_cache_dtype
                )
                key_cache = key_cache.view(dtype)
                value_cache = value_cache.view(dtype)

            if num_prefill_tokens > 0:
                q_p = query[num_decode_tokens: num_decode_tokens + num_prefill_tokens]
                cu_base = cu_seqlens_q[num_decodes]
                cu_seqlens_q_p = cu_seqlens_q[num_decodes: num_decodes + num_prefills + 1] - cu_base

                seq_lens_p = attn_metadata.seq_lens[
                    num_decodes: num_decodes + num_prefills
                ]
                block_table_p = attn_metadata.block_table[
                    num_decodes: num_decodes + num_prefills
                ]
                # max_seqlen_q_p = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
                max_seqlen_q_p = int((cu_seqlens_q_p[1:] - cu_seqlens_q_p[:-1]).max().item()) if cu_seqlens_q_p.numel() > 1 else 0
                max_seqlen_k_p = int(seq_lens_p.max().item()) if seq_lens_p.numel() > 0 else 0
                # logger.info(f"{q_p.shape=}, {key.shape=}")
                out_prefill = moba_attn_varlen_paged(
                    q=q_p,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    cu_seqlens_q=cu_seqlens_q_p,
                    seqused_k=seq_lens_p,
                    block_table=block_table_p,
                    max_seqlen_q=max_seqlen_q_p,
                    max_seqlen_k=max_seqlen_k_p,
                    moba_chunk_size=self.moba_chunk_size,
                    moba_topk=self.moba_topk,
                    kv_cache_dtype=self.kv_cache_dtype,
                    k_scale=layer._k_scale,
                    v_scale=layer._v_scale,
                )
                # chk("out_prefill", out_prefill)
                output[num_decode_tokens: num_decode_tokens + num_prefill_tokens] = out_prefill

            if num_decode_tokens > 0:
                seqused_k = attn_metadata.seq_lens
                max_seqlen_k = attn_metadata.max_seq_len
                block_table = attn_metadata.block_table
                scheduler_metadata = attn_metadata.scheduler_metadata

                descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)
                # print(f"{num_decodes=}, {num_decode_tokens=}, {num_prefills=}, {num_prefill_tokens=};\n Shape of query[:num_decode_tokens]: {query[:num_decode_tokens].shape}, block_table shape: {block_table.shape}, cu_seqlens_q[:num_decodes + 1] shape: {cu_seqlens_q[:num_decodes + 1].shape}, seqused_k[:num_decode_tokens] shape: {seqused_k[:num_decode_tokens].shape}")
                flash_attn_varlen_func(
                    q=query[:num_decode_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_decode_tokens],
                    cu_seqlens_q=cu_seqlens_q[:num_decodes + 1],
                    max_seqlen_q=1,
                    seqused_k=seqused_k[:num_decode_tokens],
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=self.sliding_window,
                    block_table=block_table[:num_decodes],
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                )
                # chk("out_decode", output[:num_decode_tokens], show=True)
            return output

