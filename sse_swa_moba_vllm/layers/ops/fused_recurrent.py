
import torch

from vllm.triton_utils import tl, triton

from vllm.model_executor.layers.fla.ops.op import exp


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        "IS_SSE": lambda args: args["ssm_state_expert_indices"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def sse_fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    ssm_state_expert_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.constexpr, # batch size
    H: tl.constexpr, # num of head q/k
    HV: tl.constexpr, # num of head v
    K: tl.constexpr, # head dimension of key/query
    V: tl.constexpr, # head dimension of value
    BK: tl.constexpr, # block size for K
    BV: tl.constexpr, # block size for V
    stride_init_state_token: tl.constexpr, # stride to jump to next token in initial state
    stride_init_state_expert: tl.constexpr, # stride to jump to next expert in initial state
    stride_final_state_token: tl.constexpr, # stride to jump to next token in final state
    stride_final_state_expert: tl.constexpr, # stride to jump to next expert in final state 
    stride_indices_seq: tl.constexpr, # stride to jump to next sequence in ssm_state_indices
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
    IS_SSE: tl.constexpr,
    num_partitions: tl.constexpr,
):
    # i_k K方向块编号， i_v V方向块编号， i_nh 序列编号和head v编号的组合
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV # n means sequence index, hv means head v index
    i_h = i_hv // (HV // H) # corresponding head index in q/k (like multi-query attention)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        ) # begin and end offset of the sequence
        all = T
        T = eos - bos
    else:
        # if fixed-length, all sequences have the same length T
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK) # offset for key/query block
    o_v = i_v * BV + tl.arange(0, BV) # offset for value block

    p_q = q + (bos * H + i_h) * K + o_k # pointer to the current query block
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE: # beta is headwise vector for each value dimension
        p_beta = beta + (bos * HV + i_hv) * V + o_v # beta shape: [B, T, HV, V]
    else: # beta is scalar for each head v
        p_beta = beta + bos * HV + i_hv # beta shape: [B, T, HV]

    if not IS_KDA:
        # scalar g for each head v
        p_g = g + bos * HV + i_hv
    else:
        # vector g for each head v, shape: [B, T, HV, K] 
        # one key head can correspond to multiple value heads
        p_gk = g + (bos * HV + i_hv) * K + o_k
    # 为什么是 i_k * all？ 是因为 o shape [NK, B, T, HV, V]
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v 

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :] # mask_k[:, None] is [BK, 1], mask_v[None, :] is [1, BV]
    # mask_h = 1 only when both key and value indices are valid

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if IS_SSE:
        i_ex = tl.load(ssm_state_expert_indices + i_n).to(tl.int64) # ssm_state_expert_indices[i_n] indicate the expert index for sequence i_n
    else:
        i_ex = 0 # don't have ssm_state_expert_indices, 默认全部映射到 expert 0

    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING: # you have ssm_state_indices to map the sequences to initial states
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1 # num_accepted_tokens[i_n] is number of accepted tokens for sequence i_n, so the index of the current token is num_accepted_tokens[i_n] - 1
            else:
                i_t = 0 # 没有 spec decoding 时，即每次只接受一个新 token，初始状态对应第 0 个 token
            
            # 加上 slot 和 expert 的偏移
            p_h0 = (
                h0
                + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to( # ssm_state_indices[i_n, i_t] indicate the slot index
                    tl.int64
                )
                * stride_init_state_token
                + i_ex * stride_init_state_expert
            ) 
        else: # don't have ssm_state_indices, 默认按照 token-level 一一对应
            p_h0 = h0 + bos * num_partitions * HV * K * V + i_ex * stride_init_state_expert
        # shape [, expert, HV, K, V]
        # 加上 head v, key and value 的偏移
        p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        # [BK, BV]
        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)
        else:
            b_gk = tl.load(p_gk).to(tl.float32)
            b_h *= exp(b_gk[:, None])
        # [BV]
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BK, BV]
        b_h += b_k[:, None] * b_v[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        # 计算 slot 和 expert 的偏移
        if INPLACE_FINAL_STATE:
            p_ht = (
                ht
                + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                    tl.int64
                )
                * stride_final_state_token
                + i_ex * stride_final_state_expert
            )
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token + i_ex * stride_final_state_expert
        # 计算 head v, key and value 的偏移
        p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)


def sse_fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    ssm_state_expert_indices: torch.Tensor | None = None,
    num_partitions: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2] # head v
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    o = q.new_empty(NK, *v.shape) # [NK, B, T, HV, V] but NK is expected to be 1
    if inplace_final_state: # store final state in-place
        final_state = initial_state
    else: # store all the states for each token
        final_state = q.new_empty(T, num_partitions, HV, K, V, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0) # stride to jump to next slot in initial state
    stride_init_state_expert = initial_state.stride(1) # stride to jump to next expert in initial state
    stride_final_state_token = final_state.stride(0)
    stride_final_state_expert = final_state.stride(1)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1: # only sequence dimension
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else: # two dimensions: sequence and token (used in spec decoding)
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    sse_fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        ssm_state_expert_indices=ssm_state_expert_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_init_state_expert=stride_init_state_expert,
        stride_final_state_token=stride_final_state_token,
        stride_final_state_expert=stride_final_state_expert,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        IS_KDA=False,
        num_warps=num_warps,
        num_stages=num_stages,
        num_partitions=num_partitions,
    )
    o = o.squeeze(0)
    return o, final_state


class SseFusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        inplace_final_state: bool = True,
        cu_seqlens: torch.LongTensor | None = None,
        ssm_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        ssm_state_expert_indices: torch.Tensor | None = None,
        num_partitions: int = 1,
    ):
        o, final_state = sse_fused_recurrent_gated_delta_rule_fwd(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            ssm_state_expert_indices=ssm_state_expert_indices,
            num_partitions=num_partitions,
        )

        return o, final_state


def sse_fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    ssm_state_expert_indices: torch.Tensor | None = None,
    num_partitions: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        inplace_final_state: bool:
            Whether to store the final state in-place to save memory.
            Default: `True`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        ssm_state_indices (Optional[torch.Tensor]):
            Indices to map the input sequences to the initial/final states.
        num_accepted_tokens (Optional[torch.Tensor]):
            Number of accepted tokens for each sequence during decoding.
        num_partitions: int:
            Number of partitions for ssm state. 

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            cu_seqlens=cu_seqlens
        )
    """
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = SseFusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
        ssm_state_expert_indices,
        num_partitions,
    )
    return o, final_state
