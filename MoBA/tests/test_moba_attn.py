import torch
import pytest
import random
from moba.moba_naive import moba_attn_varlen_naive
from moba.moba_efficient import moba_attn_varlen


def generate_data(batch, seqlen, num_q_head, num_kv_head, headdim, dtype):
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.cuda.current_device()

    # gen qkv
    q = torch.randn(
        (seqlen, num_q_head, headdim), dtype=dtype, device=device, requires_grad=True
    )
    k = torch.randn(
        (seqlen, num_kv_head, headdim), dtype=dtype, device=device, requires_grad=True
    )
    v = torch.randn(
        (seqlen, num_kv_head, headdim), dtype=dtype, device=device, requires_grad=True
    )

    # gen cu seqlen
    cu_seqlen = random.sample(range(1, seqlen - 1), batch - 1) if batch > 1 else []
    cu_seqlen.sort()
    cu_seqlen = [0] + cu_seqlen + [seqlen]
    cu_seqlen = torch.tensor(cu_seqlen, device=device, dtype=torch.int32)

    # max_seqlen
    max_seqlen = torch.amax(cu_seqlen[1:] - cu_seqlen[:-1])

    return q, k, v, cu_seqlen, max_seqlen.item()


@pytest.mark.parametrize("batch", [1, 4, 7])  # can be arbitrary
@pytest.mark.parametrize("head", [1, 2, 4, 8])
@pytest.mark.parametrize("seqlen", [512, 1024, 2048])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("moba_chunk_size", [128, 256, 1024])
@pytest.mark.parametrize("moba_topk", [2, 3, 4])
def test_attn_varlen_moba(batch, head, seqlen, head_dim, moba_chunk_size, moba_topk):
    dtype = torch.bfloat16
    eps = 2e-2

    # Get data
    q, k, v, cu_seqlen, max_seqlen = generate_data(
        batch, seqlen, head, head, head_dim, dtype
    )
    vo_grad = torch.randn_like(q)

    # varlen func
    o = moba_attn_varlen(
        q,
        k,
        v,
        cu_seqlen,
        max_seqlen,
        moba_chunk_size=moba_chunk_size,
        moba_topk=moba_topk,
    )
    torch.autograd.backward(o, vo_grad)
    gqkv = torch.stack((q._grad.clone(), k._grad.clone(), v._grad.clone()), dim=1)

    # ref with bf16
    q._grad.zero_()
    k._grad.zero_()
    v._grad.zero_()
    o_ref = moba_attn_varlen_naive(
        q,
        k,
        v,
        cu_seqlen,
        max_seqlen,
        moba_chunk_size=moba_chunk_size,
        moba_topk=moba_topk,
    )
    torch.autograd.backward(o_ref, vo_grad)
    gqkv_ref = torch.stack((q._grad.clone(), k._grad.clone(), v._grad.clone()), dim=1)

    # diff on bf16
    o_diff = (o - o_ref).abs()
    print("output diff:", o_diff.max().item(), o_diff.mean().item())
    assert torch.allclose(o, o_ref, atol=eps, rtol=eps), (
        (o - o_ref).abs().mean(),
        (o - o_ref).abs().max(),
    )

    gqkv_diff = (gqkv - gqkv_ref).abs()
    print("grad diff:", gqkv_diff.max().item(), gqkv_diff.mean().item())
    assert torch.allclose(gqkv, gqkv_ref, atol=eps, rtol=eps), (
        (gqkv - gqkv_ref).abs().mean(),
        (gqkv - gqkv_ref).abs().max(),
    )

    assert o_diff.max() < 4e-2, f"o_diff max {o_diff.max()}"
    assert o_diff.mean() < 4e-4, f"o_diff mean {o_diff.mean()}"
    assert gqkv_diff[:].max() < 4e-2, f"gqkv_diff max {gqkv_diff[:].max()}"
    assert gqkv_diff[:].mean() < 4e-4, f"gqkv_diff mean {gqkv_diff[:].mean()}"
