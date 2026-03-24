import torch
import random
import time
from flash_attn import flash_attn_varlen_func
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



def test_attn_varlen_moba_speed(batch, head, seqlen, head_dim, moba_chunk_size, moba_topk, dtype=torch.bfloat16):
    """Speed test comparing v3 vs v4 moba attention"""
    # Get data
    q, k, v, cu_seqlen, max_seqlen = generate_data(batch, seqlen, head, head, head_dim, dtype)
    vo_grad = torch.randn_like(q)
    
    # Warmup
    warmup_iters = 3
    perf_test_iters = 10

    # Warmup
    for _ in range(warmup_iters):
        o = flash_attn_varlen_func(q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen, causal=True)
        torch.autograd.backward(o, vo_grad)
    
    torch.cuda.synchronize()
    start_flash = time.perf_counter()
    for _ in range(perf_test_iters):
        o = flash_attn_varlen_func(q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen, causal=True)
        torch.autograd.backward(o, vo_grad)
        
    torch.cuda.synchronize()
    time_flash = (time.perf_counter() - start_flash) / perf_test_iters * 1000


    # Warmup
    for _ in range(warmup_iters):
        om = moba_attn_varlen(q, k, v, cu_seqlen, max_seqlen, moba_chunk_size=moba_chunk_size, moba_topk=moba_topk)
        torch.autograd.backward(om, vo_grad)

        
    torch.cuda.synchronize()
    start_moba = time.perf_counter()
    for _ in range(perf_test_iters):
        om = moba_attn_varlen(q, k, v, cu_seqlen, max_seqlen, moba_chunk_size=moba_chunk_size, moba_topk=moba_topk)
        torch.autograd.backward(om, vo_grad)
    
    torch.cuda.synchronize()
    time_moba = (time.perf_counter() - start_moba) / perf_test_iters * 1000
    
    print(f"\nbatch:{batch} head:{head} seqlen:{seqlen} chunk:{moba_chunk_size} topk:{moba_topk}")
    print(f"Flash: {time_flash:.2f}ms, MoBA: {time_moba:.2f}ms")
    print(f"Speedup:  {time_flash / time_moba:.2f}x")


if __name__ == "__main__":
    test_attn_varlen_moba_speed(batch=1, head=1, seqlen=32768, head_dim=128, moba_chunk_size=512, moba_topk=3)
    print("simple speed test finished")

