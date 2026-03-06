# https://github.com/vllm-project/vllm/blob/d44e9df7d49a9bb3400b002c38c06fae2dd7d1e8/vllm/model_executor/layers/kda.py
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from vllm.config.cache import MambaDType
from vllm.config.model import ModelDType
from vllm.utils.torch_utils import get_kv_cache_torch_dtype

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, 
    ReplicatedLinear,
    RowParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
# from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator, MambaStateShapeCalculator
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.qwen3_next import fused_gdn_gating
from vllm.model_executor.model_loader.weight_utils import LoaderFunction

import math
# from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
# from fla.modules import RMSNorm, ShortConvolution
# from fla.ops.gla import chunk_gla, fused_recurrent_gla
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.sse import prepare_sample_relpos_global_index_flat
from .ops.fused_recurrent import sse_fused_recurrent_gated_delta_rule

from vllm.attention.layer import Attention
# from fla.modules import FusedRMSNormGated, RMSNorm
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.fla.ops.kda import FusedRMSNormGated
# from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding import get_rope

logger = init_logger(__name__)

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


def sort_along_l(q, k, v, gk, beta, e, cu_seqlens, K, emulq, emulk):
    _, L, H, D = q.shape
    N = e.size(-1)
    S = len(cu_seqlens) - 1

    e = F.softmax(e, dim=-1, dtype=torch.float) 
    topk_value, topk_expert = torch.topk(e, k=K, dim=2)  # [1, L, K]
    topk_value, e = topk_value.to(q.dtype), e.to(q.dtype)
    # mask_w 为每个 token 选择的 partition 置 1.
    mask_w = torch.zeros_like(e, dtype=torch.bool).scatter_(dim=-1, index=topk_expert, src=torch.ones_like(topk_expert, dtype=torch.bool))
    experts_flat = topk_expert.reshape(L * K)  # [L*K] 选择的专家
    values_flat  = topk_value.reshape(L * K)   # [L*K] 专家的分数

    # 因为 prepare_sample_relpos_global_index_flat 内部调用了 repeat_interleave 导致 cuda graph capture 失败，所以改成外部调用一次，传入结果
    sample_idx_flat, relpos_flat, global_idx_flat, lengths = prepare_sample_relpos_global_index_flat(cu_seqlens, K)  # ([L*K] * 3, S)
    # 分别表示每个 (token, expert) 对所属的样本 ID；且是从 [L] 复制成 [L, K] -> [L * K]
    # 每个 (token, expert) 对在样本内的相对位置；
    # 每个 (token, expert) 对对应的原始 token 全局索引
    assert sample_idx_flat.dtype == torch.long and relpos_flat.dtype == torch.long and global_idx_flat.dtype == torch.long

    bits_pos = int(lengths.max().item()).bit_length()
    bits_exp = int((N - 1)).bit_length()
    shift_exp  = bits_pos
    shift_samp = bits_pos + bits_exp
    # 把上面三种信息全部二进制编码到 key，这样可以按以下顺序排序
    # 即同一样本的排在一起后，同一专家的按全局顺序排在一起
    ## sort by (sample_idx <- expert_idx <- relpos_in_sample)
    key = (sample_idx_flat << shift_samp) | (experts_flat << shift_exp) | relpos_flat
    order = torch.argsort(key, stable=False)
    experts_sorted = experts_flat.take(order)
    sample_sorted  = sample_idx_flat.take(order)
    global_sorted  = global_idx_flat.take(order)   # gather index
    values_sorted  = values_flat.take(order)       # sorted eta
    # pos_sorted   = relpos_flat.take(order)
    ## x: [1, L, H, D] -> y: [1, L*K, H, D]
    # 按排序顺序 gather 张量
    index4gather = global_sorted[None, :, None, None].expand(1, L * K, H, D)
    if beta is None:
        q, k, v, gk = [torch.gather(x, dim=1, index=index4gather) for x in (q, k, v, gk)]  # GLA
    else:
        q, k, v = [torch.gather(x, dim=1, index=index4gather) for x in (q, k, v)]          # GDN
        gk, beta = [torch.gather(x, dim=1, index=index4gather[..., 0]) for x in (gk, beta)] # TODO: situations for GQA
    # 应用 expert 权重
    if emulq:
        q = q * values_sorted[None, :, None, None]
    if emulk:
        k = k * values_sorted[None, :, None, None]

    ## calculate offsets (new cu_seqlens)
    #  唯一标识每个 (sample, expert) 组合（0 ~ S*N-1）即把来自同一样本，同一专家的标识成一样
    pair_id = sample_sorted * N + experts_sorted  # [L*K]
    counts = torch.bincount(pair_id, minlength=S * N)  # [S*N] # 统计每个标识包含多少 token（即块大小）
    state_sizes = counts.view(S, N) # [S, N] 每个样本每个专家对应的块大小
    offsets = torch.zeros(1 + S * N, dtype=torch.long, device=q.device)
    offsets[1:] = counts.cumsum(dim=0) # 用同一标识的块构造新的 cu_seqlens
    offsets = torch.unique(offsets) # 去掉没有 token 的块对应的 cu_seqlens 重复值
    
    return q, k, v, gk, beta, e, mask_w, offsets, state_sizes, global_sorted


def sse_swa_gdn_h_func(
    q1: torch.Tensor, k1: torch.Tensor,
    q2: torch.Tensor, k2: torch.Tensor, v: torch.Tensor,
    g1: torch.Tensor, g2: torch.Tensor, 
    b1: torch.Tensor, b2: torch.Tensor,
    eta: torch.Tensor, core_attn_out: torch.Tensor,
    layer_name: str
) -> None:
    forward_context: ForwardContext = get_forward_context()
    # print("no_compile_layers keys:", list(forward_context.no_compile_layers.keys())[:50])
    # print("wanted layer_name:", layer_name)
    self = forward_context.no_compile_layers[layer_name]
    self._forward(
        q1, k1, q2, k2, v, 
        g1, g2, b1, b2, eta, core_attn_out,
    )

def sse_swa_gdn_h_func_fake(
    q1: torch.Tensor, k1: torch.Tensor,
    q2: torch.Tensor, k2: torch.Tensor, v: torch.Tensor,
    g1: torch.Tensor, g2: torch.Tensor, 
    b1: torch.Tensor, b2: torch.Tensor,
    eta: torch.Tensor, core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    return

direct_register_custom_op(
    op_name="sse_swa_gdn_h_func",
    op_func=sse_swa_gdn_h_func,
    mutates_args=["core_attn_out"],
    fake_impl=sse_swa_gdn_h_func_fake,
)

        
class SSE_GDN_H(nn.Module, MambaBase):
    @property
    def mamba_type(self):
        return "gdn_attention"
    
    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("ModelConfig and CacheConfig must be set.")
        return SSE_GDN_H.SSE_GDN_H_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype, self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # return MambaStateShapeCalculator.gdn_attention_state_shape(
        #     self.tp_size, self.num_heads, self.head_dim
        # )
        conv_state_shape = (0, 0)
        if self.use_short_conv:
            conv_state_shape = (self.tp_heads, self.conv_size - 1)
        
        num_partition = 1 + self.num_sparse_partition
        recurrent_state_shape = (num_partition, self.sse_tp_kv_heads, self.sse_head_k_dim, self.sse_head_v_dim)
        if self.use_short_conv:
            return (conv_state_shape, conv_state_shape, recurrent_state_shape)
        else:
            return ((), (), recurrent_state_shape)

    @classmethod
    def SSE_GDN_H_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType = "auto",
    ):
        state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        ssm_dtype = torch.float32 if mamba_ssm_cache_dtype == "auto" \
            else get_kv_cache_torch_dtype(mamba_ssm_cache_dtype, model_dtype)
        # logger.info(f"SSE_SWA {state_dtype=}, {ssm_dtype=}")
        return (state_dtype, state_dtype, ssm_dtype)

    @classmethod
    def SSE_GDN_H_state_shape(
        cls,
        tp_world_size: int,
        num_heads: int,
        num_v_heads: int | None = None,
        head_k_dim: int | None = None,
        head_v_dim: int | None = None,
        use_short_conv: bool = False,
        conv_kernel_size: int = 4,
        sparse_partition: int = 0,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        conv_state_shape = (0, 0)
        if use_short_conv:
            conv_state_shape = (num_heads // tp_world_size, conv_kernel_size - 1)
        
        num_partition = 1 + sparse_partition
        recurrent_state_shape = (num_partition, num_v_heads // tp_world_size, head_k_dim, head_v_dim)
        if use_short_conv:
            return (conv_state_shape, conv_state_shape, recurrent_state_shape)
        else:
            return ((), (), recurrent_state_shape)

    """
    为 channel-major 布局的参数 (如 A_log, dt_bias) 创建自定义 loader
    假设原始权重形状: [2 * total_heads] (布局: [a1_all_heads, a2_all_heads])
    目标: 每个 TP rank 持有 head-major 布局的局部权重 [a1_i, a2_i, a1_j, a2_j, ...]
    
    Args:
        total_heads: 模型总 head 数
    """
    def make_channel_major_loader(total_heads: int):
        def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            
            # 验证原始权重形状
            assert loaded_weight.ndim == 1, f"Expected 1D weight, got {loaded_weight.shape}"
            assert loaded_weight.size(0) == 2 * total_heads, \
                f"Weight size {loaded_weight.size(0)} != 2*total_heads ({2*total_heads})"
            
            # 1. 将原始权重拆分为两个通道 (channel-major -> split)
            mid = total_heads
            a1_all = loaded_weight[:mid]   # [total_heads] : 第一个通道 (e.g., B)
            a2_all = loaded_weight[mid:]   # [total_heads] : 第二个通道 (e.g., C)
            
            # 2. 计算每个 rank 负责的 head 范围
            heads_per_rank = total_heads // tp_size
            start_idx = tp_rank * heads_per_rank
            end_idx = start_idx + heads_per_rank
            
            # 3. 从每个通道切分对应 heads
            a1_local = a1_all.narrow(0, start_idx, heads_per_rank)  # [heads_per_rank]
            a2_local = a2_all.narrow(0, start_idx, heads_per_rank)  # [heads_per_rank]
            
            # 4. 重排为 head-major 布局: [a1_i, a2_i, a1_j, a2_j, ...]
            #    通过 stack(dim=1) + reshape 实现高效 interleaving
            local_weights = torch.stack([a1_local, a2_local], dim=1).reshape(-1)  # [2 * heads_per_rank]
            
            # 5. 验证目标形状并复制
            assert param.shape == local_weights.shape, \
                f"Shape mismatch: param={param.shape}, local_weights={local_weights.shape}"
            param.data.copy_(local_weights)

        return loader

    def __init__(
        self, 
        layer_idx: int,
        hidden_size: int =2048,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        model_config: ModelConfig | None = None,
        expand_v: float = 1.0,
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
        use_k_softmax: bool = False, # but gla True
        emulq: bool = True,
        emulk: bool = True,
        qkv_bias: bool = False,
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()

        self.prefix = prefix
        self.layer_idx = layer_idx
        self.quant_config = quant_config
        self.cache_config = cache_config
        self.model_config = model_config

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        assert num_reader < num_sparse_partition and num_writer < num_sparse_partition, \
            "num_reader and num_writer must be less than num_sparse_partition."
        assert sse_implementation in ["mask", "varlen"], \
            f"Unknown SSE implementation {sse_implementation}"

        self.num_sparse_partition = num_sparse_partition
        self.num_writer = num_writer
        self.num_reader = num_reader
        assert self.num_writer == self.num_reader, "Only support num_writer == num_reader for varlen."
        assert sse_implementation == "varlen", "Only support varlen implementation for vllm."

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

        self.sse_head_k_dim = head_dim
        self.sse_head_v_dim = int(self.head_dim * self.expand_v)
        self.sse_key_dim = int(self.sse_num_kv_heads * self.sse_head_k_dim)
        self.sse_value_dim = int(self.sse_num_kv_heads * self.sse_head_v_dim)
        
        self.qkv_bias = qkv_bias

        assert self.expand_v == 1.0, "Only support expand_v == 1.0 for GDN-H."

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # Validate and calculate tensor parallel heads
        assert self.num_heads % self.tp_size == 0, \
            f"num_heads ({self.num_heads}) must be divisible by tp_size ({self.tp_size})"
        assert self.sse_num_kv_heads % self.tp_size == 0, \
            f"sse_num_kv_heads ({self.sse_num_kv_heads}) must be divisible by tp_size ({self.tp_size})"

        self.tp_heads = self.num_heads // self.tp_size
        self.sse_tp_kv_heads = max(1, self.sse_num_kv_heads // self.tp_size)

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.sse_num_kv_heads * self.head_dim * expand_v, self.sse_value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.sse_key_dim}. "
                f"Resulting value_dim would be {self.sse_num_kv_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )
        if self.sse_num_kv_heads > self.num_heads and self.sse_num_kv_heads % self.num_heads != 0:
            raise ValueError(
                f"sse_num_kv_heads={self.sse_num_kv_heads} must be divisible by num_heads={self.num_heads}.",
            )

        if not math.isclose(head_dim * expand_v, self.sse_head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
            )
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        self.sse_q_proj = ColumnParallelLinear(
            hidden_size, self.sse_key_dim, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.sse_q_proj",
        )
        self.sse_k_proj = ColumnParallelLinear(
            hidden_size, self.sse_key_dim, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.sse_k_proj",
        )
        self.sse_v_proj = ColumnParallelLinear(
            hidden_size, self.sse_value_dim, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.sse_v_proj",
        )

        self.lora_q_proj_A = ReplicatedLinear(
            hidden_size, self.sse_head_v_dim, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.lora_q_proj_loraA",
        )
        self.lora_q_proj_B = ColumnParallelLinear(
            self.sse_head_v_dim, self.sse_key_dim, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.lora_q_proj_loraB",
        )
        self.lora_k_proj_A = ReplicatedLinear(
            hidden_size, self.sse_head_v_dim, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.lora_k_proj_loraA",
        )
        self.lora_k_proj_B = ColumnParallelLinear(
            self.sse_head_v_dim, self.sse_key_dim, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.lora_k_proj_loraB",
        )
        # 因为 hf 的 a,b_proj 是 [a1 weight, a2 weight] concat 在一起的
        # 直接用 ColumnParallelLinear 不能正确 tensor parallel 切分
        self.sse_a_proj = MergedColumnParallelLinear(
            hidden_size, [self.sse_num_kv_heads] * 2, bias=False,
            # quant_config=quant_config, # no quant for a,b proj
            prefix=f"{prefix}.sse_a_proj",
        )
        self.sse_b_proj = MergedColumnParallelLinear(
            hidden_size, [self.sse_num_kv_heads] * 2, bias=False,
            # quant_config=quant_config,
            prefix=f"{prefix}.sse_b_proj",
        )
        self.A_log = nn.Parameter(torch.empty(self.sse_tp_kv_heads * 2, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.sse_tp_kv_heads * 2, dtype=torch.float32))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        # 只加载当前 rank 负责的 head 部分
        set_weight_attrs(
            self.A_log, {"weight_loader": SSE_GDN_H.make_channel_major_loader(self.sse_num_kv_heads)}
        )
        set_weight_attrs(
            self.dt_bias, {"weight_loader": SSE_GDN_H.make_channel_major_loader(self.sse_num_kv_heads)}
        )

        self.sse_e_proj = ReplicatedLinear(
            hidden_size, self.num_sparse_partition, bias=False,
            # quant_config=quant_config,
            prefix=f"{prefix}.sse_e_proj",
        ) # 用于之后计算每个 token 的 TopK expert（K = num_writer）

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d_shared = ColumnParallelLinear(
                input_size=self.conv_size,
                output_size=self.sse_key_dim,
                bias=conv_bias,
                prefix=f"{prefix}.q_conv1d_shared",
            )
            self.k_conv1d_shared = ColumnParallelLinear(
                input_size=self.conv_size,
                output_size=self.sse_key_dim,
                bias=conv_bias,
                prefix=f"{prefix}.k_conv1d_shared",
            )

            self.q_conv1d_shared.weight.data = self.q_conv1d_shared.weight.unsqueeze(1)
            self.k_conv1d_shared.weight.data = self.k_conv1d_shared.weight.unsqueeze(1)

        if use_output_gate:
            self.sse_g_proj_A = ReplicatedLinear(
                hidden_size, self.sse_head_v_dim, bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.sse_g_proj_loraA",
            )
            self.sse_g_proj_B = ColumnParallelLinear(
                self.sse_head_v_dim, self.sse_value_dim, bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.sse_g_proj_loraB",
            )
            self.sse_o_norm = FusedRMSNormGated(self.sse_head_v_dim, eps=rms_norm_eps)
        else:
            self.sse_o_norm = RMSNorm(self.sse_head_v_dim, eps=rms_norm_eps)
        
        self.sse_o_proj = RowParallelLinear(
            self.sse_value_dim, hidden_size, bias=False,
            quant_config=quant_config, 
            prefix=f"{prefix}.sse_o_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        
        # hidden_state [num_tokens, hidden_size]
        num_tokens = hidden_states.size(0)

        # forward_context = get_forward_context()
        # attn_metadata: AttentionMetadata = forward_context.attn_metadata
        # if attn_metadata is not None:
        #     assert isinstance(attn_metadata, dict)
        #     attn_metadata = attn_metadata[self.prefix]
        #     assert isinstance(attn_metadata, GDNAttentionMetadata)

        #     sample_idx_flat, relpos_flat, global_idx_flat, lengths = \
        #         prepare_sample_relpos_global_index_flat(attn_metadata.non_spec_query_start_loc, self.num_writer)
        # else: # V1 profile run, no attn_metadata
        #     sample_idx_flat = torch.empty(num_tokens * self.num_writer, dtype=torch.long, device=hidden_states.device)
        #     relpos_flat = torch.empty(num_tokens * self.num_writer, dtype=torch.long, device=hidden_states.device)
        #     global_idx_flat = torch.empty(num_tokens * self.num_writer, dtype=torch.long, device=hidden_states.device)
        #     lengths = torch.empty(num_tokens * self.num_writer, dtype=torch.long, device=hidden_states.device)
        
        sse_q1, _ = self.sse_q_proj(hidden_states)
        sse_k1, _ = self.sse_k_proj(hidden_states)
        sse_q2 = sse_q1 + self.lora_q_proj_B(self.lora_q_proj_A(hidden_states)[0])[0] # [0] because Linear returns output and bias
        sse_k2 = sse_k1 + self.lora_k_proj_B(self.lora_k_proj_A(hidden_states)[0])[0]
        sse_v, _ = self.sse_v_proj(hidden_states)
        sse_a, _ = self.sse_a_proj(hidden_states)
        sse_b, _ = self.sse_b_proj(hidden_states) # [num_tokens, 2 * sse_tp_kv_heads]
        g, beta = fused_gdn_gating(self.A_log, sse_a, sse_b, self.dt_bias)
        if self.allow_neg_eigval:
            beta = beta * 2.        
        b1, b2 = torch.chunk(beta, 2, dim=-1) # [1, num_tokens, 2 * HV] -> [1, num_tokens, HV]
        g1, g2 = torch.chunk(g, 2, dim=-1)
        eta = self.sse_e_proj(hidden_states)[0] # [num_tokens, num_sparse_partition]
        # chk("sse_q1", sse_q1, self.prefix, show=True)
        # chk("sse_q2", sse_q2, self.prefix, show=True)
        # chk("sse_k1", sse_k1, self.prefix, show=True)
        # chk("sse_k2", sse_k2, self.prefix, show=True)
        # chk("sse_v", sse_v, self.prefix, show=True)
        # chk("sse_beta", beta, self.prefix, show=True)
        # chk("sse_g", g, self.prefix, show=True)
        # print(f"{hidden_states.shape=}, first 10 hidden states: {hidden_states[:10]}")
        # chk("eta", eta, self.prefix, show=True)

        core_attn_out = torch.zeros(
            (1, num_tokens, self.sse_tp_kv_heads, self.sse_head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.sse_swa_gdn_h_func(
            sse_q1, sse_k1, sse_q2, sse_k2, sse_v, 
            g1, g2, b1, b2,
            eta, core_attn_out,
            self.prefix
        )

        if self.use_output_gate:
            g = self.sse_g_proj_B(self.sse_g_proj_A(hidden_states)[0])[0]
            g = rearrange(g, "n (h d) -> 1 n h d", d=self.sse_head_v_dim)
            core_attn_out = self.sse_o_norm(core_attn_out, g)
        else:
            core_attn_out = self.sse_o_norm(core_attn_out)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        sse_o, _ = self.sse_o_proj(core_attn_out)
        output[:] = sse_o
        # output[:] = (self.sse_merge_norm(sse_o) + self.swa_merge_norm(swa_o)) / 2


    def _forward(
        self,
        q1: torch.Tensor, k1: torch.Tensor,
        q2: torch.Tensor, k2: torch.Tensor, v: torch.Tensor,
        g1: torch.Tensor, g2: torch.Tensor,
        b1: torch.Tensor, b2: torch.Tensor,
        eta: torch.Tensor, core_attn_out: torch.Tensor,
    ) -> None:
        # see https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backend.py#L284 for AttentionMetadata definition
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return
        
        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        has_initial_state = attn_metadata.has_initial_state
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        num_actual_tokens = attn_metadata.num_actual_tokens
        constant_caches = self.kv_cache[forward_context.virtual_engine]

        q1, k1 = q1[:num_actual_tokens], k1[:num_actual_tokens]
        q2, k2 = q2[:num_actual_tokens], k2[:num_actual_tokens]
        v = v[:num_actual_tokens]
        g1, g2 = g1[:, :num_actual_tokens], g2[:, :num_actual_tokens]
        b1, b2 = b1[:, :num_actual_tokens], b2[:, :num_actual_tokens]
        eta = eta[:num_actual_tokens]

        (conv_state_q, conv_state_k, recurrent_state) = constant_caches
        if self.use_short_conv:
            conv_state_q = conv_state_q.transpose(-1, -2)
            conv_state_k = conv_state_k.transpose(-1, -2)

            q_conv_weights = self.q_conv1d_shared.weight.view(
                self.q_conv1d_shared.size(0), self.q_conv1d_shared.size(2)
            )
            k_conv_weights = self.k_conv1d_shared.weight.view(
                self.k_conv1d_shared.size(0), self.k_conv1d_shared.size(2)
            )

            if attn_metadata.num_prefills > 0:
                q1 = causal_conv1d_fn(
                    q1,
                    q_conv_weights,
                    self.q_conv1d_shared.bias,
                    activation="silu",
                    conv_states=conv_state_q,
                    has_initial_state=has_initial_state,
                    cache_indices=non_spec_state_indices_tensor,
                    query_start_loc=non_spec_query_start_loc,
                    metadata=attn_metadata,
                ).transpose(0, 1)
                k1 = causal_conv1d_fn(
                    k1,
                    k_conv_weights,
                    self.k_conv1d_shared.bias,
                    activation="silu",
                    conv_states=conv_state_k,
                    has_initial_state=has_initial_state,
                    cache_indices=non_spec_state_indices_tensor,
                    query_start_loc=non_spec_query_start_loc,
                    metadata=attn_metadata,
                ).transpose(0, 1)
            else:
                decode_conv_indices = non_spec_state_indices_tensor[:attn_metadata.num_actual_tokens]
                q1 = causal_conv1d_update(
                    q1, 
                    conv_state_q,
                    q_conv_weights,
                    self.q_conv1d_shared.bias,
                    activation="silu",
                    conv_state_indices=decode_conv_indices,
                    validate_data=True,
                )
                k1 = causal_conv1d_update(
                    k1, 
                    conv_state_k,
                    k_conv_weights,
                    self.k_conv1d_shared.bias,
                    activation="silu",
                    conv_state_indices=decode_conv_indices,
                    validate_data=True,
                )
        q1, q2, k1, k2 = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.sse_head_k_dim), (q1, q2, k1, k2)
        )
        v = rearrange(v, "n (h d) -> 1 n h d", d=self.sse_head_v_dim)
        eta = eta.unsqueeze(0) # [1, num_tokens, num_sparse_partition]

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

        # print(f"{q1.shape=}, {k1.shape=}, {v.shape=}, {g1.shape=}, {b1.shape=}, {eta.shape=}, {self.sse_num_kv_heads}, {self.sse_tp_kv_heads=}")
        # actually dont need to repeat, but keep for code consistency
        # TODO: can optimize later
        # if self.sse_tp_kv_heads > self.tp_heads:
        #     group_factor = self.sse_tp_kv_heads // self.tp_heads
        #     q1, q2, k1, k2, g1, g2, b1, b2 = map(
        #         lambda x: repeat(x, '... h d -> ... (h g) d', g=group_factor),
        #         (q1, q2, k1, k2, g1, g2, b1, b2)
        #     )
        
        # |=====| start of sse_gdn_attention_varlen |=====|
        v1 = v
        v2 = v
        cu_seqlens = non_spec_query_start_loc
        S = len(cu_seqlens) - 1
        q2, k2, v2, g2, b2, eta, mask, offsets, state_sizes, global_sorted = sort_along_l(
            q2, k2, v2, g2, b2, eta, cu_seqlens, self.num_writer, self.emulq, self.emulk,
        )
        active_mask = state_sizes > 0 # [S, N], 有效的 (seq, partition) 对
        active_seq_ids2, active_partition_ids2 = torch.nonzero(active_mask, as_tuple=True)
        active_seq_slots2 = non_spec_state_indices_tensor[active_seq_ids2]  # [M]
        # print(f"active_mask: {active_mask}, active_seq_ids2: {active_seq_ids2}, active_partition_ids2: {active_partition_ids2}, active_seq_slots2: {active_seq_slots2}")

        q, k, g, b, v = [torch.cat(pair, dim=1) for pair in zip((q1, k1, g1, b1, v1), (q2, k2, g2, b2, v2))]
        active_seq_slots = torch.cat((
            non_spec_state_indices_tensor,
            active_seq_slots2
        ), dim=0)  # [M1 + M2], M means number of active (seq, partition) pairs
        active_partition_ids = torch.cat((
            torch.zeros_like(non_spec_state_indices_tensor),
            active_partition_ids2 + 1
        ), dim=0)  # [M1 + M2], 0 for shared, 1~N for sparse
        offsets = torch.cat([cu_seqlens.to(offsets), offsets[1:] + cu_seqlens[-1]])
        new_cu_seqlens = offsets

        # chk("sse_q", q, self.prefix, show=True)
        # chk("sse_k", k, self.prefix, show=True)
        # chk("sse_v", v, self.prefix, show=True)
        # chk("sse_b", b, self.prefix, show=True)
        # chk("sse_g", g, self.prefix, show=True)
        # print(f"{offsets=}, {state_sizes=}, {global_sorted=}")

        assert active_seq_slots.numel() == new_cu_seqlens.numel() - 1
        # print(f"active_seq_slots: {active_seq_slots}, active_partition_ids: {active_partition_ids}, new_cu_seqlens: {new_cu_seqlens}")

        if attn_metadata.num_prefills > 0:
            if has_initial_state is not None:
                # zero_idx 表示没有初始 state 的那些序列在 non_spec_state_indices_tensor 中的位置
                zero_idx = non_spec_state_indices_tensor[~has_initial_state]
                recurrent_state[zero_idx] = 0
                # print(f"{has_initial_state=}, {zero_idx=}, {active_seq_slots=}")
                # print("initial_state shape:", recurrent_state.shape) # torch.Size([3062, 5, 8, 128, 128])
            initial_state = recurrent_state[active_seq_slots, active_partition_ids].contiguous()
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_gated_delta_rule(
                q = q,
                k = k,
                v = v,
                g = g,
                beta = b,
                scale = None,
                initial_state = initial_state,
                output_final_state = True,
                cu_seqlens = new_cu_seqlens,
                head_first = False,
                use_qk_l2norm_in_kernel = True,
            )
            # chk("fla_o", o, self.prefix, show=True)
            # chk("fla_recurrent_state_rec", recurrent_state_rec, self.prefix, show=True)
            recurrent_state[active_seq_slots, active_partition_ids] = last_recurrent_state.to(
                dtype=recurrent_state.dtype,
            )
            # assert torch.allclose(recurrent_state[active_seq_slots, active_partition_ids], last_recurrent_state.to(
            #     dtype=recurrent_state.dtype,
            #     device=recurrent_state.device,
            # ))
        elif attn_metadata.num_decodes > 0:
            
            # TODO: 需要修改 gdn 内核, 使得能够直接读取和写入指定位置的 state
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = sse_fused_recurrent_gated_delta_rule(
                q = q,
                k = k,
                v = v,
                g = g,
                beta = b,
                scale = None,
                initial_state = recurrent_state,
                inplace_final_state = True,
                cu_seqlens = new_cu_seqlens,
                ssm_state_indices = active_seq_slots,
                num_accepted_tokens = None,
                use_qk_l2norm_in_kernel = True,
                ssm_state_expert_indices = active_partition_ids,
                num_partitions = self.num_sparse_partition + 1,
            )
            # tmp = torch.isclose(core_attn_out_non_spec1, core_attn_out_non_spec, rtol=1e-3, atol=1e-2)
            # # how many elements are not close?
            # not_close = core_attn_out_non_spec1.numel() - tmp.sum().item()
            # print(f"not close: {not_close}, percentage: {not_close / core_attn_out_non_spec1.numel() * 100}%")
            # # max diff between core_attn_out_non_spec1 and core_attn_out_non_spec
            # diff = torch.max(torch.abs(core_attn_out_non_spec1 - core_attn_out_non_spec))
            # print(f"max diff: {diff}")
            # # average diff
            # diff = torch.mean(torch.abs(core_attn_out_non_spec1 - core_attn_out_non_spec))
            # print(f"average diff: {diff}")
            # assert torch.allclose(core_attn_out_non_spec1, core_attn_out_non_spec, rtol=1e-5, atol=1e-5), f"first 10, core1 {core_attn_out_non_spec[:, :10, :5, :5]}, core {core_attn_out_non_spec[:, :10, :5, :5]}"
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None
        
        # if last_recurrent_state is not None and core_attn_out_non_spec is not None:
        #     chk("last_recurrent_state", last_recurrent_state, prefix=self.prefix, show=True)
        
        o1, o2 = core_attn_out_non_spec[:, :cu_seqlens[-1]], core_attn_out_non_spec[:, cu_seqlens[-1]:]
        # chk("o1", o1, self.prefix, show=True)
        # chk("o2", o2, self.prefix, show=True)
        
        o2_reduce = torch.zeros_like(o1)
        o2_reduce.index_add_(dim=1, index=global_sorted, source=o2) # 把 o2 按照 global index 汇总到对应位置
        # chk("o2_reduce", o2_reduce, self.prefix, show=True)

        core_attn_out[0, :num_actual_tokens] = o1 + o2_reduce
        # |=====| end of sse_gdn_attention_varlen |=====|

class SlidingWindowAttention(nn.Module):
    
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int = 2048,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        model_config: ModelConfig | None = None,
        head_dim: int = 256,
        num_heads: int = 6,
        qkv_bias: bool = False,
        swa_num_kv_heads: int | None = None,
        swa_qk_norm: bool = False,
        swa_dropout: float = 0.5,
        window_size: int | None = None,
        rope_theta: float | None = 10000.,
        rope_scaling: float | None = None,
        max_position_embeddings: int | None = None,
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.quant_config = quant_config
        self.cache_config = cache_config
        self.model_config = model_config

        self.head_dim = head_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = swa_num_kv_heads if swa_num_kv_heads is not None else num_heads
        
        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim # only this tp slice
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = self.head_dim**(-0.5)
        self.swa_dropout = swa_dropout
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.qk_norm = swa_qk_norm
        self.max_position_embeddings = max_position_embeddings

        self.swa_qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.swa_qkv_proj",
        )

        if self.qk_norm:
            self.swa_q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.swa_k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # self.rotary = RotaryEmbedding(
        #     head_size=self.head_dim,
        #     rotary_dim=self.head_dim,
        #     max_position_embeddings=self.max_position_embeddings,
        #     base=self.rope_theta,
        #     is_neox_style=True,
        #     dtype=torch.float32,
        # )
        rope_parameters = dict()
        rope_parameters['rope_theta'] = rope_theta
        if rope_scaling is not None:
            rope_parameters['factor'] = rope_scaling
        self.rotary = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_parameters,
            dtype=torch.float32,
            dual_chunk_attention_config=None,
        )

        self.swa_attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=window_size,
            # attn_backend=FlashAttentionBackend(),
            prefix=f"{prefix}.swa_attn",
        )
        self.swa_o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, hidden_size, bias=False,
            quant_config=quant_config, 
            prefix=f"{prefix}.swa_o_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ):
        qkv, _ = self.swa_qkv_proj(hidden_states)
        swa_q, swa_k, swa_v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.qk_norm:
            # swa_q, swa_k = self.swa_q_norm(swa_q), self.swa_k_norm(swa_k)
            swa_q = self.swa_q_norm(swa_q.view(-1, self.num_heads, self.head_dim)).view(
                -1, self.q_size
            )
            swa_k = self.swa_k_norm(swa_k.view(-1, self.num_kv_heads, self.head_dim)).view(
                -1, self.kv_size
            )

        
        swa_q, swa_k = self.rotary(
            positions, swa_q, swa_k
        )
        swa_o = self.swa_attn(
            swa_q, swa_k, swa_v,
        )
        swa_o, _ = self.swa_o_proj(swa_o)

        output[:] = swa_o

class SSE_SWA_Hybrid(nn.Module):

    def __init__(
        self, 
        layer_idx: int,
        hidden_size: int =2048,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        model_config: ModelConfig | None = None,
        expand_v: float = 1.0,
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
        rope_scaling: float | None = None,
        max_position_embeddings: int | None = None,
        # ===================
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()

        self.prefix = prefix
        self.sse_attn = SSE_GDN_H(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            quant_config=quant_config,
            cache_config=cache_config,
            model_config=model_config,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            mode=mode,
            use_output_gate=use_output_gate,
            use_short_conv=use_short_conv,
            allow_neg_eigval=allow_neg_eigval,
            conv_size=conv_size,
            conv_bias=conv_bias,
            num_sparse_partition=num_sparse_partition,
            num_writer=num_writer,
            num_reader=num_reader,
            sse_implementation=sse_implementation,
            sse_qk_relu=sse_qk_relu,
            use_q_softmax=use_q_softmax,
            use_k_softmax=use_k_softmax,
            emulq=emulq,
            emulk=emulk,
            qkv_bias=qkv_bias,
            rms_norm_eps=rms_norm_eps,
            prefix=f"{prefix}.sse_attn",
        )
        self.sliding_window_attn = SlidingWindowAttention(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            quant_config=quant_config,
            cache_config=cache_config,
            model_config=model_config,
            head_dim=head_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            swa_num_kv_heads=swa_num_kv_heads,
            swa_qk_norm=swa_qk_norm,
            swa_dropout=swa_dropout,
            window_size=window_size,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            prefix=f"{prefix}.swa_attn",
        )
        self.sse_merge_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.swa_merge_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ):
        sse_o = torch.empty_like(output)
        swa_o = torch.empty_like(output)
        self.sse_attn(hidden_states, positions, sse_o)
        self.sliding_window_attn(hidden_states, positions, swa_o)
        # chk("sse_o", sse_o, prefix=self.prefix, show=True)
        # chk("swa_o", swa_o, prefix=self.prefix, show=True)
        output[:] = (self.sse_merge_norm(sse_o) + self.swa_merge_norm(swa_o)) / 2
        # chk("sse_swa_o", output, prefix=self.prefix, show=True)
        
