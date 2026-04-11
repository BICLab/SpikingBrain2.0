
import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F

import vllm.envs as envs
from vllm.config.vllm import VllmConfig
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
# from vllm.attention.backends.abstract import AttentionMetadata
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
# from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.models.vision import get_vit_attn_backend

from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, 
    ReplicatedLinear,
    RowParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)
# from fla.modules import RMSNorm
from vllm.model_executor.layers.attention.attention import (
    unified_attention_with_output,
    unified_attention,
    unified_kv_cache_update,
    should_load_quant_weights,
    set_default_quant_scales
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
# from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding import get_rope

# from vllm.attention.layers.mm_encoder_attention import maybe_get_vit_flash_attn_backend
from vllm.v1.attention.selector import get_attn_backend
# from vllm.attention.utils.fa_utils import get_flash_attn_version
# from vllm.attention.utils.kv_transfer_utils import maybe_transfer_kv_layer

from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend, AttentionType

from vllm.utils.torch_utils import (
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.model_executor.layers.attention.attention import (
    _init_kv_cache_quant,
    validate_kv_sharing_target,
    get_attention_context,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

from vllm.logger import init_logger
logger = init_logger(__name__)

def chk(name, x, prefix="", show=False):
    if x is None: return
    if prefix != "":
        name = prefix + "." + name
    if show:
        max = torch.abs(x).max().item()
        min = torch.abs(x).min().item()
        print(f"[GOOD] {name}: dtype={x.dtype}, shape={tuple(x.shape)}, max={max}, min={min}")
        return
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).sum().item()
        max = torch.abs(x).max().item()
        min = torch.abs(x).min().item()
        print(f"[BAD] {name}: nonfinite={bad}, dtype={x.dtype}, shape={tuple(x.shape)}, max={max}, min={min}")
        
        raise RuntimeError(f"nonfinite in {name}")

class MixtureOfBlocksAttention(nn.Module, AttentionLayerBase):
    # Placeholder for MoBA attention implementation
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        use_alibi_sqrt: bool | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        attn_backend: type[AttentionBackend] | None = None,
        is_moba: bool = False,
        moba_topk: int | None = None,
        moba_chunk_size: int | None = None,
        **extra_impl_args,
    ) -> None:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.
        """
        super().__init__()
        
        if per_layer_sliding_window is not None:
            # per-layer sliding window
            sliding_window = per_layer_sliding_window
        elif cache_config is not None:
            # model-level sliding window
            sliding_window = cache_config.sliding_window
        else:
            sliding_window = None
        
        vllm_config = get_current_vllm_config()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            calculate_kv_scales = False
        # llm-compressor mdls need to set cache_dtype to "fp8" manually.
        kv_cache_scheme = getattr(quant_config, "kv_cache_scheme", None)
        if kv_cache_scheme is not None:
            kv_cache_dtype = "fp8"
            calculate_kv_scales = False
            if cache_config is not None:
                cache_config.cache_dtype = "fp8"
                cache_config.calculate_kv_scales = False
        
        # Check if per-head quant scales are required based on kv_cache_scheme
        use_per_head_quant_scales = (
            kv_cache_scheme is not None
            and kv_cache_scheme.get("strategy") == "attn_head"
        )

        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            kv_cache_dtype, vllm_config.model_config
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.calculate_kv_scales = calculate_kv_scales
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})"
        )
        self.layer_name = prefix
        self.quant_config = quant_config
        
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.has_sink = extra_impl_args.get("sinks") is not None
        self.is_moba = is_moba
        self.moba_topk = moba_topk
        self.moba_chunk_size = moba_chunk_size

        # NOTE: model_config may be None during certain tests
        model_config = vllm_config.model_config
        self.use_mm_prefix = model_config is not None and model_config.is_mm_prefix_lm

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        if attn_backend is None:
            self.attn_backend = get_attn_backend(
                head_size,
                dtype,
                kv_cache_dtype,
                block_size,
                use_mla=False,
                has_sink=self.has_sink,
                use_mm_prefix=self.use_mm_prefix,
                attn_type=attn_type,
            )
        else:
            self.attn_backend = attn_backend
        backend_supports_alibi_sqrt = self.attn_backend.supports_alibi_sqrt()
        use_alibi_sqrt = use_alibi_sqrt if use_alibi_sqrt else False
        if use_alibi_sqrt and not backend_supports_alibi_sqrt:
            raise ValueError(
                f"use_alibi_sqrt is not supported by backend "
                f"{self.attn_backend.get_name()}."
            )
        self.use_alibi_sqrt = bool(use_alibi_sqrt)
        if backend_supports_alibi_sqrt:
            extra_impl_args["use_alibi_sqrt"] = self.use_alibi_sqrt
        # prefix caching + batch invariance is currently not supported for
        # FLASHINFER and TRITON_MLA.
        if (
            cache_config is not None
            and cache_config.enable_prefix_caching
            and vllm_is_batch_invariant()
            and (
                self.attn_backend.get_name() == "FLASHINFER"
                or self.attn_backend.get_name() == "TRITON_MLA"
            )
        ):
            logger.warning_once(
                "Disabling prefix caching for FLASHINFER/TRITON_MLA "
                "with batch invariance, as it is not yet supported.",
                scope="local",
            )
            cache_config.enable_prefix_caching = False

        # 收集所有可能的参数
        potential_kwargs = {
            'num_heads': num_heads,
            'head_size': head_size,
            'scale': scale,
            'num_kv_heads': num_kv_heads,
            'alibi_slopes': alibi_slopes,
            'sliding_window': sliding_window,
            'kv_cache_dtype': kv_cache_dtype,
            'logits_soft_cap': logits_soft_cap,
            'attn_type': attn_type,
            'kv_sharing_target_layer_name': kv_sharing_target_layer_name,
            # MoBA-specific
            'is_moba': is_moba,
            'moba_topk': moba_topk,
            'moba_chunk_size': moba_chunk_size,
            **extra_impl_args,
        }

        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            **{
                k: v
                for k, v in potential_kwargs.items()
                if k in inspect.signature(impl_cls.__init__).parameters
            }
        )
        backend_name = self.attn_backend.get_name()
        self.backend = AttentionBackendEnum.__members__.get(backend_name)
        self.dtype = dtype

        # print(f"backend_name: {backend_name}, impl_cls: {impl_cls}, backend_enum: {self.backend}, is_moba: {is_moba}, moba_topk: {moba_topk}, moba_chunk_size: {moba_chunk_size}")

        # For cuda-alike (CUDA and ROCM) and cpu platforms, we control how
        # torch.compile works by registering the attention as one giant
        # opaque custom op. For other platforms, we directly call them
        # and let torch.compile handle them.
        self.use_direct_call = not current_platform.opaque_attention_op()

        self.use_output = self.attn_backend.accept_output_buffer
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        
        self.attn_type = attn_type

        if kv_sharing_target_layer_name is not None:
            validate_kv_sharing_target(
                prefix,
                kv_sharing_target_layer_name,
                compilation_config.static_forward_context,
            )
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # use a placeholder kv cache tensor during init, which will be replaced
        # by bind_kv_cache
        # this variable will not be accessed if use_direct_call is True
        self.kv_cache = [
            torch.tensor([])
            for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
        ]

        # Initialize KV cache quantization attributes
        _init_kv_cache_quant(self, quant_config, prefix)

        # for attn backends supporting query quantization
        self.query_quant = None
        if self.impl.supports_quant_query_input and self.kv_cache_dtype.startswith(
            "fp8"
        ):
            is_per_head = (
                hasattr(self, "q_scale") and self.q_scale.numel() == self.num_kv_heads
            )
            block_size = self.head_size * self.num_heads // self.num_kv_heads
            self.query_quant = QuantFP8(
                static=True,
                group_shape=GroupShape(-1, block_size)
                if is_per_head
                else GroupShape.PER_TENSOR,
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(query, key, value, self.layer_name)
        output_dtype = query.dtype
        if self.query_quant is not None:
            assert self.kv_cache_dtype in {"fp8", "fp8_e4m3"}

            if self.impl.supports_quant_query_input:
                query, _ = self.query_quant(query, self._q_scale)
        
        if self.use_output:
            output_shape = output_shape if output_shape is not None else query.shape
            output = torch.empty(
                output_shape, dtype=output_dtype, device=query.device
            )
            hidden_size = output_shape[-1]

            query = query.view(-1, self.num_heads, self.head_size)
            output = output.view(-1, self.num_heads, self.head_size)
            if key is not None:
                key = key.view(-1, self.num_kv_heads, self.head_size)
            if value is not None:
                value = value.view(-1, self.num_kv_heads, self.head_size)
            
            kv_cache_dummy_dep = None
            if self.use_direct_call:
                # Skip this if sharing KV cache with an earlier attention layer.
                if (
                    not self.attn_backend.forward_includes_kv_cache_update
                    and self.kv_sharing_target_layer_name is None
                    and key is not None
                    and value is not None
                ):
                    kv_cache_dummy_dep = unified_kv_cache_update(
                        key, value, self.layer_name
                    )
                unified_attention_with_output(
                    query,
                    key,
                    value,
                    output,
                    self.layer_name,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
            else:
                # Skip this if sharing KV cache with an earlier attention layer.
                if (
                    not self.attn_backend.forward_includes_kv_cache_update
                    and self.kv_sharing_target_layer_name is None
                    and key is not None
                    and value is not None
                ):
                    kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                        key, value, self.layer_name
                    )
                torch.ops.vllm.unified_attention_with_output(
                    query,
                    key,
                    value,
                    output,
                    self.layer_name,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
            return output.view(-1, hidden_size)
        else:
            assert self.attn_backend.forward_includes_kv_cache_update, (
                "Split KV cache update not supported when output tensor not provided."
            )
            if self.use_direct_call:
                return unified_attention(query, key, value, self.layer_name)
            else:
                return torch.ops.vllm.unified_attention(
                    query, key, value, self.layer_name
                )
    
    def calc_kv_scales(self, query, key, value):
        self._q_scale.copy_(torch.abs(query).max() / self.q_range)
        self._k_scale.copy_(torch.abs(key).max() / self.k_range)
        self._v_scale.copy_(torch.abs(value).max() / self.v_range)
        self._q_scale_float = self._q_scale.item()
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()
        # We only calculate the scales once
        self.calculate_kv_scales = False

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        self.impl.process_weights_after_loading(act_dtype)

        quant_method = (
            self.quant_config.get_quant_method(self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Block size may get updated after model loading, refresh it
        block_size = vllm_config.cache_config.block_size
        # Should not be called for enc-dec or encoder-only attention.
        assert self.attn_type == AttentionType.DECODER
        if self.sliding_window is not None:
            assert not vllm_config.model_config.use_mla, (
                "MLA is not supported for slidingwindow"
            )
            return SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_torch_dtype,
                sliding_window=self.sliding_window,
            )
        else:
            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_torch_dtype,
            )

class MoBA_Attention(nn.Module):

    def __init__(
        self, 
        hidden_size: int = 2048,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        model_config: ModelConfig | None = None,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        head_dim: int = 128,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10000.,
        rope_scaling: dict | float | None = None,
        is_moba: bool = False,
        moba_chunk_size: int = 1024,
        moba_topk: int = 4,
        max_position_embeddings: int | None = None,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.prefix = prefix
        self.layer_idx = layer_idx
        self.quant_config = quant_config
        self.cache_config = cache_config
        self.model_config = model_config
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = divide(self.num_heads, self.num_kv_heads)
        self.head_dim = head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.scaling = self.head_dim**(-0.5)
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.is_moba = is_moba
        self.moba_chunk_size = moba_chunk_size
        self.moba_topk = moba_topk
        self.max_position_embeddings = max_position_embeddings

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        assert self.num_heads % self.tp_size == 0, \
            f"num_heads ({self.num_heads}) must be divisible by tp_size ({self.tp_size})"
        assert self.num_kv_heads % self.tp_size == 0, \
            f"num_kv_heads ({self.num_kv_heads}) must be divisible by tp_size ({self.tp_size})"
        
        self.tp_heads = self.num_heads // self.tp_size
        self.tp_kv_heads = self.num_kv_heads // self.tp_size

        self.q_dim = self.tp_heads * self.head_dim
        self.kv_dim = self.tp_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=self.qkv_bias,
            quant_config=self.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)
        
        # self.rotary = RotaryEmbedding(
        #     head_size=self.head_dim,
        #     rotary_dim=self.head_dim,
        #     max_position_embeddings=self.max_position_embeddings,
        #     base=self.rope_theta,
        #     is_neox_style=True,
        #     dtype=torch.float32,
        # )
        rope_parameters = {"rope_theta": rope_theta}
        if rope_scaling is not None:
            if isinstance(rope_scaling, dict):
                # vLLM get_rope expects rope_type-style dictionaries.
                rope_parameters.update(rope_scaling)
                if "type" in rope_parameters and "rope_type" not in rope_parameters:
                    rope_parameters["rope_type"] = rope_parameters.pop("type")
            else:
                # Backward-compatible path for legacy float scaling configs.
                rope_parameters["factor"] = rope_scaling
        self.rotary = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_parameters,
            dtype=torch.float32,
            dual_chunk_attention_config=None,
        )
        self.attn = MixtureOfBlocksAttention(
            num_heads=self.tp_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.tp_kv_heads,
            cache_config=self.cache_config,
            quant_config=self.quant_config,
            per_layer_sliding_window=self.window_size,
            prefix=f"{prefix}.attn",
            is_moba=is_moba,
            moba_topk=self.moba_topk,
            moba_chunk_size=self.moba_chunk_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        # chk("q", q, self.prefix, show=True)
        if self.qk_norm:
            q = self.q_norm(q.view(-1, self.tp_heads, self.head_dim)).view(
                -1, self.q_dim
            )
            k = self.k_norm(k.view(-1, self.tp_kv_heads, self.head_dim)).view(
                -1, self.kv_dim
            )
        # chk("q_normed", q, self.prefix, show=True)
        q, k = self.rotary(positions, q, k)
        o = self.attn(
            query=q, key=k, value=v, 
        )
        # chk("moba_q", q, self.prefix, show=True)
        # chk("moba_k", k, self.prefix, show=True)
        # chk("moba_v", v, self.prefix, show=True)
        # chk("moba_o", o, self.prefix, show=True)
        output[:], _ = self.o_proj(o)
        # chk("moba_output", output, self.prefix, show=True)

        
