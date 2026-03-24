
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from einops import rearrange
from transformers.activations import ACT2FN

# from vllm.attention.backends.abstract import AttentionMetadata
# from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)

# from vllm.forward_context import ForwardContext, get_forward_context

from vllm.logger import init_logger

# from vllm.model_executor.layers.fused_moe import SharedFusedMoE
# from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.layernorm import RMSNorm
# from fla.modules import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

KVCache = tuple[torch.Tensor, torch.Tensor]

from ..layers.mlp import SseSwaMobaMLP
from ..layers.sse_swa_h import SSE_SWA_Hybrid, SSE_GDN_H
from ..layers.moba import MoBA_Attention
from .configuration_SseSwaMoba import SseSwaMobaConfig

def chk(name, x, prefix=""):
    if x is None: return
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).sum().item()
        print(f"[BAD] {name}: nonfinite={bad}, dtype={x.dtype}, shape={tuple(x.shape)}")
    
        raise RuntimeError(f"nonfinite in {name}")

class SseSwaMobaDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        speculative_config = vllm_config.speculative_config
        self.prefix = prefix

        # self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        # RMSNorm(config.hidden_size, eps=config.norm_eps)
        
        if config.attn is not None and self.layer_idx in config.attn['layers']:
            if self.layer_idx in config.attn['full_layers']:
                # Global Attention layers
                self.attn = MoBA_Attention(
                    hidden_size=config.hidden_size,
                    quant_config=quant_config,
                    cache_config=cache_config,
                    model_config=model_config,
                    num_heads=config.attn['num_heads'],
                    num_kv_heads=config.attn['num_kv_heads'],
                    head_dim=config.head_dim,
                    qkv_bias=config.attn['qkv_bias'],
                    qk_norm=config.attn['qk_norm'],
                    window_size=None, 
                    rope_theta=config.attn['rope_theta'],
                    rope_scaling=config.rope_scaling,
                    is_moba=False,
                    max_position_embeddings=config.max_position_embeddings,
                    layer_idx=self.layer_idx,
                    norm_eps=config.norm_eps,
                    prefix=f"{prefix}.attn",
                )
            else:
                # Sparse Attention Layers
                self.attn = MoBA_Attention(
                    hidden_size=config.hidden_size,
                    quant_config=quant_config,
                    cache_config=cache_config,
                    model_config=model_config,
                    num_heads=config.attn['num_heads'],
                    num_kv_heads=config.attn['num_kv_heads'],
                    head_dim=config.head_dim,
                    qkv_bias=config.attn['qkv_bias'],
                    qk_norm=config.attn['qk_norm'],
                    window_size=None,
                    rope_theta=config.attn['rope_theta'],
                    rope_scaling=config.rope_scaling,
                    is_moba=True,
                    moba_chunk_size=config.attn['moba_chunk_size'],
                    moba_topk=config.attn['moba_topk'],
                    max_position_embeddings=config.max_position_embeddings,
                    layer_idx=self.layer_idx,
                    norm_eps=config.norm_eps,
                    prefix=f"{prefix}.attn",
                )
        else:
            if config.linear_attn_type == "gdn":
                self.attn = SSE_SWA_Hybrid(
                    layer_idx=self.layer_idx,
                    hidden_size=config.hidden_size,
                    quant_config=quant_config,
                    cache_config=cache_config,
                    model_config=model_config,
                    expand_v=config.expand_v,
                    head_dim=config.head_dim,
                    num_heads=config.num_heads,
                    num_v_heads=config.num_v_heads,
                    use_output_gate=config.use_output_gate,
                    use_short_conv=config.use_short_conv,
                    conv_size=config.conv_size,
                    num_sparse_partition=config.num_sparse_partition,
                    num_writer=config.num_writer,
                    num_reader=config.num_reader,
                    sse_implementation=config.sse_implementation,
                    sse_qk_relu=config.sse_qk_relu,
                    qkv_bias=config.attn['qkv_bias'],
                    swa_num_kv_heads=config.attn['num_kv_heads'],
                    swa_qk_norm=config.attn['qk_norm'],
                    swa_dropout=config.swa_dropout,
                    window_size=config.attn['window_size'],
                    rope_theta=config.attn['rope_theta'],
                    rope_scaling=config.rope_scaling,
                    max_position_embeddings=config.max_position_embeddings,
                    norm_eps=config.norm_eps,
                    prefix=f"{prefix}.attn",
                )
            elif config.linear_attn_type == "gla":
                raise NotImplementedError("GLA is not implemented yet.")
            else:
                raise ValueError(f"Unsupported linear_attn_type: {config.linear_attn_type}")
        
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)# RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = SseSwaMobaMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor = None,
        positions: torch.Tensor = None,
        **kwargs,
    ):
        residual = hidden_states
        # chk(self.prefix + ".input_hidden_states", hidden_states)
        hidden_states = self.attn_norm(hidden_states)
        # chk(self.prefix + ".attn_norm_out", hidden_states)
        attention_output = torch.empty_like(hidden_states)
        self.attn(
            hidden_states=hidden_states,
            positions=positions,
            output=attention_output,
        )
        # hidden_states = attention_output
        # chk(self.prefix + ".attn_out", hidden_states)
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(attention_output, residual)
        else:
            hidden_states = residual + attention_output
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = hidden_states + residual
        return hidden_states, residual
    
@support_torch_compile
class SseSwaMobaModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config: SseSwaMobaConfig = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        
        self.config = config
        self.vocab_size = config.vocab_size 

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            return SseSwaMobaDecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
            )
        
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(
                config.hidden_size, eps = config.norm_eps
            )
        else:
            self.norm = PPMissingLayer()
    
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None, "Intermediate tensors must be provided for non-first PP ranks."
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                # residual=residual,
                positions=positions,
            )
        
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })
        hidden_states = self.norm(hidden_states)
        return hidden_states
    
    
    
class SseSwaMobaForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsPP,
    IsHybrid,
):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config
        assert not cache_config.enable_prefix_caching, (
            "SseSwaMoba currently does not support prefix caching"
        )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config

        self.model = SseSwaMobaModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
    
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states
    
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return SSE_GDN_H.SSE_GDN_H_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        # num_spec = (
        #     vllm_config.speculative_config.num_speculative_tokens
        #     if vllm_config.speculative_config
        #     else 0
        # )
        return SSE_GDN_H.SSE_GDN_H_state_shape(
            tp_size,
            num_heads=hf_config.num_heads,
            num_v_heads=hf_config.num_heads,
            head_k_dim=hf_config.head_dim,
            head_v_dim=hf_config.head_dim * hf_config.expand_v,
            use_short_conv=hf_config.use_short_conv,
            conv_kernel_size=hf_config.conv_size,
            sparse_partition=hf_config.num_sparse_partition,
        )
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """
        将 HuggingFace 权重映射并加载到当前 vLLM 模型中。
        请将此方法放置在你的主模型类（例如 ModelForCausalLM 或 BaseModel）中。
        """
        
        # 1. HF 命名到 vLLM 模块路径的映射字典
        name_mapping = {
            "embeddings": "embed_tokens",
            # === SSE_GDN_H (sse_attn) 内部模块 ===
            "attn.A_log": "attn.sse_attn.A_log",
            "attn.dt_bias": "attn.sse_attn.dt_bias",
            # "attn.sse_q_proj": "attn.sse_attn.sse_q_proj",
            # "attn.sse_k_proj": "attn.sse_attn.sse_k_proj",
            # "attn.sse_v_proj": "attn.sse_attn.sse_v_proj",
            "attn.lora_q_proj.0": "attn.sse_attn.lora_q_proj_A",
            "attn.lora_q_proj.1": "attn.sse_attn.lora_q_proj_B",
            "attn.lora_k_proj.0": "attn.sse_attn.lora_k_proj_A",
            "attn.lora_k_proj.1": "attn.sse_attn.lora_k_proj_B",
            "attn.sse_a_proj": "attn.sse_attn.sse_a_proj",
            "attn.sse_b_proj": "attn.sse_attn.sse_b_proj",
            "attn.sse_e_proj": "attn.sse_attn.sse_e_proj",
            "attn.sse_o_norm": "attn.sse_attn.sse_o_norm",
            "attn.sse_o_proj": "attn.sse_attn.sse_o_proj",
            # if use_output_gate is True :
            "attn.sse_g_proj.0": "attn.sse_attn.sse_g_proj_A",
            "attn.sse_g_proj.1": "attn.sse_attn.sse_g_proj_B",
            
            # === SlidingWindowAttention 内部模块 ===
            "attn.swa_o_proj": "attn.swa_attn.swa_o_proj",
            "attn.swa_q_norm": "attn.swa_attn.swa_q_norm",
            "attn.swa_k_norm": "attn.swa_attn.swa_k_norm",
            
            # 注：sse_merge_norm 和 swa_merge_norm 直接在 SSE_SWA_Hybrid 中，路径无需映射
        }

        # 2. 需要合并权重的张量映射 (HF 名称 -> vLLM 目标名称, QKV shard_id)
        stacked_params_mapping = [
            # MoBA and Full Attention QKV 合并
            ("attn.qkv_proj", "attn.q_proj", "q"),
            ("attn.qkv_proj", "attn.k_proj", "k"),
            ("attn.qkv_proj", "attn.v_proj", "v"),
            # SSE Attention QKV 合并
            ("attn.sse_attn.sse_qkv_proj", "attn.sse_q_proj", 0),
            ("attn.sse_attn.sse_qkv_proj", "attn.sse_k_proj", 1),
            ("attn.sse_attn.sse_qkv_proj", "attn.sse_v_proj", 2),
            # SWA Attention QKV 合并 
            ("attn.swa_attn.swa_qkv_proj", "attn.swa_q_proj", "q"),
            ("attn.swa_attn.swa_qkv_proj", "attn.swa_k_proj", "k"),
            ("attn.swa_attn.swa_qkv_proj", "attn.swa_v_proj", "v"),
            
            # MLP Gate/Up 合并 
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_param_names = set()

        for name, loaded_weight in weights:
            # --- 步骤 A: 替换路径映射 ---
            for hf_path, vllm_path in name_mapping.items():
                if hf_path in name:
                    name = name.replace(hf_path, vllm_path)
                    break  # 命中即退出循环

            # --- 步骤 B: 拦截需要合并/拆分的 Tensor ---
            shard_id = None
            for target_name, hf_shard_name, shard in stacked_params_mapping:
                if hf_shard_name in name:
                    name = name.replace(hf_shard_name, target_name)
                    shard_id = shard
                    break

            # 如果遇到模型中不需要的参数（例如不需要加载的 rotary_emb.inv_freq），可在这里跳过
            if "rotary_emb.inv_freq" in name:
                continue

            if name not in params_dict:
                # logger.warning(f"跳过未在 vLLM 模型结构中找到的权重: {name}")
                continue

            param = params_dict[name]
            loaded_param_names.add(name)
            
            # --- 步骤 C: 获取当前参数绑定的 weight_loader ---
            # 如果你自定义了 loader (比如 A_log 的 make_channel_major_loader)，
            # 它会优先于 default_weight_loader 被调用。
            weight_loader = getattr(param, "weight_loader", default_weight_loader)

            # --- 步骤 D: 加载权重 ---
            if shard_id is not None:
                # 针对 QKVParallelLinear / MergedColumnParallelLinear 加载指定 shard
                weight_loader(param, loaded_weight, shard_id)
            else:
                # 标准加载
                weight_loader(param, loaded_weight)
        
        # --- 步骤 E: 检查是否有模型参数未被加载 ---
        missing_params = set(params_dict.keys()) - loaded_param_names
        if missing_params:
            logger.warning(f"以下模型参数未在加载的权重中找到: {missing_params}")
        else:
            logger.info(f"所有模型参数均已成功加载.")
