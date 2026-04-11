from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import lru_cache, partial
from itertools import islice
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature

from .configuration_spb2_vl import (
    SPB2VLVisionConfig, 
    SPB2VLTextConfig, 
    SPB2VLConfig
)
from .modeling_sse_swa_moba import (
    SseSwaMobaModel,
    SseSwaMobaForCausalLM,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

from vllm.distributed import get_pp_group
from vllm.compilation.decorators import support_torch_compile
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config

from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm.model_executor.models.interfaces import (
    IsHybrid,
)
from ..layers.sse_swa_h import SSE_SWA_Hybrid, SSE_GDN_H
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc

from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
)

@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        # the same shape as input_embeds
        "deepstack_input_embeds": 0,
    }
)
class SPB2LLMModel(SseSwaMobaModel):
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        # args for deepstack
        deepstack_input_embeds: IntermediateTensors | None = None,
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

        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                positions=positions,
                # residual=residual,
            )

            if deepstack_input_embeds is not None and layer_idx in range(
                0, len(deepstack_input_embeds)
            ):
                hidden_states = (
                    hidden_states + 
                    deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                )
        
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        
        hidden_state = self.norm(hidden_states)
        return hidden_state


class SPB2VLTextModel(SseSwaMobaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super(SseSwaMobaForCausalLM, self).__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.model = SPB2LLMModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix="lm_head",
                )
        else:
            self.lm_head = PPMissingLayer()
        
        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # HF name -> vLLM name
        name_mapping = {
            "input_layernorm": "attn_norm",
            "post_attention_layernorm": "mlp_norm",
            # === MoBA Attention 内部模块 ===
            "self_attn.k_norm": "attn.k_norm",
            "self_attn.q_norm": "attn.q_norm",
            "self_attn.o_proj": "attn.o_proj",

            # === SSE_GDN_H (sse_attn) 内部模块 ===
            "self_attn.fp32_params.A_log": "attn.sse_attn.A_log",
            "self_attn.fp32_params.dt_bias": "attn.sse_attn.dt_bias",
            "self_attn.lora_q_proj.0": "attn.sse_attn.lora_q_proj_A",
            "self_attn.lora_q_proj.1": "attn.sse_attn.lora_q_proj_B",
            "self_attn.lora_k_proj.0": "attn.sse_attn.lora_k_proj_A",
            "self_attn.lora_k_proj.1": "attn.sse_attn.lora_k_proj_B",
            "self_attn.sse_a_proj": "attn.sse_attn.sse_a_proj",
            "self_attn.sse_b_proj": "attn.sse_attn.sse_b_proj",
            "self_attn.sse_e_proj": "attn.sse_attn.sse_e_proj",
            "self_attn.sse_o_norm": "attn.sse_attn.sse_o_norm",
            "self_attn.sse_o_proj": "attn.sse_attn.sse_o_proj",
            # if use_output_gate is True :
            "self_attn.sse_g_proj.0": "attn.sse_attn.sse_g_proj_A",
            "self_attn.sse_g_proj.1": "attn.sse_attn.sse_g_proj_B",
            
            # === SlidingWindowAttention 内部模块 ===
            "self_attn.swa_o_proj": "attn.swa_attn.swa_o_proj",
            "self_attn.swa_q_norm": "attn.swa_attn.swa_q_norm",
            "self_attn.swa_k_norm": "attn.swa_attn.swa_k_norm",
            
            # 注：sse_merge_norm 和 swa_merge_norm 直接在 SSE_SWA_Hybrid 中
            "self_attn.sse_merge_norm": "attn.sse_merge_norm",
            "self_attn.swa_merge_norm": "attn.swa_merge_norm",
        }
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # MoBA and Full Attention QKV 合并
            ("attn.qkv_proj", "self_attn.q_proj", "q"),
            ("attn.qkv_proj", "self_attn.k_proj", "k"),
            ("attn.qkv_proj", "self_attn.v_proj", "v"),
            # SSE Attention QKV 合并
            ("attn.sse_attn.qkv_proj", "self_attn.sse_q_proj", 0),
            ("attn.sse_attn.qkv_proj", "self_attn.sse_k_proj", 1),
            ("attn.sse_attn.qkv_proj", "self_attn.sse_v_proj", 2),
            # SWA Attention QKV 合并 
            ("attn.swa_attn.qkv_proj", "self_attn.swa_q_proj", "q"),
            ("attn.swa_attn.qkv_proj", "self_attn.swa_k_proj", "k"),
            ("attn.swa_attn.qkv_proj", "self_attn.swa_v_proj", "v"),
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

            if name not in params_dict:
                logger.warning(f"跳过未在 vLLM 文本模型结构中找到的权重: {name}")
                continue
            
            # print(f"加载权重: {name} (源权重名: {name}, shard_id: {shard_id})")
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
                weight_loader(param, loaded_weight)
        
        # --- 步骤 E: 检查是否有模型参数未被加载 ---
        # missing_params = set(params_dict.keys()) - loaded_param_names
        # if missing_params:
        #     logger.warning(f"以下文本模型参数未在加载的权重中找到: {missing_params}, 已加载参数: {loaded_param_names}， 已加载数量: {len(loaded_param_names)}/{len(params_dict)}")
        # else:
        #     logger.info(f"所有文本模型参数均已成功加载.")
        
        return loaded_param_names



@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class SPB2VLForConditionalGeneration(
    Qwen3VLForConditionalGeneration,
    IsHybrid,
):
    
    supports_multimodal_pruning = False
    supports_encoder_tp_data = True

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        "qkv": ["qkv"],  # For vision tower's already-packed QKV
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # super(ClassName, self).__init__() to avoid calling Qwen3VLForConditionalGeneration.__init__, 
        # but still initialize the parent class of Qwen3VLForConditionalGeneration.
        super(Qwen3VLForConditionalGeneration, self).__init__()
        
        config: SPB2VLConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        # self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.is_multimodal_pruning_enabled = False
        
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        
        self.prefix = prefix

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

            # register buffer for deepstack
            if self.use_deepstack:
                self.deepstack_input_embeds = [
                    torch.zeros(
                        vllm_config.scheduler_config.max_num_batched_tokens,
                        config.text_config.hidden_size,
                    )
                    for _ in range(self.deepstack_num_level)
                ]
        
        with self._mark_language_model(vllm_config):
            self.language_model = SPB2VLTextModel(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        if not get_pp_group().is_first_rank and hasattr(
            config.vision_config, "deepstack_visual_indexes"
        ):
            assert self.language_model.start_layer >= len(
                config.vision_config.deepstack_visual_indexes
            ), (
                "start_layer should be greater than or equal to "
                "len(deepstack_visual_indexes)"
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    # ============================= #
    # method for IsHybrid interface #
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype, torch.dtype] | tuple[torch.dtype]:
        return SSE_GDN_H.SSE_GDN_H_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | tuple[tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size

        n_v = (
            hf_config.num_v_heads
            if getattr(hf_config, "num_v_heads", None) is not None
            else hf_config.num_heads
        )
        return SSE_GDN_H.SSE_GDN_H_state_shape(
            tp_size,
            num_heads=hf_config.num_heads,
            num_v_heads=n_v,
            head_k_dim=hf_config.head_dim,
            head_v_dim=int(hf_config.head_dim * hf_config.expand_v),
            use_short_conv=hf_config.use_short_conv,
            conv_kernel_size=hf_config.conv_size,
            sparse_partition=hf_config.num_sparse_partition,
        )
    
    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc] | tuple[MambaStateCopyFunc]:
        return SSE_GDN_H.get_SSE_GDN_H_state_copy_func()

