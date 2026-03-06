from vllm import ModelRegistry
from transformers import AutoConfig

def register_model():
    
    from .configuration_SseSwaMoba import SseSwaMobaConfig
    AutoConfig.register("sse_swa_moba", SseSwaMobaConfig)
    # from .modeling_sse_swa_moba import SseSwaMobaForCausalLM
    # lazy init
    ModelRegistry.register_model(
        "SSESWAMoBAForCausalLM",
        "sse_swa_moba_vllm.sse_swa_moba.modeling_sse_swa_moba:SseSwaMobaForCausalLM",
    )