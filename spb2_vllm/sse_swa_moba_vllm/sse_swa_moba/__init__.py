from vllm import ModelRegistry
from transformers import AutoConfig

def register_model():
    
    from .configuration_SseSwaMoba import SseSwaMobaConfig
    from .configuration_spb2_vl import SPB2VLConfig
    AutoConfig.register("sse_swa_moba", SseSwaMobaConfig)
    AutoConfig.register("spb2_vl", SPB2VLConfig)
    # from .modeling_sse_swa_moba import SseSwaMobaForCausalLM
    # lazy init
    ModelRegistry.register_model(
        "SSESWAMoBAForCausalLM",
        "sse_swa_moba_vllm.sse_swa_moba.modeling_sse_swa_moba:SseSwaMobaForCausalLM",
    )
    ModelRegistry.register_model(
        "SPB2VLForConditionalGeneration",
        "sse_swa_moba_vllm.sse_swa_moba.spb2_vl:SPB2VLForConditionalGeneration",
    )