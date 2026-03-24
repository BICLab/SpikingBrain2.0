from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_spb2_vl import SPB2VLConfig, SPB2VLTextConfig, SPB2VLVisionConfig
from .modeling_spb2_vl import (
    SPB2VLForConditionalGeneration,
    SPB2VLModel,
    SPB2VLPreTrainedModel,
    SPB2VLTextModel,
    SPB2VLVisionModel,
)
from .processing_spb2_vl import SPB2VLProcessor
from lmms_engine.mapping_func import register_model

# Register with AutoClasses
AutoConfig.register(SPB2VLConfig.model_type, SPB2VLConfig, exist_ok=True)
AutoModel.register(SPB2VLConfig, SPB2VLModel, exist_ok=True)
AutoModelForCausalLM.register(SPB2VLConfig, SPB2VLForConditionalGeneration, exist_ok=True)

register_model(
    model_type="spb2_vl",  
    model_config=SPB2VLConfig,  
    model_class=SPB2VLForConditionalGeneration,
    model_general_type="causal_lm",  
)

__all__ = [
    "SPB2VLConfig",
    "SPB2VLTextConfig", 
    "SPB2VLVisionConfig",
    "SPB2VLForConditionalGeneration",
    "SPB2VLModel",
    "SPB2VLPreTrainedModel",
    "SPB2VLTextModel",
    "SPB2VLVisionModel",
    "SPB2VLProcessor",
]


