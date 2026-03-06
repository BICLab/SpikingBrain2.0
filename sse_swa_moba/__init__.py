from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_sse_swa_moba import SSESWAMoBAConfig
from .modeling_sse_swa_moba import SSESWAMoBAForCausalLM, SSESWAMoBAModel

AutoConfig.register(SSESWAMoBAConfig.model_type, SSESWAMoBAConfig, exist_ok=True)
AutoModel.register(SSESWAMoBAConfig, SSESWAMoBAModel, exist_ok=True)
AutoModelForCausalLM.register(SSESWAMoBAConfig, SSESWAMoBAForCausalLM, exist_ok=True)

__all__ = ['SSESWAMoBAConfig', 'SSESWAMoBAForCausalLM', 'SSESWAMoBAModel']
