
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.sse.configuration_sse import SSEConfig
from fla.models.sse.modeling_sse import SSEForCausalLM, SSEModel

AutoConfig.register(SSEConfig.model_type, SSEConfig, exist_ok=True)
AutoModel.register(SSEConfig, SSEModel, exist_ok=True)
AutoModelForCausalLM.register(SSEConfig, SSEForCausalLM, exist_ok=True)

__all__ = ['SSEConfig', 'SSEForCausalLM', 'SSEModel']
