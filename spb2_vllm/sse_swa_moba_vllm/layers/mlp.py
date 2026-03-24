import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.linear import (
    RowParallelLinear,
    MergedColumnParallelLinear,
)

class SseSwaMobaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: int | None = None,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        expert_gate: torch.nn.Linear | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != 'swish':
            raise ValueError(f'Unsupported hidden_act: {hidden_act}; Only swish is supported.')
        
        self.act_fn = SiluAndMul() # SiluAndMul is equivalent to swiglu activation, but fused for better performance.
        self.expert_gate = expert_gate

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)

        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)) * out

        return out