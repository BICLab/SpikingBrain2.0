# SpikingBrain2.0：Brain-Inspired Foundation Models 
**Hybrid Sparse Attention, Dual Quantization Paths, and Multimodal Conversion**

📄 Technical Report: [English]()  
🚀 Arxiv: [arXiv:](https://www.arxiv.org/abs/)  
🧩 Models: [Available Models](#available-models)      

---

## About SpikingBrain2.0

SpikingBrain2.0 is a brain-inspired hybrid model family for long-context language modeling and vision-language modeling. Building on  [SpikingBrain1.0](https://github.com/BICLab/SpikingBrain-7B), it adopts an inter-layer hybrid architecture that combines Sparse Softmax Attention ([MoBA](https://github.com/MoonshotAI/MoBA)) with Sparse Linear Attention ([SSE](https://openreview.net/pdf?id=R6DrJ4tnGV)), forming a sparse-memory design that better balances modeling quality, long-context efficiency, and memory usage while alleviating contextual interference in long sequences. It is further supported by a lightweight Transformer-to-Hybrid (T2H) conversion pipeline, enabling both LLMs and VLMs to be adapted from open-source Qwen3 backbones at very low cost; With fewer than 7k A100 GPU hours, it recovers most of the backbone models’ capabilities and achieves competitive results across general, reasoning, and vision-language benchmarks. 

Beyond capability recovery, SpikingBrain2.0 delivers substantial deployment advantages, including up to 10.13× TTFT speedup at 4M context length, support for 10M+ token serving on 8×A100 GPUs, and dual FP8 and INT8-Spiking quantization paths for both practical GPU acceleration and neuromorphic-friendly execution. Notably, the INT8-Spiking path achieves 64.31% spike-sequence sparsity with minimal accuracy loss, while hardware simulation shows about 46.5%–48.1% power reduction, making SpikingBrain2.0 a lightweight, scalable, and energy-efficient solution for building efficient long-context foundation models.

![](assets/fig1.png)


## Repository Structure

```text
SpikingBrain2.0/
├── spb2/                        # Hugging Face implementation of SpikingBrain2.0 LLM
├── spb2vl/                      # Hugging Face implementation of SpikingBrain2.0-VL
├── spb2_vllm/                   # vLLM inference plugin adapted for both SpikingBrain2.0 LLM and SpikingBrain2.0-VL
├── flash-linear-attention_dev/  # Customized flash-linear-attention with SSE support
├── MoBA/                        # Customized MoBA adapted to the newer FlashAttention interface for Hugging Face
├── run_model/                   # Example scripts for running models with the released checkpoints
└── README.md
```

## Dependency Notes

This repository includes two important local dependency trees.


`flash-linear-attention_dev/` contains a modified version of flash-linear-attention with added SSE support.

In SpikingBrain2.0, [SSE](https://openreview.net/pdf?id=R6DrJ4tnGV) is built as a **Sparse State Expansion** mechanism over [Gated DeltaNet](https://openreview.net/pdf?id=r8H7xhYPwz). By extending the compressed recurrent memory of Gated DeltaNet into multiple sparsely updated state partitions, SSE improves effective memory capacity and long-context retrieval while largely preserving the efficiency benefits of recurrent linear modeling.

---

`MoBA/` contains a customized [MoBA](https://github.com/MoonshotAI/MoBA) implementation whose interfaces were adapted to the newer FlashAttention API used by this repository.This bundled `MoBA/` directory is intended for the **Hugging Face side** of the repository.For the **vLLM side**, `spb2_vllm` does **not** use the bundled `MoBA/`. Instead, it depends on the official **`flash-moba`** package.

Official repository:

- `https://github.com/mit-han-lab/flash-moba`


## Environment Setup

It is recommended to create separate environments for different components if needed.

### Hugging Face LLM (spb2)

#### Setup suggestion

```text
transformers==4.57.1
triton==3.2.0
flash-attn==2.7.3
flash-linear-attention_dev  # use the local version in this repo
MoBA                        # use the local version in this repo
```

### Hugging Face VLM (spb2)

#### Setup suggestion

```text
transformers==4.57.3
flash_attn==2.6.3
flash-linear-attention_dev  # use the local version in this repo
MoBA                        # use the local version in this repo
```

### vLLM inference plugin (spb2_vllm)

Note: **Supports both LLM and VLM inference**

#### Setup suggestion

```text
torch>=2.10.0
transformers>=4.57.0
triton==3.6.0
flash_attn==2.8.3
vllm==0.17.1
setuptools
scipy
flash-linear-attention_dev  # use the local version in this repo
flash_moba==2.0.0           # https://github.com/mit-han-lab/flash-moba
```

## Available Models

Model weights are hosted on **ModelScope**:

- [SpikingBrain-2.0-base-8k](https://www.modelscope.cn/models/Panyuqi/SpikingBrain-2.0-base-8k)
- [SpikingBrain-2.0-base-64k](https://www.modelscope.cn/models/Panyuqi/SpikingBrain-2.0-base-64k)
- [SpikingBrain-2.0-base-256k](https://www.modelscope.cn/models/Panyuqi/SpikingBrain-2.0-base-256k)
- [SpikingBrain-2.0-base-512k](https://www.modelscope.cn/models/Panyuqi/SpikingBrain-2.0-base-512k)
- [SpikingBrain-2.0-instruct](https://www.modelscope.cn/models/Panyuqi/SpikingBrain-2.0-instruct)
- [SpikingBrain-2.0-think](https://www.modelscope.cn/models/Panyuqi/SpikingBrain-2.0-think)
- [SpikingBrain-2.0-VL](https://www.modelscope.cn/models/zhongfangzhi/SpikeBrain-2.0-VL)

### Usage

Example scripts are provided in [`run_model/`](run_model) for running the released checkpoints.

- **Hugging Face**  
  Load the model with `AutoModelForCausalLM` and use it as a standard CausalLM for either forward passes or text generation; see [`run_model/run_model_hf_base.py`](run_model/run_model_hf_base.py).  
  For the SFT model, use the chat template script; see [`run_model/run_model_hf_chat.py`](run_model/run_model_hf_chat.py).  
  For the vision-language model, see [`run_model/run_model_hf_vl.py`](run_model/run_model_hf_vl.py).

- **vLLM**  
  Run inference with the provided **spb2_vllm** plugin; see [`run_model/run_model_vllm.py`](run_model/run_model_vllm.py) and [`run_model/run_model_vllm_vl.py`](run_model/run_model_vllm_vl.py).  
  Before using vLLM, make sure to remove the `auto_map` field from `config.json`. Specifically, delete the following block if it is present:

```json
"auto_map": {
  "AutoConfig": "configuration_sse_swa_moba.SSESWAMoBAConfig",
  "AutoModelForCausalLM": "modeling_sse_swa_moba.SSESWAMoBAForCausalLM"
}
```

You can also launch a vLLM server directly from the terminal:

```bash
vllm serve <your_model_path> \
  --served-model-name <model_name> \
  --max-model-len 524288 \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.6 \
  --tensor-parallel-size 8 \
  --block-size 128 \
  --dtype bfloat16 \
  --trust-remote-code \
  --port 8000 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

### Performance Evaluation

Table 1: **Performance evaluation of the SpikingBrain2.0-5B-base model.** 
![](assets/table1.png)

SpikingBrain2.0-5B is evaluated using the checkpoint after the LongCT-512k stage, with only **14B tokens** of continued training after conversion. Despite the lightweight training budget, it achieves performance comparable to other strong open-source base models, remains close to **Qwen3-4B** overall, and even surpasses the base Transformer on several tasks such as **BBH**.

Table 2: **Performance evaluation of the SpikingBrain2.0-VL-5B model.** 
![](assets/table2.png)

After instruction SFT, SpikingBrain2.0-VL-5B is evaluated on a comprehensive suite of multimodal benchmarks. It delivers competitive performance against strong open-source baselines such as **Qwen2.5-VL-3B** and **LLaVA-OneVision-7B**, while largely recovering the multimodal capability of the base **Qwen3-VL-4B**.

--- 


## Citation

If you find our work useful, please consider citing SpikingBrain2.0:

```bibtex
@article{pan2026spikingbrain2.0,
  title={SpikingBrain2.0},
  author={},
  journal={arXiv preprint arXiv:},
  year={2026}
}

```

