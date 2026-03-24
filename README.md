# SpikingBrain2.0

---

## About SpikingBrain2.0

Building on [SpikingBrain1.0](https://github.com/BICLab/SpikingBrain-7B), **SpikingBrain2.0** marks our next step toward brain-inspired foundation models for long-context intelligence, comprising [SpikingBrain2.0-5B](https://www.modelscope.cn/profile/Panyuqi) for language modeling and [SpikingBrain2.0-VL-5B](https://www.modelscope.cn/models/zhongfangzhi/SpikeBrain-2.0-VL) for vision-language modeling. It adopts an inter-layer hybrid architecture that combines Sparse Softmax Attention ([MoBA](https://github.com/MoonshotAI/MoBA)) with Sparse Linear Attention ([SSE](https://openreview.net/pdf?id=R6DrJ4tnGV)), achieving a stronger balance between modeling capability and computational efficiency while alleviating contextual memory interference in long sequences. To support efficient architecture migration, **SpikingBrain2.0** is further built upon a lightweight Transformer-to-Hybrid conversion pipeline, enabling both LLMs and VLMs to be adapted from open-source Transformer backbones at very low cost. With fewer than 7k A100 GPU hours, it recovers most of the backbone model’s capabilities and delivers competitive performance across general, reasoning, and multimodal benchmarks.

---

## Available Models 🧩
The model weights are hosted on **ModelScope**. Please select the appropriate version based on your needs:

- **SpikingBrain-2.0-base-8k :** https://www.modelscope.cn/models/Panyuqi/SpikingBrain-2.0-base-8k
- **SpikingBrain-2.0-base-64k :** https://www.modelscope.cn/models/Panyuqi/SpikingBrain-2.0-base-64k
- **SpikingBrain-2.0-base-256k :** https://www.modelscope.cn/models/Panyuqi/SpikingBrain-2.0-base-256k
- **SpikingBrain-2.0-base-512k :** https://www.modelscope.cn/models/Panyuqi/SpikingBrain-2.0-base-512k
- **SpikingBrain-2.0-instruct :** https://www.modelscope.cn/models/Panyuqi/SpikingBrain-2.0-instruct
- **SpikingBrain-2.0-think :** https://www.modelscope.cn/models/Panyuqi/SpikingBrain-2.0-think
- **SpikingBrain-2.0-VL :** https://www.modelscope.cn/models/zhongfangzhi/SpikeBrain-2.0-VL

---

## Project Structure
This repository provides the full implementation **SpikingBrain2.0**, including the **HuggingFace version LLM**, **vLLM inference version LLM**, and the implementation **SpikingBrain2.0-VL**, enabling flexible deployment and research across different scenarios.

```
SpikingBrain2.0/
├── spb2/ # Hugging Face implementation of SpB2.0 LLM, with configuration files for each stage of training.
├── spb2_vllm/ # vLLM plugin for SpB2.0 inference
├── spb2vl/ # Hugging Face implementation of SpB2.0-VL
├── flash-linear-attention_dev # fla with SSE implementation
├── MoBA # MoBA adapted to the new FlashAttention interface
└── README.md 
```

--- 

## Dependency Notes

This repository includes two important local dependency trees:

### `flash-linear-attention_dev`

This directory contains a **modified version of flash-linear-attention with added SSE support**. In **SpikingBrain2.0**, the [SSE](https://openreview.net/pdf?id=R6DrJ4tnGV) model is built as a **S**parse **S**tate **E**xpansion over [Gated DeltaNet](https://openreview.net/pdf?id=r8H7xhYPwz). By extending the compressed recurrent memory of GDN into multiple sparsely updated state partitions, **SSE** increases effective memory capacity and enhances long-context retrieval, while largely preserving the efficiency benefits of linear recurrent modeling.

### `MoBA`

This directory contains a **MoBA implementation whose interfaces were adapted to the newer FlashAttention API**.

- The bundled **`MoBA/` directory mainly serves the Hugging Face side**, including both **`spb2` (LLM)** and **`spb2vl` (VLM)**.
- The **vLLM side does not use this bundled `MoBA/` implementation**. Instead, **`spb2_vllm` uses the official `flash-moba` package**.

For `spb2_vllm`, the runtime environment additionally requires:

- **`flash_moba==2.0.0`**
- official repository: `https://github.com/mit-han-lab/flash-moba`

---

## `spb2` (Hugging Face LLM)

### Environment Setup

`spb2` requires the following core versions:


```bash
# create and activate your environment first,

pip install transformers==4.57.1
pip install triton==3.2.0
pip install flash-attn==2.7.3

# flash-linear-attention should be installed from the bundled `flash-linear-attention_dev/`
cd flash-linear-attention_dev
pip install -e .
cd ..

# MoBA should be installed from the bundled `MoBA/`
cd MoBA
pip install -e .
cd ..
```

---

## `spb2vl` (Hugging Face VLM)

### Environment Setup

`spb2vl` requires the following core versions:


```bash
# create and activate your environment first

pip install transformers==4.57.3
pip install flash_attn==2.6.3

# flash-linear-attention should be installed from the bundled `flash-linear-attention_dev/`
cd flash-linear-attention_dev
pip install -e .
cd ..

# MoBA should be installed from the bundled `MoBA/`
cd MoBA
pip install -e .
cd ..
```

---

## `spb2_vllm` (vLLM plugin)

### Environment

`spb2_vllm` requires the following core versions:


```bash
# create and activate your environment first

pip install "torch>=2.9.0"
pip install "transformers>=4.57.0"
pip install triton==3.5.0
pip install flash_attn==2.8.3
pip install vllm==0.13.0
pip install setuptools scipy

#  details in https://github.com/mit-han-lab/flash-moba
git clone https://github.com/mit-han-lab/flash-moba
cd flash-moba
MAX_JOBS=32 python setup.py install
cd ..

# flash-linear-attention should be installed from the bundled `flash-linear-attention_dev/`
cd flash-linear-attention_dev
pip install -e .
cd ..

cd spb2_vllm
pip install -e .
cd ..
```

### Usage

After installing the plugin, you can launch SpB2.0 with vLLM. A typical command is:


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

Please make sure to remove the `auto_map` field from `config.json`. Specifically, delete the following block if it is present:

```json
"auto_map": {
  "AutoConfig": "configuration_sse_swa_moba.SSESWAMoBAConfig",
  "AutoModelForCausalLM": "modeling_sse_swa_moba.SSESWAMoBAForCausalLM"
}
```

---

