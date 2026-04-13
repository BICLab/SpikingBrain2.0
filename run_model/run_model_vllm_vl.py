import argparse
import sys
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Run SpikeBrain-2.0-VL with vLLM.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Local model directory path.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="image.png",
        help="Input image path.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="这张图主要说明了什么自然过程？请按顺序说明水是如何循环的。",
        help="Question about the input image.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generated tokens.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization.",
    )
    parser.add_argument("--max-model-len", type=int, default=32768, help="Max model context length.")
    return parser.parse_args()


def import_plugin():
    try:
        import sse_swa_moba_vllm  # type: ignore
        return sse_swa_moba_vllm
    except ImportError:
        # Fallback for environments where editable install is not activated.
        plugin_root = "SpikingBrain2.0/spb2_vllm"
        if plugin_root not in sys.path:
            sys.path.append(plugin_root)
        import sse_swa_moba_vllm  # type: ignore
        return sse_swa_moba_vllm


def main():
    args = parse_args()

    plugin = import_plugin()
    plugin.register_model()

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.0,
        max_tokens=args.max_tokens,
    )

    # Build multimodal prompt in Qwen-style VL format.
    prompt = (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        f"{args.question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    image = Image.open(args.image_path).convert("RGB")
    request = {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }
    outputs = llm.generate(request, sampling_params)
    print(outputs[0].outputs[0].text.strip())


if __name__ == "__main__":
    main()
