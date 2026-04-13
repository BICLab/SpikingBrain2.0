import argparse
from pathlib import Path

from vllm import LLM, SamplingParams

import sse_swa_moba_vllm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SPB2 model with vLLM."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Local model directory path.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt used for chat mode.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="请详细解释一下什么是注意力机制，并给一个直观例子。",
        help="User prompt text.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of output tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization ratio for vLLM.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum context length for vLLM engine.",
    )
    parser.add_argument(
        "--disable-chat",
        action="store_true",
        help="Use raw generate API instead of chat API.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Register custom SPB2 model + attention backend for vLLM.
    sse_swa_moba_vllm.register_model()

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

    if args.disable_chat:
        outputs = llm.generate(args.prompt, sampling_params)
    else:
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ]
        outputs = llm.chat(messages, sampling_params)

    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
