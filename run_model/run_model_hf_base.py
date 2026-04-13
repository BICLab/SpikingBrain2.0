import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SPB2 base model without chat template."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Local model directory path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Raw text prompt for base model generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to run on, e.g. "auto", "cuda:1", or "cpu".',
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate.",
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
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        return "cpu"

    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is requested, but CUDA is not available.")
        index = int(device_arg.split(":")[1]) if ":" in device_arg else 0
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Invalid CUDA index {index}. Available device count: {torch.cuda.device_count()}."
            )
        return f"cuda:{index}"

    if device_arg == "cpu":
        return "cpu"

    raise RuntimeError(f'Unsupported device "{device_arg}".')


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="left",
        truncation_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")

    print(f"Loading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    print(f"Model loaded on {device}.")

    # Base model expects plain text prompt, no chat template is used.
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # Decode only newly generated tokens to show model continuation.
    input_len = inputs.input_ids.shape[1]
    new_tokens = generated[0][input_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print("\n===== Prompt =====")
    print(args.prompt)
    print("===== Completion =====")
    print(completion if completion else "[Empty completion]")


if __name__ == "__main__":
    main()
