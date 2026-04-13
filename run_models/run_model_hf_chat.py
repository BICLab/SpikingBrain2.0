import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SPB2 instruct/thinking model."
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
        help="System prompt.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="请用三点简要介绍你自己。",
        help="User prompt.",
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
        if ":" in device_arg:
            index = int(device_arg.split(":")[1])
        else:
            index = 0
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
    model_path = args.model_path

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
        truncation_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")

    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    if device.startswith("cuda"):
        cuda_index = int(device.split(":")[1])
        print(
            f"Model loaded on {device} ({torch.cuda.get_device_name(cuda_index)}) with dtype={dtype}."
        )
    else:
        print(f"Model loaded on {device} with dtype={dtype}.")

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]

    # Build a single text prompt using the model's built-in chat template.
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("\n===== Chat Template Prompt =====")
    print(prompt_text)
    print("===== End Prompt =====\n")

    # Tokenize prompt text and move tensors to model device.
    inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

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

    # Keep only newly generated tokens (excluding the prompt tokens).
    input_len = inputs.input_ids.shape[1]
    new_tokens = generated[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print("===== Model Response =====")
    print(response if response else "[Empty response]")
    print("===== End Response =====")


if __name__ == "__main__":
    main()
