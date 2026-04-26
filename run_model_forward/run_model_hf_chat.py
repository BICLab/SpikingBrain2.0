import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SPB2 instruct/thinking model forward (training) step."
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
        "--prompts",
        type=str,
        nargs="+",
        default=[
            "请用三点简要介绍你自己。",
            "请详细解释一下什么是注意力机制，并给一个直观例子。",
            "你好",
        ],
        help="User prompts (one or more).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to run on, e.g. "auto", "cuda:1", or "cpu".',
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=None,
        help=(
            "Maximum input sequence length after tokenization. "
            "Defaults to model.config.max_position_embeddings - 1024."
        ),
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
    if device.startswith("cuda"):
        cuda_index = int(device.split(":")[1])
        print(
            f"Model loaded on {device} ({torch.cuda.get_device_name(cuda_index)}) with dtype={dtype}."
        )
    else:
        print(f"Model loaded on {device} with dtype={dtype}.")

    # Build a list of prompt texts using the model's built-in chat template.
    prompt_texts = []
    for user_prompt in args.prompts:
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_texts.append(prompt_text)

    print("\n===== Chat Template Prompts =====")
    for i, text in enumerate(prompt_texts):
        print(f"--- Prompt {i} ---")
        print(text)
    print("===== End Prompts =====\n")

    if args.max_input_length is not None:
        max_input_length = args.max_input_length
    else:
        max_input_length = model.config.max_position_embeddings - 1024

    inputs = tokenizer(
        prompt_texts,
        padding=True,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    ).to(device)
    print("input shape:", inputs.input_ids.shape)

    print("\n##### [Testing Model Forward / Training Step] #####")
    model.train()
    input_ids = inputs.input_ids
    outputs = model(
        input_ids,
        labels=input_ids.clone(),
        attention_mask=None,
        use_cache=False,
    )
    print("train loss:", outputs.loss)
    print("output shape:", outputs.logits.shape)
    print("Forward step successful.")


if __name__ == "__main__":
    main()
