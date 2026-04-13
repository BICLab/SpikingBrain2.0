import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_PATH = "SpikeBrain-2.0-VL"
IMAGE_PATH = "image.png"
QUESTION = "这张图主要说明了什么自然过程？请按顺序说明水是如何循环的。"
DEVICE = "cuda"
MAX_NEW_TOKENS = 256

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer
dtype = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=dtype,
).to(DEVICE).eval()

prompt = (
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>\n"
    f"{QUESTION}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
image = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
print(tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip())
