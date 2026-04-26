import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_PATH = "SpikeBrain-2.0-VL"
IMAGE_PATH = "image.png"
QUESTION = "这张图主要说明了什么自然过程？请按顺序说明水是如何循环的。"
DEVICE = "cuda"

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer
dtype = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=dtype,
).to(DEVICE)

print(model.config)
print(model)

prompt = (
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>\n"
    f"{QUESTION}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
image = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to(DEVICE)
print("input shape:", inputs["input_ids"].shape)

print("\n##### [Testing Model Forward / Training Step] #####")
model.train()
input_ids = inputs["input_ids"]
forward_kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
forward_kwargs["attention_mask"] = None
outputs = model(
    input_ids,
    labels=input_ids.clone(),
    use_cache=False,
    **forward_kwargs,
)
print("train loss:", outputs.loss)
print("output shape:", outputs.logits.shape)
print("Forward step successful.")
