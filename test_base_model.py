from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/home/ubuntu/.cache/huggingface/hub/deepseek-ai/deepseek-coder-v2-lite-instruct/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    force_download=True,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

input_text = "Write a piece of code from the airsim library:Change drone's velocity while moving forward"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))