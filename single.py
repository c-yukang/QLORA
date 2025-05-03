from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

# 强制可见 1 号 GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_path = "results-4.2-0:35-is the best/fine_tuned_model"
#model_path = "/home/ubuntu/.cache/huggingface/hub/deepseek-ai/deepseek-coder-v2-lite-instruct/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    #bnb_4bit_compute_dtype=torch.half,
    #device_map="cuda:0"  # ⚠️ 这里要用 "cuda:0"，因为 `CUDA_VISIBLE_DEVICES=1` 让 `cuda:1` 变成 `cuda:0`
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    #torch_dtype=torch.half,
    low_cpu_mem_usage = True
)

input_text = "Write a piece of code from the airsim library:Get the drone's magnetometer data"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
