from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re
import os



model_path = "results/fine_tuned_model"
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
# 初始化对话历史
conversation_history = []
conversation_history.append("We will discuss how to write code and have conversations in the AirSim library. Please focus on this topic.")

def get_response(user_input):
    # 将用户输入添加到对话历史
    conversation_history.append(f"User: {user_input}")
    
    # 拼接对话历史作为模型输入
    input_text = "\n".join(conversation_history) + "\nAI:"
    
    # 编码输入
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
  # 生成模型输出  
    outputs = model.generate(**inputs, max_length=1024,temperature=0.2)
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取AI的回复
    ai_response = response[len(input_text):].strip()

    # 将AI回复添加到对话历史
    conversation_history.append(f"AI: {ai_response}")

    return ai_response


# 进入循环，接收用户输入直到输入 "exit(0)"
print("输入 'exit(0)' 结束对话。")
while True:
    user_input = input("用户: ")  # 接收用户输入
    if user_input.lower() == "exit(0)":
        print("对话结束。")
        break  # 如果输入 'exit(0)'，则退出循环
    response = get_response(user_input)

    print(f"AI: {response}")
    
    # 提示用户是否使用代码
    user_choice = input("是否使用代码？(y/n): ").strip().lower()
    
    if user_choice == "y":
        # 提取 [code] 标签中的内容
        matches = re.findall(r'\[code\](.*?)\[code\]', response)

        if matches:
            ai_response = "\n".join(matches)
        else:
            ai_response = ai_response.strip()
        
        print("代码获取")
        print(ai_response)

        # 清空对话历史
        conversation_history.clear()
        conversation_history.append("We will discuss how to write code and have conversations in the AirSim library. Please focus on this topic.")
    elif user_choice == "n":
        # 如果选择不使用代码，不做任何操作
        continue
    else:
        print("无效输入，请输入 'y' 或 'n'.")