from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model
import os
from datasets import Dataset, DatasetDict
import json

# 强制可见 1 号 GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 设置基本路径
model_path = "/home/ubuntu/.cache/huggingface/hub/deepseek-ai/deepseek-coder-v2-lite-instruct/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
train_data_path = "data/train_data.json"
val_data_path = "data/val_data.json"


#解析标准的json格式
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)  # 用 `json.load()` 解析整个 JSON 文件
    return data

# 加载 tokenizer（仅加载一次，使用 model_path）
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

# 配置 LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 使用 qLORA 微调模型
model = get_peft_model(model, lora_config)

###################开启梯度检查点
model.enable_input_require_grads()

train_data = load_data(train_data_path)
val_data = load_data(val_data_path)

# 将数据转换为 Hugging Face Dataset 格式
train_dataset = Dataset.from_dict({
    "nl": [item["nl"] for item in train_data],
    "code": [item["code"] for item in train_data]
})

val_dataset = Dataset.from_dict({
    "nl": [item["nl"] for item in val_data],
    "code": [item["code"] for item in val_data]
})

# 创建一个 DatasetDict，方便管理训练和验证数据
dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})


# 定义数据预处理函数
def preprocess_function(examples):
    prompt = "We will discuss how to write code and have conversations in the AirSim library. Please focus on this topic."
    
    inputs = [prompt +"\nUser: " + nl + "\nAI:"  for nl in examples["nl"]]
    
    # 对目标进行处理，将其格式化为 "[code] 实际code [code]" 形式，并加上结束符
    targets = [" [code]" + code + "[code]" + tokenizer.eos_token for code in examples["code"]]

    # 联合编码
    tokenized = tokenizer(
        text=[i + t for i, t in zip(inputs, targets)],
        padding="max_length",  
        max_length=512,
        return_tensors="pt",
        add_special_tokens=False  # 必须禁用自动加特殊符号
    )

    labels = tokenized.input_ids.clone()
    
    # 精准计算输入部分真实长度
    for i in range(len(inputs)):
        # 获取原始输入的编码序列（与联合编码使用相同设置）
        input_only = tokenizer.encode(
            inputs[i], 
            add_special_tokens=False,
            truncation=False
        )
        
        # 在联合编码结果中定位输入序列
        combined = tokenized.input_ids[i].tolist()
        input_len = len(input_only)
        
        # 安全校验（确保输入部分编码一致性）
        if combined[:input_len] == input_only:
            # 正确掩码输入部分
            labels[i, :input_len] = -100
        else:
            # 触发警告机制（实际部署时应处理异常样本）
            print(f"编码不一致于样本{i}，启用备选方案")
            # 使用动态查找策略
            overlap = 3  # 允许3个token的定位容差
            search_space = combined[:len(combined)//2]
            for pos in range(len(search_space)-overlap):
                if search_space[pos:pos+overlap] == input_only[:overlap]:
                    labels[i, :pos+overlap] = -100
                    break

        # 统一处理填充区域
        labels[i, (tokenized.input_ids[i] == tokenizer.pad_token_id)] = -100

    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "labels": labels
    }


# 应用预处理函数到训练和验证数据集
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)


# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",             # 输出路径
    eval_strategy="epoch",          # 每个 epoch 后评估一次
    learning_rate=5e-5,
    per_device_train_batch_size=2,        # 每个设备的训练批次大小
    per_device_eval_batch_size=8,        # 每个设备的评估批次大小
    num_train_epochs=5,                   # 训练轮数
    weight_decay=0.01,
    logging_dir="./logs",                 # 日志路径
    logging_steps=500,
    save_steps=500,                       # 每 500 步保存一次模型检查点
    save_total_limit=3,                   # 保留最新的 3 个检查点
    save_strategy="epoch",                # 每个 epoch 结束后保存模型
    #fp16=True,                          # 使用 16 位精度
    fp16=False,  # 关闭 fp16
    bf16=True,  # 开启 bf16 计算，更稳定
    load_best_model_at_end=True,          # 训练结束时加载最佳模型
    metric_for_best_model="eval_loss",     # 如果评估时有 accuracy 指标可以设置这里
    greater_is_better=False,  # ✅ 选最小 loss
    optim="paged_adamw_32bit",            # 启用双叶优化
)

# 创建 Trainer
trainer = Trainer(
    model=model,               # 使用微调后的模型
    args=training_args,
    train_dataset=train_dataset,  # 训练数据集
    eval_dataset=val_dataset,     # 验证数据集
    processing_class=tokenizer,
    #tokenizer=tokenizer,          # 使用 tokenizer
)

# 开始训练
trainer.train()

# 保存微调后的模型和配置
output_model_dir = "results/fine_tuned_model"
model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
