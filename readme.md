# 文件介绍
dada 文件夹中是训练数据与验证数据
results 文件夹中为训练过程中保存的loss最小的检查点和最终模型
loop_session.py 为多轮对话代码文件
QLORA.py 为训练模型文件
single.py 为单轮对话代码
test_base_model.py 用于对于下载的预训练模型进行测试
tokena.py 为分词

# 使用方法
## 1.虚拟环境
requirements.txt 为部分所需的库(pytorch版本是linux上的)

## 2.下载模型
模型下载地址[hungging face](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct)    
通过[代码](test_base_model.py)可以自动加载模型并测试，注意第四行代码换成自己的本地位置。 

## 3.模型训练
QLORA下为模型训练代码，可根据需要修改必要位置

## 4.运行代码
多轮对话代码文件loop_session.py。直接运行，无需修改(由于设置了相对路径，所以需要在coderV2下运行该代码)。
设置的逻辑为输入要求后模型生成代码，此时需要输入y/n 以判断是否使用代码。若输入n可继续输入指令更改代码，输入y则使用代码并清空记录。
