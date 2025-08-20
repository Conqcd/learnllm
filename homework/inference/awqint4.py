from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

from awq import AutoAWQForCausalLM
# model_dir = snapshot_download('qwen/qwen2-7b-instruct', cache_dir='./qwen/', revision='master')

model_name = "F:\\learnllm\\homework\\peft\\qwen\\Qwen2___5-7B-Instruct"  # 或其他 GPTQ 模型
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 原始模型路径
model_name = "Qwen/Qwen2-7B-Instruct"

# 加载模型
quantizer = AutoAWQForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 准备校准数据
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calib_data = [tokenizer(text, return_tensors="pt").input_ids for text in dataset["text"][:100]]

# AWQ量化配置
quant_config = {
    "zero_point": True,   # 使用零点量化
    "q_group_size": 128,  # 分组大小
    "w_bit": 4,           # 权重量化位数
    "version": "GEMM"     # 使用GEMM推理内核
}

# 执行AWQ量化
quantizer.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

# 保存量化模型
save_path = "./Qwen2-7B-Instruct-AWQ-int4"
quantizer.save_quantized(save_path)
tokenizer.save_pretrained(save_path)

print(f"AWQ int4 模型已保存至: {save_path}")