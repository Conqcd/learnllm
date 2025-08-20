# from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

from awq import AutoAWQForCausalLM
# model_dir = snapshot_download('qwen/qwen2-7b-instruct', cache_dir='./qwen/', revision='master')

model_name = "D:\\AI\\learnllm\\homework\\inference\\qwen\\qwen\\Qwen2___5-7B-Instruct"

# 加载模型
quantizer = AutoAWQForCausalLM.from_pretrained(model_name, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 准备校准数据 datasets--mit-han-lab--pile-val-backup   C:\Users\hovie.zeng\Downloads
from datasets import load_dataset
dataset = load_dataset('json', data_files="C:\\Users\\hovie.zeng\\Downloads\\val.jsonl.zst")["train"]
# dataset = load_dataset('mit-han-lab/pile-val-backup', split='validation')

# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calib_data=dataset[:128]["text"]

# AWQ量化配置
quant_config = {
    "zero_point": True,   # 使用零点量化
    "q_group_size": 128,  # 分组大小
    "w_bit": 4,           # 权重量化位数
    "version": "GEMM",     # 使用GEMM推理内核
}

# 执行AWQ量化
quantizer.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

# 保存量化模型
save_path = "./Qwen2___5-7B-Instruct-AWQ-int4"
quantizer.save_quantized(save_path)
tokenizer.save_pretrained(save_path)

print(f"AWQ int4 模型已保存至: {save_path}")