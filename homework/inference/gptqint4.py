# from modelscope import snapshot_download
# model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='./qwen/', revision='master')

# recipe = [
#     GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
# ]
# oneshot(
#     model="Qwen/Qwen2-7B-Instruct",
#     dataset="open_platypus",
#     recipe=recipe,
#     output_dir="Qwen/Qwen2-7B-Instruct-INT8",
#     max_seq_length=2048,
#     num_calibration_samples=512,
# )

# from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
# from auto_gptq.utils import is_cuda_extension_available

# 指定 Hugging Face 模型 ID，例如 Llama-2-7B
model_id = "D:\\AI\\learnllm\\homework\\inference\\qwen\\qwen\\Qwen2___5-7B-Instruct"
model_id = "D:\\AI\\learnllm\\homework\\inference\\qwen\\qwen\\qwen2-7b-instruct"

# 配置 GPTQ：量化到 4-bit，使用校准数据集
quantization_config = GPTQConfig(
    bits=4,  # INT4 量化
    dataset="c4",  # 使用 c4 数据集进行校准（或替换为其他如 "wikitext"）
    tokenizer=AutoTokenizer.from_pretrained(model_id)
)

# 加载并量化模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="cuda:0"
)

save_path = "qwen/Qwen2___5-7B-Instruct-GPTQ-INT4"
save_path = "qwen/Qwen2-7B-Instruct-GPTQ-INT4"
# 保存量化模型
model.save_pretrained("save_path")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("save_path")

print("模型已量化到 GPTQ INT4 并保存。")