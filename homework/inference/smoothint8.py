from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
# os.environ['HTTPS_PROXY'] = 'https://127.0.0.1:10809'


model_name = "Qwen/Qwen2-7B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# 如果数据很大，建议只取一部分做校准：例如 validation[:512]
ds = load_dataset('json', data_files="F:\\learnllm\\pile-val-backup\\val.jsonl.zst")#, split="validation[:512]")  # 或全量 split
# 可按工具要求把字段名标准化，比如 "text"
# ds = ds.map(lambda x: {"text": x["text"]}) [tokenizer(text, return_tensors="pt").input_ids for text in ds["train"]["text"][:100]]
ds.save_to_disk("pile-val-backup-512")
print("saved to pile-val-backup-512")