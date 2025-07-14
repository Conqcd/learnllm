from modelscope import snapshot_download
import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import IterableDataset
from typing import Optional, Dict, Iterable
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset


SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """
<think>
{think}
</think>
<answer>
{answer}
</answer>
"""



def extract_answer(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def extract_cot(text: str) -> str:
    if "####" not in text:
        return ""
    cot = text.split("####")
    return XML_COT_FORMAT.format(think=cot[0].strip(), answer=cot[1].strip())

def get_gsm8k_dataset(split="train", sft=False, cache_dir=None, first_half=False, second_half=False) -> IterableDataset:
    data = load_dataset("openai/gsm8k", "main", split=split)

    if first_half:
        data = data.shard(2, 0)
    elif second_half:
        data = data.shard(2, 1)

    if not sft:
        data = data.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': extract_answer(x['answer'])
        })
    else:
        data = data.map(lambda x: {
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']},
                {'role': 'assistant', 'content': extract_cot(x['answer'])},
            ]
        })
    return data

if __name__ == "__main__":

    model_id = "qwen/Qwen2.5-1.5B-Instruct"
    model_dir = "./cache/qwen/Qwen2___5-1___5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=None).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

    config = SFTConfig(output_dir="./output/Qwen-1.5B-SFT-FirstHalf",
                       per_device_train_batch_size=4,
                       num_train_epochs=5,
                       max_seq_length=1024,
                       dataset_batch_size=8,
                       gradient_accumulation_steps=4,
                        save_on_each_node=True,
                       dataset_num_proc=4,
                        logging_steps=10,
                        save_steps=100,
                        learning_rate=1e-4,

                       report_to=[])

    trainer = SFTTrainer(train_dataset=get_gsm8k_dataset(sft=True, first_half = True, second_half= False),
                         model=model,
                         tokenizer=tokenizer,
                         args=config)

    trainer.train()