import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
# from modelscope import snapshot_download,AutoModelForCausalLM, AutoTokenizer

# model_dir = snapshot_download('LLM-Research/Meta-Llama-3.1-8B-Instruct', cache_dir='/root/autodl-tmp', revision='master')
#一定要双右斜杠，单左斜杠不行，不知道为什么
model_dir = 'F:\\root\\autodl-tmp\\LLM-Research\\Meta-Llama-3___1-8B-Instruct'

model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path= model_dir, trust_remote_code=True, device_map='auto',torch_dtype=torch.bfloat16)
model.enable_input_require_grads()
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token为eos_token

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    task_type=TaskType.CAUSAL_LM
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters() # 打印总训练参数

df = pd.read_json("huanhuan.json")
data = Dataset.from_pandas(df)
def preprocess_function(examples):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{examples['instruction'] + examples['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{examples['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
tokens = data.map(preprocess_function, remove_columns=data.column_names)

args = TrainingArguments(learning_rate=1e-4,
                         per_device_train_batch_size=4,
                         output_dir="./output",
                         num_train_epochs= 3,
                         save_steps= 100,
                         save_on_each_node= True,
                         gradient_accumulation_steps= 4,
                         logging_steps= 10,
                         gradient_checkpointing= True,
                         report_to="none"
                         )

trainer = Trainer(
                model=peft_model,
                args=args,
                train_dataset=tokens,
                tokenizer=tokenizer,
                data_collator= DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()