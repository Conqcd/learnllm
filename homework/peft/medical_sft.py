import pandas as pd
import torch
import json
from datasets import Dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType, get_peft_model,PeftModel
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq,AutoTokenizer,TrainerCallback
import numpy as np
import re
from collections import defaultdict

def process_func(example):
    """
    将数据集进行预处理
    """

    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    system_prompt = """你是一个中医文本实体识别领域的专家，你需要从给定的句子中提取 中医治则; 中医治疗; 中医证候; 中医诊断; 中药; 临床表现; 其他治疗; 方剂; 西医治疗; 西医诊断. 以 json 格式输出, 如 {"entity_text": "首乌", "entity_label": "中药"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体". """,

    instruction = tokenizer(
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
            instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def extract_all_and_swap(text: str) -> dict[str, list[str]]:
    """
    从输入字符串中提取所有符合 {"text":"...","type":"..."} 的字典，
    并把它们转换为 {type: [text1, text2, ...]} 的映射。
    返回一个 dict，value 用列表收集所有同类型的 text。
    """
    pattern = re.compile(
        r"\{\s*'entity_label'\s*:\s*'(?P<label>[^']+)'\s*,\s*"
        r"'entity_text'\s*:\s*'(?P<ent>[^']+)'\s*\}"
    )
    result = defaultdict(list)
    for m in pattern.finditer(text):
        t = m.group("label")
        txt = m.group("ent")
        result[t].append(txt)
    return dict(result)


def dict_p_r_f1(pred_list: list[str], goal_list: list[str]) -> dict[str, float]:
    """
    计算两个字典的 Precision、Recall 和 F1。
    以 (key, value) 对作为“样本”，只要完全匹配才算命中。

    Args:
        pred_dict: 预测的 list，键和值均为 str。
        goal_dict: 真实的 list，键和值均为 str。

    Returns:
        {
            'precision': float,
            'recall': float,
            'f1': float
        }
    """
    # 将字典项转换为 (key, value) 的集合
    pred_items = set(pred_list)
    goal_items = set(goal_list)

    # 计算 True Positives（TP）：预测-真实交集
    tp = len(pred_items & goal_items)
    # 预测总数、真实总数
    pred_count = len(pred_items)
    goal_count = len(goal_items)

    precision = tp / pred_count if pred_count > 0 else 0.0
    recall    = tp / goal_count if goal_count > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1}

# metric = evaluate.load("seqeval")

f1_dict = {'中医治则':[], '中医治疗':[], '中医证候':[], '中医诊断':[], '中药':[], '临床表现':[], '其他治疗':[], '方剂':[], '西医治疗':[], '西医诊断':[]}
f1_total_dict = {'中医治则':[], '中医治疗':[], '中医证候':[], '中医诊断':[], '中药':[], '临床表现':[], '其他治疗':[], '方剂':[], '西医治疗':[], '西医诊断':[]}

#计算F1
def compute_metrics(p,compute_result):

    predictions, labels = p
    predictions = predictions.type(dtype=torch.float32).cpu().numpy()
    labels = labels.type(dtype=torch.float32).cpu().numpy()
    predictions = np.argmax(predictions, axis=2)
    true_labels = []
    true_predictions = []
    for pred_seq, label_seq in zip(predictions, labels):
        temp_true = []
        temp_pred = []
        for pred_label, true_label in zip(pred_seq, label_seq):
            if true_label != -100:
                temp_true.append(true_label)
                temp_pred.append(pred_label)
        temp_true = tokenizer.decode(
            temp_true,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        true_dic = extract_all_and_swap(temp_true)
        temp_pred = tokenizer.decode(
            temp_pred,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        pred_dic = extract_all_and_swap(temp_pred)
        for key in f1_dict.keys():
            if key in true_dic.keys() and key in pred_dic.keys():
                score = dict_p_r_f1(pred_dic[key],true_dic[key])
                f1_dict[key].append(score)

    if compute_result:

        print(f1_dict)
        total_f1 = 0
        total_precision = 0
        total_recall = 0
        count_f1 = 0
        for key, values in f1_dict.items():
            for value in values:
                total_f1 += value['f1']
                total_precision += value['precision']
                total_recall += value['recall']
                count_f1 += 1
            key_f1 = sum([v['f1'] for v in values]) / len(values) if len(values) > 0 else 0
            f1_total_dict[key].append(key_f1)
            values.clear()

        return {
            'precision': total_precision/count_f1,
            'recall': total_recall/count_f1,
            'f1': total_f1/count_f1
        }
        # true_labels.append(dic_true)
        # true_predictions.append(dic_pre)
    # true_predictions = tokenizer.batch_decode(
    #     true_predictions,
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=True
    # )
    # true_predictions = [list(pre_string) for pre_string in true_predictions]
    # true_labels = tokenizer.batch_decode(
    #     true_labels,
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=True
    # )
    # true_labels = [list(label_string) for label_string in true_labels]
    # results = metric.compute(predictions=[true_predictions], references=[true_labels])
    # 返回 dict, 包含 'overall_f1', 'overall_precision', 'overall_recall', 以及各标签性能

    # return {
    #     'precision': results['overall_precision'],
    #     'recall': results['overall_recall'],
    #     'f1': results['overall_f1']
    # }
    return {
            'precision': 0,
            'recall': 0,
            'f1': 0
        }

class F1PrintCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            f1 = metrics.get("eval_f1", None)
            precision = metrics.get("eval_precision", None)
            recall = metrics.get("eval_recall", None)

            if f1 is not None:
                print(f"\n⭐ 评估结果 - Epoch {state.epoch}/{args.num_train_epochs}")
                print(f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
                print("-" * 50)


def collate_fn(batch):
    inputs = []
    answers = []
    for b in batch:
        inputs.append(b["input_ids"])
        answers.append(b["labels"])

    batch_data = tokenizer(inputs, truncation=True, padding=True, max_length=1280, return_tensors="pt")

    with tokenizer.as_target_tokenizer():
        answer_token = tokenizer(answers, truncation=True, padding=True, max_length=1280, return_tensors="pt").input_ids

        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(answer_token)
        eos_token_id = torch.where(answer_token == tokenizer.eos_token_id)[1]
        for idx, eos_id in enumerate(eos_token_id):
            answer_token[idx][eos_id + 1:] = -100  # Mask out the tokens after the EOS token
        batch_data['labels'] = answer_token

    return batch_data

def test_loop_final(dataloader, tokenizer, model, device, save_path):
    labels = []
    predictions = []
    sources = []
    model.eval()
    for batch, data in enumerate(dataloader):
        data = data.to(device)
        with torch.no_grad():
            output = model.generate(data["input_ids"],
                                    attention_mask=data["attention_mask"],
                                    max_length=1280,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    )
        if isinstance(output, tuple):
            output = output[0]

        decoded_sources = tokenizer.batch_decode(
            data["input_ids"].cpu().numpy(),
            skip_special_tokens=True,
            use_source_tokenizer=True
        )

        sources += [source.strip() for source in decoded_sources]

        decoded_preds = tokenizer.batch_decode(output, skip_special_tokens=True)
        predictions += [' '.join(pred.strip()) for pred in decoded_preds]

        label_token = data["labels"].cpu().numpy()
        label_token = np.where(label_token == -100, tokenizer.pad_token_id, label_token)
        decoded_label = tokenizer.batch_decode(label_token, skip_special_tokens=True)

        labels += [' '.join(label.strip()) for label in decoded_label]
        print(f"batch: {batch}")

    results = []
    for source, pred, label in zip(sources, predictions, labels):
        results.append({
            "context": source,
            "prediction": pred,
            "labels": label
        })
    with open(save_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


model_id = "qwen/Qwen2.5-7B-Instruct"
model_dir = "F:\\learnllm\\homework\\peft\\qwen\\Qwen2___5-7B-Instruct"
trained_model_dir = "F:\\learnllm\\homework\\peft\\output\\Qwen2.5-MedicalNER\\checkpoint-690"

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

train_jsonl_new_path = "datasets/medical_train.jsonl"

total_df = pd.read_json(train_jsonl_new_path, lines=True)
train_df = total_df[int(len(total_df) * 0.1):]
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)


dev_jsonl_new_path = "datasets/medical_dev.jsonl"

total_df = pd.read_json(dev_jsonl_new_path, lines=True)
train_df = total_df[int(len(total_df) * 0.1):]
train_ds = Dataset.from_pandas(train_df)
eval_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

test_jsonl_new_path = "datasets/medical_test.jsonl"

total_df = pd.read_json(test_jsonl_new_path, lines=True)
train_df = total_df[int(len(total_df) * 0.1):]
train_ds = Dataset.from_pandas(train_df)
test_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model=PeftModel.from_pretrained(model,trained_model_dir)
# model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="./output/Qwen2.5-MedicalNER",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=2,
    eval_strategy="steps",
    eval_steps=30,
    logging_steps=10,
    num_train_epochs=3,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    save_strategy="best",
    load_best_model_at_end=True,
    batch_eval_metrics=True,
    metric_for_best_model="f1",  # 使用F1选择最佳模型
    greater_is_better=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[F1PrintCallback()],  # 添加自定义回调
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
results = test_loop_final(test_dataset,tokenizer,model,"cuda","output/Qwen2.5-MedicalNER/test_predictions.json")
# results = trainer.evaluate()
# trainer.train()