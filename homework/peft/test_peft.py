from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from typing import List, Dict
from tqdm import tqdm
import json
import re
from sklearn.metrics import confusion_matrix

base_model_name = "F:\\data2\\anti_fraud\\models\\modelscope\\hub\\Qwen\\Qwen2-7B-Instruct"
adapter_path = "F:\\learnllm\\homework\peft\\anti-fraud\\Qwen2-7B-Instruct_0711\\checkpoint-6600"

def load_model(model_path, checkpoint_path='', device='cuda'):
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval().to(device)
    # 加载lora权重
    if checkpoint_path:
        model = PeftModel.from_pretrained(model, model_id=checkpoint_path).to(device)

    return model, tokenizer

def build_prompt(content):
    prompt = f"下面是一段对话文本, 请分析对话内容是否有诈骗风险，只以json格式输出你的判断结果(is_fraud: true/false)。\n\n{content}"
    return [{"role": "user", "content": prompt}]

def safe_loads(text, default_value=None):
    json_string = re.sub(r'^```json\n(.*)\n```$', r'\1', text.strip(), flags=re.DOTALL)
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"invalid json: {json_string}")
        return default_value

def predict_batch(model, tokenizer, contents: List[str], device='cuda', debug=False):
    prompts = [build_prompt(content) for content in contents]
    inputs = tokenizer(
        tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False),
        padding=True,
        return_tensors="pt"
    ).to(device)

    default_response = {'is_fraud': False}
    gen_kwargs = {"max_new_tokens": 2048, "do_sample": True, "top_k": 1}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        responses = []
        for i in range(outputs.size(0)):
            output = outputs[i, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(output, skip_special_tokens=True)
            responses.append(safe_loads(response, default_response))
        return responses


def run_test_batch(model, tokenizer, test_data: List[Dict], batch_size: int = 8, device='cuda', debug=False):
    print(f"run in batch mode, batch_size={batch_size}")
    real_labels = []
    pred_labels = []
    pbar = tqdm(total=len(test_data), desc=f'progress')

    for i in range(0, len(test_data), batch_size):
        batch_data = test_data[i:i + batch_size]
        dialog_inputs = [item['input'] for item in batch_data]
        real_batch_labels = [item['label'] for item in batch_data]

        predictions = predict_batch(model, tokenizer, dialog_inputs, device)
        pred_batch_labels = [prediction['is_fraud'] for prediction in predictions]

        real_labels.extend(real_batch_labels)
        pred_labels.extend(pred_batch_labels)

        pbar.update(len(batch_data))

    return real_labels, pred_labels

def load_jsonl(path):

    with open(path, 'r',encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
        return data

def precision_recall(true_labels, pred_labels, labels=None, debug=False):
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    print(f"tn：{tn}, fp:{fp}, fn:{fn}, tp:{tp}") if debug else None
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall
def evaluate_with_model(model, tokenizer, testdata_path, device='cuda', debug=False):
    dataset = load_jsonl(testdata_path)
    run_test_func = run_test_batch
    true_labels, pred_labels = run_test_func(model, tokenizer, dataset, device=device, debug=debug)
    precision, recall = precision_recall(true_labels, pred_labels, debug=debug)
    print(f"precision: {precision}, recall: {recall}")

def evaluate(model_path, checkpoint_path, testdata_path, device='cuda', debug=False):
    model, tokenizer = load_model(model_path, checkpoint_path, device)
    evaluate_with_model(model, tokenizer, testdata_path, device, debug)

if __name__ == "__main__":
    evaluate(model_path=base_model_name,checkpoint_path=adapter_path, testdata_path="F:\\LLaMA-Factory\\data\\test0819.jsonl", device='cuda', debug=True)