import json
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, classification_report

# 加载模型
model_name = "/root/autodl-tmp/code_evaluate/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 从prompt中提取stance（如"against"或"favor"）
def extract_stance_from_prompt(prompt):
    match = re.search(r"aligns with the stance: (\w+)", prompt)
    if match:
        return match.group(1).lower()  # 返回小写（against/favor）
    return None

# 情感分析函数
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = softmax(outputs.logits.numpy()[0])
    return {
        "negative": float(scores[0]),
        "neutral": float(scores[1]),
        "positive": float(scores[2])
    }

# 处理JSON文件并计算准确率
def evaluate_stance_accuracy(input_path, output_path):
    true_labels = []
    pred_labels = []
    
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            text = data["llama3"]
            prompt = data["prompt"]
            
            # 从prompt中提取target_stance
            target_stance = extract_stance_from_prompt(prompt)
            if not target_stance:
                continue  # 跳过无法解析的数据
            
            # 情感分析
            sentiment = analyze_sentiment(text)
            
            # 映射真实标签（against→negative, favor→positive）
            true_label = "negative" if target_stance == "against" else "positive"
            
            # 模型预测标签（选择概率最高的情感，忽略neutral）
            pred_label = "negative" if sentiment["negative"] > sentiment["positive"] else "positive"
            
            # 保存结果
            result = {
                "text": text,
                "target_stance": target_stance,
                "true_label": true_label,
                "pred_label": pred_label,
                "sentiment_scores": sentiment,
                "is_correct": (true_label == pred_label)
            }
            f_out.write(json.dumps(result) + '\n')
            
            # 收集标签
            true_labels.append(true_label)
            pred_labels.append(pred_label)
    
    # 计算指标
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"Total Samples: {len(true_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=["negative", "positive"]))
    
    return {"accuracy": accuracy, "output_path": output_path}

# 示例调用
results = evaluate_stance_accuracy(
    input_path="/root/autodl-tmp/code_evaluate/result/llama3/stance_llama3.jsonl",
    output_path="llama3_results_contro_against.jsonl"
)