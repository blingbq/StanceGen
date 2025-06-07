import json
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np

def calculate_cos_similarity(text_pairs, emb_model_name_or_path, device="cuda"):
    """计算文本对的余弦相似度"""
    # 加载模型和分词器
    model = AutoModel.from_pretrained(emb_model_name_or_path, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(emb_model_name_or_path, trust_remote_code=True)

    # 提取文本对
    texts1, texts2 = zip(*text_pairs)

    # 编码文本
    encoded1 = tokenizer(list(texts1), padding=True, truncation=True, return_tensors="pt").to(device)
    encoded2 = tokenizer(list(texts2), padding=True, truncation=True, return_tensors="pt").to(device)

    # 计算嵌入
    with torch.no_grad():
        output1 = model(**encoded1)
        output2 = model(**encoded2)
        emb1 = output1.last_hidden_state[:, 0]  # 取[CLS] token的嵌入
        emb2 = output2.last_hidden_state[:, 0]

    # 归一化并计算相似度
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cos_scores = torch.sum(emb1 * emb2, dim=1)

    # 清理GPU内存
    del model, tokenizer
    torch.cuda.empty_cache()

    return cos_scores.cpu().numpy()

def evaluate_similarity(input_path, emb_model_name_or_path, output_path, batch_size=32):
    """评估predict和target的相似性"""
    # 读取数据
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # 准备文本对
    text_pairs = []
    for line in lines:
        data = json.loads(line)
        text_pairs.append((data['llama3'], data['target']))

    # 分批处理以避免OOM
    all_scores = []
    for i in tqdm(range(0, len(text_pairs), batch_size), desc="Calculating similarity"):
        batch = text_pairs[i:i+batch_size]
        scores = calculate_cos_similarity(batch, emb_model_name_or_path)
        all_scores.extend(scores)

    # 保存结果
    results = []
    for line, score in zip(lines, all_scores):
        data = json.loads(line)
        data['cosine_similarity'] = float(score)
        results.append(data)

    with open(output_path, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')

    # 统计指标
    avg_score = np.mean(all_scores)
    print(f"Average cosine similarity: {avg_score:.4f}")
    return avg_score

# 使用示例
if __name__ == "__main__":
    input_path = "/root/autodl-tmp/code_evaluate/result/llama3/stance_llama3.jsonl"  # 你的输入文件
    emb_model_name_or_path = "/root/autodl-tmp/code_evaluate/bge-large-en-v1.5"  # 嵌入模型
    output_path = "llama3_similarity_results.jsonl"  # 输出文件

    avg_score = evaluate_similarity(input_path, emb_model_name_or_path, output_path)
    print(f"Results saved to {output_path}")