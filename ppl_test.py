import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def calculate_ppl_scores(
    sentences: list,
    model,
    tokenizer,
    max_length=512,
    device="cuda",
    max_ppl_threshold=1000,
) -> list:
    """Calculate the perplexity scores for the given sentences using the PPL model, with an upper limit."""
    all_ppl = []
    for sentence in tqdm(sentences, desc="Calculating PPL", unit="sentence"):
        if sentence == "":
            all_ppl.append(None)
            continue

        inputs = tokenizer.encode(
            sentence, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            ppl = torch.exp(loss).item()
            # 检查是否为 NaN
            if torch.isnan(loss).item() or ppl > max_ppl_threshold:
                ppl = None
            all_ppl.append(ppl)

    return all_ppl

def load_experiment_results(file_path: str) -> list:
    """Load experiment results from a JSONL file."""
    results = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line: {line}")
    return results

def evaluate_perplexity_for_all_models(results_dict: dict, model, tokenizer, device="cuda"):
    """Evaluate perplexity scores for all models based on JSONL format."""
    ppl_scores = {}

    for model_name, result_file in results_dict.items():
        print(f"Evaluating on model: {model_name}")

        # 加载 JSONL 结果
        results = load_experiment_results(result_file)

        # 提取待评估文本字段（以 gpt_4o 为例）
        sentences = [item['llama3'] for item in results if 'llama3' in item]

        # 计算困惑度
        model_ppl_scores = calculate_ppl_scores(sentences, model, tokenizer, device=device)
        ppl_scores[model_name] = model_ppl_scores

        valid_scores = [s for s in model_ppl_scores if s is not None]
        avg_ppl = sum(valid_scores) / len(valid_scores) if valid_scores else None
        print(f"Average perplexity scores for {model_name}: {avg_ppl}")

    average_ppl_scores = {
        model_name: (
            sum(scores for scores in score_list if scores is not None) /
            len([s for s in score_list if s is not None])
            if any(s is not None for s in score_list) else None
        )
        for model_name, score_list in ppl_scores.items()
    }

    return ppl_scores, average_ppl_scores

# 示例设置
results_dict = {
    "example": "/root/autodl-tmp/code_evaluate/result/llama3/stance_llama3.jsonl"
}
output_file = "llama3_ppl_output.json"

model_path = "/root/autodl-tmp/code_evaluate/gpt2-large"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# 执行 PPL 评估
ppl_scores, average_ppl_scores = evaluate_perplexity_for_all_models(
    results_dict, model, tokenizer, device="cuda"
)

# 保存结果为 JSON
ppl_results = []
for model_name, scores in ppl_scores.items():
    avg = average_ppl_scores[model_name]
    ppl_results.append({
        "model": model_name,
        "scores": scores,
        "avg": avg,
    })
    print(f"Average perplexity scores for {model_name}: {avg}")

with open(output_file, "w") as f:
    json.dump(ppl_results, f, indent=4)

print(f"Results saved to {output_file}")
    