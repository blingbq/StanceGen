import json
import io
import sys
from llava.eval.run_llava import eval_model

image_folder = "/root/autodl-tmp/data/"

# 加载 test.json 文件
with open("/root/autodl-tmp/data/test.json", "r") as f:
    test_data = json.load(f)

# 用于保存每个评估结果
eval_results = []

# 遍历 test.json 中的每个数据项
for item in test_data:
    image_file = image_folder + item["image"]  # 图片路径
    prompt = item["conversations"][0]["value"]  # 获取第一个 human 对话的内容作为 prompt

    # 设置评估参数
    args = type('Args', (), {
        "model_path": "/root/autodl-tmp/model/llava-v1.5-7b",  # 原始模型路径
        "model_base": None,
        "model_name": "liuhaotian/llava-v1.5-7b",
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 128
    })()
    # print(f"Processing image: {image_file}, with prompt: {prompt}")
    fine_tuned_output = io.StringIO()
    sys.stdout = fine_tuned_output  # 重定向标准输出
    eval_model(args)
    sys.stdout = sys.__stdout__  # 恢复标准输出

    # 获取微调后的模型的输出
    fine_tuned_output_str = fine_tuned_output.getvalue()
    
    print(f"image: {image_file}, prompt: {prompt}")
    print(f"output: {fine_tuned_output_str}")

    # 保存微调后的模型生成的回答
    eval_results.append({
        "image": image_file,
        "prompt": prompt,
        "output": fine_tuned_output_str.strip()
    })
    
    break

# 保存评估结果到文件
with open("generation_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)

print("生成结果已保存到 'generation_results.json'")
