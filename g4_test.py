import json
from tqdm import tqdm
import os
import re
from openai import OpenAI
import base64

#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

client = OpenAI(
    # openai系列的sdk，包括langchain，都需要这个/v1的后缀
    base_url='',
    api_key='',
)

# 定义API调用的函数
def call_api_for_data(image_path, text_content):
    # 构建messages
    base64_image = encode_image(image_path)
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                },
                {"type": "text", "text": text_content},
            ],
        }
    ]
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=128,
    )
    
    # 返回GPT的回复
    return completion.choices[0].message.content

# 读取原始JSON文件
input_json_file = '/root/autodl-tmp/data/test.json'  # 请替换成你自己的文件路径
output_json_file = 'gpt_4o_image_results.json'  # 输出结果文件路径
image_folder = '/root/autodl-tmp/data'

# 读取数据集
with open(input_json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理每一条数据
with open(output_json_file, 'a', encoding='utf-8') as f:  # 使用 'a' 模式表示追加
    for entry in tqdm(data, desc="Evaluating dataset"):
        image_path = entry['image']
        image_path = f"{image_folder}/{entry['image']}"
        # 获取human的value并去掉<image>\n标识符
        human_value = entry['conversations'][0]['value']
        text_content = human_value.replace('<image>\n', '').strip()  # 去掉<image>\n标识符
        
        # 提取结尾的立场词（即 'against' 或 'favor'），不区分大小写
        match = re.search(r"stance:\s*(against|favor)", text_content, re.IGNORECASE)
        if match:
            stance = match.group(1).lower()  # 保证提取出来的立场词是小写
            # 根据立场生成新的prompt
            new_prompt = f"Regarding this image, please generate a comment with stance: {stance}."
        
        # 调用API获取结果
        api_response = call_api_for_data(image_path, new_prompt)
        
        # 将结果添加到结果列表
        result = {
            "image": entry['image'],
            "user": new_prompt,
            "qwen": api_response
        }
        print(f"user: {new_prompt}")
        print(f"qwen: {api_response}")
        
        # 逐条写入输出文件
        json.dump([result], f, ensure_ascii=False, indent=4)
        f.write("\n")  # 每条记录后写入换行符，便于区分每条记录
        
        # continue
        # break

# 将结果写入新的JSON文件
# with open(output_json_file, 'w', encoding='utf-8') as f:
#     json.dump(test_results, f, ensure_ascii=False, indent=4)

print(f"Test results saved to {output_json_file}")
