import json
from tqdm import tqdm
import os
from openai import OpenAI
import base64

client = OpenAI(

    base_url='',
    api_key='',
)

# 定义API调用的函数
def call_api_for_data(text_content):
    messages=[
                {
                "role": "user",
                "content": text_content,
                }
            ]
    
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=128,
    )
    
    # 返回GPT的回复
    return completion.choices[0].message.content

# 读取原始JSON文件
input_json_file = '/root/autodl-tmp/data/test.json'  # 请替换成你自己的文件路径
output_json_file = 'gpt_4_results.json'  # 输出结果文件路径

# 读取数据集
with open(input_json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理每一条数据
with open(output_json_file, 'a', encoding='utf-8') as f:  # 使用 'a' 模式表示追加
    for entry in tqdm(data, desc="Evaluating dataset"):
        # 获取human的value并去掉<image>\n标识符
        human_value = entry['conversations'][0]['value']
        text_content = human_value.replace('<image>\n', '').strip()  # 去掉<image>\n标识符
        
        # 调用API获取结果
        api_response = call_api_for_data(text_content)
        
        # 将结果添加到结果列表
        result = {
            "user": text_content,
            "gpt4": api_response
        }
        
        print(f"user: {text_content}")
        print(f"gpt4: {api_response}")
        
        # 逐条写入输出文件
        json.dump([result], f, ensure_ascii=False, indent=4)
        f.write("\n")  # 每条记录后写入换行符，便于区分每条记录
        
        # continue
        # break

# 将结果写入新的JSON文件
# with open(output_json_file, 'w', encoding='utf-8') as f:
#     json.dump(test_results, f, ensure_ascii=False, indent=4)

print(f"Test results saved to {output_json_file}")
