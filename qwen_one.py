import os
from dashscope import MultiModalConversation
import re

# 将xxxx/test.png替换为你本地图像的绝对路径
local_path = "/root/autodl-tmp/data/resources/1851256192527417619_image.jpg"
image_path = f"file://{local_path}"
content = "User post: .@Tim_Walz will be an extraordinary vice president. He has brought the joy to people in small towns, big towns, and everywhere in between, and he understands what it means to be a leader who lifts people up instead of trying to beat people down. https://t.co/0aQmRj413Y Regarding this image and Harris' post, please generate a comment with stance: against."

# 提取结尾的立场词（即 'against' 或 'favor'），不区分大小写
match = re.search(r"stance:\s*(against|favor)", content, re.IGNORECASE)

if match:
    stance = match.group(1).lower()  # 保证提取出来的立场词是小写
    # 根据立场生成新的prompt
    new_prompt = f"Regarding this image, please generate a comment with stance: {stance}."
    print(new_prompt)
    messages = [{'role':'user',
                'content': [{'image': image_path},
                            {'text': new_prompt}]}]
    
    response = MultiModalConversation.call(
  
        api_key="",
        model='qwen-vl-plus',
        messages=messages,
        max_tokens=128
        )
    
    # print(response)
    print(response["output"]["choices"][0]["message"].content[0]["text"])
    
else:
    print("No stance found in the content.")

