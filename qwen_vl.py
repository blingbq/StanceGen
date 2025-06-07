import os
from openai import OpenAI

# 设置 API Key 和 base URL，替换为你实际的 API Key
client = OpenAI(
    api_key="",  # 请替换为你的 API 密钥
    base_url="",  # Dashscope API的基础URL
)

def generate_comment_with_stance(image_url, user_post, stance):
    # 构造 prompt 内容，这里将图像 URL 和用户帖子结合
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": f"User post: {user_post}\nRegarding this image and Harris' post, please generate a comment with stance: {stance}."},  # 包含立场的提示
            {"type": "image_url", "image_url": {"url": image_url}}  # 使用图片 URL
        ]}
    ]

    try:
        # 调用 API，进行立场生成
        completion = client.chat.completions.create(
            model="qwen-vl-plus",  # 使用 Qwen-VL 模型
            messages=messages
        )
        
        # 输出生成的结果
        print(completion.model_dump_json())

    except Exception as e:
        print(f"Error while generating comment: {str(e)}")

# 示例使用
image_url = "https://s21.ax1x.com/2025/02/12/pEunIfO.jpg"  # 使用有效的图片 URL
user_post = "Hopped on the phone to speak with voters (and one future voter) today. Thanks for picking up my call, Jennifer and Sage!\n\nHelp us get every last voter to the polls: https://t.co/l8recL55qG https://t.co/AMlhOWdxZh"
stance = "against"  # 立场要求

# 调用函数生成评论
generate_comment_with_stance(image_url, user_post, stance)
