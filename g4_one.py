from openai import OpenAI
import os
import base64


#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 将xxxx/test.png替换为你本地图像的绝对路径
base64_image = encode_image("/root/autodl-tmp/data/resources/1851256192527417619_image.jpg")

client = OpenAI(
    # openai系列的sdk，包括langchain，都需要这个/v1的后缀
    base_url='',
    api_key='',
)

messages=[
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                # PNG图像：  f"data:image/png;base64,{base64_image}"
                # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                # WEBP图像： f"data:image/webp;base64,{base64_image}"
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
            },
            {"type": "text", "text": "User post: .@Tim_Walz will be an extraordinary vice president. He has brought the joy to people in small towns, big towns, and everywhere in between, and he understands what it means to be a leader who lifts people up instead of trying to beat people down. https://t.co/0aQmRj413Y Regarding this image and Harris' post, please generate a comment with stance: against."},
        ],
    }

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_tokens=128,
)

print(completion)
print(completion.choices[0].message.content)

