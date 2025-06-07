from openai import OpenAI
import os
import base64

client = OpenAI(
 
    base_url='',
    api_key='',
)

messages=[
            {
                "role": "user",
                "content": "User post: .@Tim_Walz will be an extraordinary vice president. He has brought the joy to people in small towns, big towns, and everywhere in between, and he understands what it means to be a leader who lifts people up instead of trying to beat people down. https://t.co/0aQmRj413Y Regarding this image and Harris' post, please generate a comment with stance: against.",
            }
        ]

completion = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    max_tokens=128,
)

print(completion)
print(completion.choices[0].message.content)
