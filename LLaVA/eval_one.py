from llava.eval.run_llava import eval_model

model_path = "/root/autodl-tmp/model/llava-v1.5-7b"
prompt = "User post: .@Tim_Walz will be an extraordinary vice president. He has brought the joy to people in small towns, big towns, and everywhere in between, and he understands what it means to be a leader who lifts people up instead of trying to beat people down. https://t.co/0aQmRj413Y Regarding this image and Harris' post, please generate a comment with stance: against."
image_file = "/root/autodl-tmp/image/resources/1851256192527417619_image.jpg"

args = type('Args', (), {
    "model_path": "/root/autodl-tmp/model/llava-v1.5-7b",
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

print("原始模型输出为：")
eval_model(args)
'''

args = type('Args', (), {
    "model_path": "/root/autodl-tmp/model/llava-v1.5-7b-merged",
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

print("微调后的模型输出为：")
eval_model(args)
'''