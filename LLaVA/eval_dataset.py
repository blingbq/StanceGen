import json
import os
import argparse
import torch
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from PIL import Image
import re
from tqdm import tqdm


def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image


def eval_single_image(model, tokenizer, image_processor, context_len, args, image_path):
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # print(f"qs: {qs}\n")
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = load_image(image_path)
    images_tensor = process_images(
        [image],
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    # print(f"images_tensor shape: {images_tensor.shape}")
    image_size = image.size

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def eval_dataset(args):
    # Load model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # Auto-detect conversation mode
    if "llama-2" in model_name.lower():
        args.conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        args.conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        args.conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        args.conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        args.conv_mode = "mpt"
    else:
        args.conv_mode = "llava_v0"

    results = []

    with open(args.test_json, 'r') as f:
        for line in tqdm(f, desc="Evaluating dataset"):
            entry = json.loads(line)
            image_path = f"{args.image_folder}/{entry['image_path']}"
            args.query = entry['prompt']

            output = eval_single_image(
                model, tokenizer, image_processor, context_len, args, image_path
            )
            print(f"prompt: {entry['prompt']}\n")
            print(f"output: {output}\n")

            results.append({
                "unique_id": entry['unique_id'],
                "image_path": entry['image_path'],
                "prompt": entry['prompt'],
                "target": entry['target'],
                "output": output
            })

    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--test-json", type=str, required=True, help="Path to the test JSON file.")
    parser.add_argument("--image-folder", type=str, required=True, help="Root folder for image files.")
    parser.add_argument("--query", type=str, default=None, help="Query string for evaluation.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the evaluation results.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()

    eval_dataset(args)
    