import json
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
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


def get_feature_representation(model, input_ids, images_tensor, image_sizes):
    """获取模型内部的特征表示"""
    # 这个函数需要根据具体模型架构进行调整
    # 这里假设模型有一个获取特征的方法或可以钩入中间层
    
    with torch.inference_mode():
        # 这里获取模型内部的特征表示，具体方法取决于模型架构
        # 对于LLaVA模型，我们需要获取视觉特征和文本特征
        features = model.get_visual_features(images=images_tensor, image_sizes=image_sizes)
        # 或者我们可以钩入模型的前向传播过程
        # outputs = model(input_ids, images=images_tensor, image_sizes=image_sizes, output_hidden_states=True)
        # features = outputs.hidden_states[-1]  # 使用最后一层的隐藏状态
    
    return features


def feature_fusion(features_list, weights=None):
    """融合多个模型的特征表示"""
    if weights is None:
        # 如果未指定权重，则平均融合
        weights = [1.0 / len(features_list)] * len(features_list)
    
    # 确保权重和为1
    weights = [w / sum(weights) for w in weights]
    
    # 初始化融合特征
    fused_features = None
    
    for i, features in enumerate(features_list):
        if fused_features is None:
            fused_features = weights[i] * features
        else:
            # 可能需要处理不同模型特征维度不同的情况
            # 这里假设所有特征具有相同的维度
            fused_features += weights[i] * features
    
    return fused_features


def eval_single_image_with_fusion(models, tokenizers, image_processors, context_lens, args, image_path, fusion_weights=None):
    """使用特征融合评估单个图像"""
    
    qs = args.query
    
    # 准备每个模型的输入
    feature_list = []
    for i, (model, tokenizer, image_processor, context_len) in enumerate(zip(models, tokenizers, image_processors, context_lens)):
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        
        # 处理提示文本中的图像标记
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                curr_qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                curr_qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                curr_qs = image_token_se + "\n" + qs
            else:
                curr_qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        
        # 构建对话模板
        conv_mode = args.conv_modes[i] if i < len(args.conv_modes) else args.conv_modes[0]
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], curr_qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # 处理图像
        image = load_image(image_path)
        images_tensor = process_images(
            [image],
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)
        image_size = image.size
        
        # 准备输入ID
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        
        # 获取特征表示
        features = get_feature_representation(model, input_ids, images_tensor, [image_size])
        feature_list.append(features)
    
    # 特征融合
    fused_features = feature_fusion(feature_list, fusion_weights)
    
    # 使用融合特征生成输出
    # 这部分需要根据模型架构进行调整
    # 这里假设主模型可以接受外部特征进行生成
    primary_model = models[0]
    primary_tokenizer = tokenizers[0]
    
    with torch.inference_mode():
        output_ids = primary_model.generate_from_features(
            fused_features,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    
    outputs = primary_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def eval_single_image(model, tokenizer, image_processor, context_len, args, image_path):
    """评估单个图像（保持原始功能）"""
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
    # 加载所有模型
    disable_torch_init()
    
    models = []
    tokenizers = []
    image_processors = []
    context_lens = []
    conv_modes = []
    
    # 加载主模型
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    
    # 检测对话模式
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    
    models.append(model)
    tokenizers.append(tokenizer)
    image_processors.append(image_processor)
    context_lens.append(context_len)
    conv_modes.append(conv_mode)
    
    # 加载额外模型（如果有）
    if args.extra_model_paths:
        for i, extra_model_path in enumerate(args.extra_model_paths):
            extra_model_base = args.extra_model_bases[i] if i < len(args.extra_model_bases) else None
            
            extra_model_name = get_model_name_from_path(extra_model_path)
            extra_tokenizer, extra_model, extra_image_processor, extra_context_len = load_pretrained_model(
                extra_model_path, extra_model_base, extra_model_name
            )
            
            # 检测对话模式
            if "llama-2" in extra_model_name.lower():
                extra_conv_mode = "llava_llama_2"
            elif "mistral" in extra_model_name.lower():
                extra_conv_mode = "mistral_instruct"
            elif "v1.6-34b" in extra_model_name.lower():
                extra_conv_mode = "chatml_direct"
            elif "v1" in extra_model_name.lower():
                extra_conv_mode = "llava_v1"
            elif "mpt" in extra_model_name.lower():
                extra_conv_mode = "mpt"
            else:
                extra_conv_mode = "llava_v0"
            
            models.append(extra_model)
            tokenizers.append(extra_tokenizer)
            image_processors.append(extra_image_processor)
            context_lens.append(extra_context_len)
            conv_modes.append(extra_conv_mode)
    
    # 设置对话模式
    args.conv_mode = conv_modes[0]  # 主模型的对话模式
    args.conv_modes = conv_modes  # 所有模型的对话模式
    
    # 设置融合权重
    fusion_weights = None
    if args.fusion_weights:
        fusion_weights = [float(w) for w in args.fusion_weights.split(',')]
        # 确保权重数量与模型数量一致
        if len(fusion_weights) != len(models):
            print(f"Warning: Number of fusion weights ({len(fusion_weights)}) doesn't match number of models ({len(models)}). Using equal weights.")
            fusion_weights = None
    
    results = []
    
    with open(args.test_json, 'r') as f:
        for line in tqdm(f, desc="Evaluating dataset"):
            entry = json.loads(line)
            image_path = f"{args.image_folder}/{entry['image_path']}"
            args.query = entry['prompt']
            
            if len(models) > 1 and args.use_fusion:
                # 使用特征融合
                output = eval_single_image_with_fusion(
                    models, tokenizers, image_processors, context_lens,
                    args, image_path, fusion_weights
                )
            else:
                # 使用单一模型
                output = eval_single_image(
                    models[0], tokenizers[0], image_processors[0], context_lens[0],
                    args, image_path
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
    # 基本参数
    parser.add_argument("--model-path", type=str, required=True, help="Path to the primary model.")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--test-json", type=str, required=True, help="Path to the test JSON file.")
    parser.add_argument("--image-folder", type=str, required=True, help="Root folder for image files.")
    parser.add_argument("--query", type=str, default=None, help="Query string for evaluation.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the evaluation results.")
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    
    # 多模型融合参数
    parser.add_argument("--extra-model-paths", type=str, nargs='+', default=None, 
                        help="Paths to additional models for fusion.")
    parser.add_argument("--extra-model-bases", type=str, nargs='+', default=None,
                        help="Base models for additional models.")
    parser.add_argument("--use-fusion", action="store_true", default=False,
                        help="Whether to use feature fusion.")
    parser.add_argument("--fusion-weights", type=str, default=None,
                        help="Comma-separated weights for model fusion (e.g., '0.7,0.3').")
    
    args = parser.parse_args()
    
    eval_dataset(args)