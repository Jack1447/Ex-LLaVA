import os
# 强制使用FP32精度以避免数值不稳定问题
os.environ["LLAVA_FORCE_FP32"] = "1"

import torch
import argparse
import json
from PIL import Image
from transformers import TextIteratorStreamer
from threading import Thread
from glob import glob

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

def generate_from_model(model_path, clip_path, device, task_prompt, image, temperature=0.7, top_p=0.9):
    """使用指定模型从图片生成内容的通用函数"""
    # 加载模型
    print(f"正在加载模型: {model_path}")
    print(f"正在加载CLIP模型: {clip_path}")
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        None,  # model_base
        model_path.split('/')[-1],  # model_name
        False,  # load_8bit
        False,  # load_4bit
        device=device,
        lora_path=None,  # 假设使用合并后的模型
        clip_path=clip_path
    )
    
    # 确保模型在正确的设备上
    model = model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    # 准备对话模板
    conv_template = conv_templates["llava_v1"]
    
    # 预处理图像 
    images = [image]
    images = process_images(images, image_processor, model.config)
    
    # 确保图像数据类型与模型一致 
    target_dtype = getattr(model, "dtype", torch.float16)
    if type(images) is list:
        images = [img.to(model.device, dtype=target_dtype) for img in images]
    else:
        images = images.to(model.device, dtype=target_dtype)
    
    # 构建提示
    conv = conv_template.copy()
    conv.append_message(conv.roles[0], task_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # 处理图像标记 
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    
    # 移动到设备
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    
    # 确保生成配置的pad/eos一致
    if getattr(model, "generation_config", None) is not None:
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        if model.generation_config.eos_token_id is None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
    
    # 计算max_new_tokens
    max_context_length = getattr(model.config, 'max_position_embeddings', 4096)
    max_new_tokens_cap = 2048
    max_new_tokens = 2048
    
    # 计算图像token数量
    num_image_tokens = 0
    if images is not None:
        image_token_count = prompt.count(DEFAULT_IMAGE_TOKEN)
        num_image_tokens = image_token_count * model.get_vision_tower().num_patches
    
    # 限制max_new_tokens
    max_new_tokens = min(max_new_tokens, max_new_tokens_cap)
    max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)
    
    # 检查是否有足够空间生成token
    if max_new_tokens < 1:
        raise ValueError("超出最大token长度限制")
    
    do_sample = True
    stop_str = tokenizer.eos_token if tokenizer.eos_token else "</s>"
    
    # 创建image_args字典
    image_args = {"images": images} if images is not None else {}
    
    # 生成回复
    print("正在生成内容...")
    with torch.inference_mode():
        # 构建生成参数
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300)
        
        generate_kwargs = {
            'inputs': input_ids,
            'attention_mask': attention_mask.to(model.device),
            'do_sample': do_sample,
            'temperature': temperature,
            'top_p': top_p,
            'max_new_tokens': max_new_tokens,
            'use_cache': True,
            'stopping_criteria': [stopping_criteria],
            'streamer': streamer,
            **image_args
        }
        
        # 使用线程进行生成
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()
        
        # 获取生成的文本
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
    
    return generated_text

def process_single_image(image_path, args, concepts_prompt, disease_prompt):
    """处理单张图片并返回结果"""
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # 打开图像并转换为RGB
    image = Image.open(image_path).convert('RGB')
    print(f"成功加载图片: {image_path}")
    
    # 结果字典
    results = {}
    
    # 根据任务类型执行不同的推理
    if args.task == "concepts":
        print("执行概念预测任务...")
        concepts_result = generate_from_model(
            args.concept_model, args.clip_path, args.device,
            concepts_prompt, image, args.temperature, args.top_p
        )
        
        print("\n概念预测结果:")
        print(concepts_result)
        results["concepts"] = concepts_result
        
    elif args.task == "disease":
        print("执行疾病分类任务...")
        disease_result = generate_from_model(
            args.disease_model, args.clip_path, args.device,
            disease_prompt, image, args.temperature, args.top_p
        )
        
        print("\n疾病分类结果:")
        print(disease_result)
        results["disease"] = disease_result
        
    elif args.task == "both":
        print("执行完整分析流程: 概念预测 -> 疾病分类")
        
        # 先执行概念预测
        print("\n阶段1: 概念预测")
        concepts_result = generate_from_model(
            args.concept_model, args.clip_path, args.device,
            concepts_prompt, image, args.temperature, args.top_p
        )
        
        print("\n概念预测结果:")
        print(concepts_result)
        results["concepts"] = concepts_result
        
        # 清理内存以避免OOM
        torch.cuda.empty_cache()
        
        # 再执行疾病分类
        print("\n阶段2: 疾病分类")
        # 修改疾病prompt，添加概念阶段生成的内容
        # 确保保留完整的原prompt结构
        modified_disease_prompt = "You are a professional dermoscopy analysis assistant specialized in feature-level interpretation of skin disease images. Your task is to analyze skin disease images and predict the disease label, based on detailed visual cues and concepts information.\nHere are the concepts and their respective rationales:" + concepts_result + ".### Specific requirements:\n- **The possible labels for disease are**: \"nevus\", \"melanoma\".\n- The output MUST strictly follow the format below and include all required fields.\n\nThe expected output format is:\n\n<BEGIN_OUTPUT>\n{\n    \"label\": \"nevus\",   # or \"melanoma\"\n    \"positive evidence\": \"Describe the observable features from the image that support the assigned category, including characteristics such as pigment network type and arrangement, streaks, dots and globules, blue-whitish veil, and regression structures.\",\n    \"negative evidence\": \"Describe features that rule out the opposite category, such as absent or inconsistent structures.\",\n    \"summary\": \"Provide a concise rationale explaining why the assigned label is correct based on diagnostic criteria and concepts information.\"\n}\n<END_OUTPUT>"
        disease_result = generate_from_model(
            args.disease_model, args.clip_path, args.device,
            modified_disease_prompt, image, args.temperature, args.top_p
        )
        
        print("\n疾病分类结果:")
        print(disease_result)
        results["disease"] = disease_result
        
    else:
        raise ValueError(f"不支持的任务类型: {args.task}. 请使用 'concepts', 'disease' 或 'both'.")
    
    return results

def generate_from_image(args):
    """支持单张图片和文件夹的生成功能"""
    # 定义概念预测任务提示
    concepts_prompt = "The output MUST strictly follow the format below. Choose one label for each concept from the possible options:\n\n<BEGIN_OUTPUT>\n{\n    \"pigment network\": {\n        \"label\": \"choose one from [\"absent\", \"typical\", \"atypical\"]\",\n        \"rationale\": \"Explain why you chose the label for pigment network.\"\n    },\n    \"streaks\": {\n        \"label\": \"choose one from [\"absent\", \"regular\", \"irregular\"]\",\n        \"rationale\": \"Explain why you chose the label for streaks.\"\n    },\n    \"dots and globules\": {\n        \"label\": \"choose one from [\"absent\", \"regular\", \"irregular\"]\",\n        \"rationale\": \"Explain why you chose the label for dots and globules.\"\n    },\n    \"blue-whitish veil\": {\n        \"label\": \"choose one from [\"absent\", \"present\"]\",\n        \"rationale\": \"Explain why you chose the label for blue-whitish veil.\"\n    },\n    \"regression structures\": {\n        \"label\": \"choose one from [\"absent\", \"present\"]\",\n        \"rationale\": \"Explain why you chose the label for regression structures.\"\n    }\n}\n<END_OUTPUT>\n\nPlease make sure to choose one label from the list of possible labels for each concept. Provide a rationale for each label based on visual features that can be observed in the image."
    
    # 定义疾病分类任务提示
    disease_prompt = "You are a professional dermoscopy analysis assistant specialized in feature-level interpretation of skin disease images. Your task is to analyze skin disease images and predict the disease label, based on detailed visual cues and concepts information.\n\n### Specific requirements:\n- **The possible labels for disease are**: \"nevus\", \"melanoma\".\n- The output MUST strictly follow the format below and include all required fields.\n\nThe expected output format is:\n\n<BEGIN_OUTPUT>\n{\n    \"label\": \"nevus\",   # or \"melanoma\"\n    \"positive evidence\": \"Describe the observable features from the image that support the assigned category, including characteristics such as pigment network type and arrangement, streaks, dots and globules, blue-whitish veil, and regression structures.\",\n    \"negative evidence\": \"Describe features that rule out the opposite category, such as absent or inconsistent structures.\",\n    \"summary\": \"Provide a concise rationale explaining why the assigned label is correct based on diagnostic criteria and concepts information.\"\n}\n<END_OUTPUT>"
    
    # 确定处理模式：单张图片或文件夹
    if args.folder_path:
        # 文件夹模式
        if not os.path.isdir(args.folder_path):
            raise FileNotFoundError(f"文件夹不存在: {args.folder_path}")
        
        # 获取文件夹中的所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(args.folder_path, ext)))
        
        if not image_files:
            raise ValueError(f"文件夹中没有找到图片文件: {args.folder_path}")
        
        print(f"发现 {len(image_files)} 张图片待处理")
        
        # 构建输出数据结构
        output_data = {
            "stage": args.task,
        }
        
        # 根据任务类型添加相应的模型路径
        if args.task == "concepts":
            output_data["concepts_model_path"] = args.concept_model
        elif args.task == "disease":
            output_data["disease_model_path"] = args.disease_model
        elif args.task == "both":
            output_data["concepts_model_path"] = args.concept_model
            output_data["disease_model_path"] = args.disease_model
        
        # 添加其他字段
        output_data["clip_path"] = args.clip_path
        output_data["generate_text"] = {}
        
        # 遍历处理每张图片
        for image_path in image_files:
            print(f"\n=== 处理图片: {image_path} ===")
            try:
                # 处理单张图片
                results = process_single_image(image_path, args, concepts_prompt, disease_prompt)
                
                # 获取图片ID（文件名无后缀）
                image_filename = os.path.basename(image_path)
                image_id = os.path.splitext(image_filename)[0]
                
                # 根据任务类型构建图片结果
                if args.task == "concepts":
                    output_data["generate_text"][image_id] = {
                        "image_path": image_path,
                        "concepts": results["concepts"]
                    }
                elif args.task == "disease":
                    output_data["generate_text"][image_id] = {
                        "image_path": image_path,
                        "disease": results["disease"]
                    }
                elif args.task == "both":
                    output_data["generate_text"][image_id] = {
                        "image_path": image_path,
                        "concepts": results["concepts"],
                        "disease": results["disease"]
                    }
                
                # 清理内存
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {str(e)}")
                continue
        
        # 生成输出文件名（如果未指定）
        if not args.output_file:
            args.output_file = f"inference_{args.task}.json"
    
    else:
        # 单张图片模式
        # 处理单张图片
        results = process_single_image(args.image_path, args, concepts_prompt, disease_prompt)
        
        # 获取图片ID（文件名无后缀）
        image_filename = os.path.basename(args.image_path)
        image_id = os.path.splitext(image_filename)[0]
        
        # 构建输出数据结构
        output_data = {
            "stage": args.task,
        }
        
        # 根据任务类型添加相应的模型路径
        if args.task == "concepts":
            output_data["concepts_model_path"] = args.concept_model
        elif args.task == "disease":
            output_data["disease_model_path"] = args.disease_model
        elif args.task == "both":
            output_data["concepts_model_path"] = args.concept_model
            output_data["disease_model_path"] = args.disease_model
        
        # 添加其他字段
        output_data["clip_path"] = args.clip_path
        output_data["generate_text"] = {}
        
        # 根据任务类型添加结果
        if args.task == "concepts":
            output_data["generate_text"][image_id] = {
                "image_path": args.image_path,
                "concepts": results["concepts"]
            }
        elif args.task == "disease":
            output_data["generate_text"][image_id] = {
                "image_path": args.image_path,
                "disease": results["disease"]
            }
        elif args.task == "both":
            output_data["generate_text"][image_id] = {
                "image_path": args.image_path,
                "concepts": results["concepts"],
                "disease": results["disease"]
            }
        
        # 生成输出文件名（如果未指定）
        if not args.output_file:
            args.output_file = f"inference_single_{args.task}_{image_id}.json"
    
    # 保存结果到JSON文件
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存至: {args.output_file}（JSON格式）")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用LLaVA-Med模型进行图片推理（支持单张图片或文件夹）")
    parser.add_argument("--task", type=str, choices=["concepts", "disease", "both"], required=True,
                        help="任务类型: concepts (概念预测), disease (疾病分类) 或 both (依次执行两个模型)")
    
    # 图片路径和文件夹路径二选一
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image-path", type=str, help="单张图片路径")
    input_group.add_argument("--folder-path", type=str, help="包含图片的文件夹路径")
    
    parser.add_argument("--concept-model", type=str, 
                        default="/root/autodl-tmp/model/llava-med-concepts", 
                        help="概念阶段模型路径")
    parser.add_argument("--disease-model", type=str, 
                        default="/root/autodl-tmp/model/llava-med-disease", 
                        help="疾病阶段模型路径")
    parser.add_argument("--clip-path", type=str, 
                        default="/root/autodl-tmp/model/clip-vit-large-patch14-336", 
                        help="CLIP模型路径")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度参数")
    parser.add_argument("--top-p", type=float, default=0.9, help="生成top_p参数")
    parser.add_argument("--output-file", type=str, help="结果输出文件路径（不指定则自动生成）")
    
    args = parser.parse_args()
    generate_from_image(args)