import os
# 强制使用FP32精度以避免数值不稳定问题
os.environ["LLAVA_FORCE_FP32"] = "1"

import json
import torch
import argparse
from PIL import Image
from transformers import TextIteratorStreamer
from threading import Thread
from llava.mm_utils import KeywordsStoppingCriteria
import base64
from io import BytesIO
import re
from collections import defaultdict
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

def load_json_dataset(file_path):
    """加载JSON格式的测试数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def encode_image(image_path):
    """将图像编码为base64字符串"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_image_from_base64(base64_string):
    """从base64字符串加载图像"""
    image = Image.open(BytesIO(base64.b64decode(base64_string)))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def extract_concept_labels(output):
    """从模型输出中提取概念标签"""
    try:
        # 尝试直接解析JSON（根据用户提供的标准格式）
        try:
            # 清理输出，移除可能的前后文本
            output = output.strip()
            # 尝试找到JSON的开始和结束位置
            start_idx = output.find('{')
            end_idx = output.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = output[start_idx:end_idx]
                data = json.loads(json_str)
                # 验证是否包含所有需要的概念
                required_concepts = ["pigment network", "streaks", "dots and globules", "blue-whitish veil", "regression structures"]
                valid_data = {}
                for concept in required_concepts:
                    if concept in data and isinstance(data[concept], dict) and "label" in data[concept]:
                        valid_data[concept] = {
                            "label": data[concept]["label"].strip().lower()
                        }
                return valid_data
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON解析失败: {e}")
            
        # 输出解析失败时的调试信息
        print(f"无法解析输出格式: {output[:100]}...")
        return {}
    except Exception as e:
        print(f"提取概念标签时出错: {e}")
        return {}

def extract_disease_label(output):
    """从模型输出中提取疾病标签"""
    try:
        # 尝试直接解析JSON（根据用户提供的标准格式）
        try:
            # 清理输出，移除可能的前后文本
            output = output.strip()
            # 尝试找到JSON的开始和结束位置
            start_idx = output.find('{')
            end_idx = output.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = output[start_idx:end_idx]
                data = json.loads(json_str)
                # 验证是否包含label字段
                if "label" in data and isinstance(data["label"], str):
                    return data["label"].strip().lower()
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON解析失败: {e}")
            
        # 输出解析失败时的调试信息
        print(f"无法解析输出格式: {output[:100]}...")
        return None
    except Exception as e:
        print(f"提取疾病标签时出错: {e}")
        return None

def evaluate_model(args):
    """评估模型在指定任务上的性能"""
    # 加载模型
    print(f"正在加载基础模型: {args.model_path}")
    print(f"正在加载LoRA权重: {args.lora_path}")
    print(f"正在加载CLIP模型: {args.clip_path}")
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        None,  # model_base
        args.model_path.split('/')[-1],  # model_name
        False,  # load_8bit
        False,  # load_4bit
        device=args.device,
        lora_path=args.lora_path,
        clip_path=args.clip_path
    )
    
    # 确保模型在正确的设备上
    model = model.to(args.device)
    
    # 设置为评估模式
    model.eval()
    
    # 加载测试数据集
    print(f"正在加载测试数据集: {args.dataset_path}")
    dataset = load_json_dataset(args.dataset_path)
    
    # 正确获取数据样本列表
    test_data = dataset.get("data", [])
    
    # 准备对话模板
    conv_template = conv_templates["llava_v1"]
    
    # 根据任务类型设置提示和相关参数
    if args.task_type == "concepts":
        print("执行概念预测任务评估...")
        # 定义概念预测任务提示
        task_prompt = "The output MUST strictly follow the format below. Choose one label for each concept from the possible options:\n\n<BEGIN_OUTPUT>\n{\n    \"pigment network\": {\n        \"label\": \"choose one from [\"absent\", \"typical\", \"atypical\"]\",\n        \"rationale\": \"Explain why you chose the label for pigment network.\"\n    },\n    \"streaks\": {\n        \"label\": \"choose one from [\"absent\", \"regular\", \"irregular\"]\",\n        \"rationale\": \"Explain why you chose the label for streaks.\"\n    },\n    \"dots and globules\": {\n        \"label\": \"choose one from [\"absent\", \"regular\", \"irregular\"]\",\n        \"rationale\": \"Explain why you chose the label for dots and globules.\"\n    },\n    \"blue-whitish veil\": {\n        \"label\": \"choose one from [\"absent\", \"present\"]\",\n        \"rationale\": \"Explain why you chose the label for blue-whitish veil.\"\n    },\n    \"regression structures\": {\n        \"label\": \"choose one from [\"absent\", \"present\"]\",\n        \"rationale\": \"Explain why you chose the label for regression structures.\"\n    }\n}\n<END_OUTPUT>\n\nPlease make sure to choose one label from the list of possible labels for each concept. Provide a rationale for each label based on visual features that can be observed in the image."
        # 初始化概念统计变量
        concept_stats = defaultdict(lambda: {"total": 0, "correct": 0})
        # 初始化概念错误统计 - 记录每个概念的正确标签和错误预测分布
        concept_error_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # 初始化样本预测记录
        sample_predictions = []
    elif args.task_type == "disease":
        print("执行疾病分类任务评估...")
        # 定义疾病分类任务提示
        task_prompt = "You are a professional dermoscopy analysis assistant specialized in feature-level interpretation of skin disease images. Your task is to analyze skin disease images and predict the disease label, based on detailed visual cues.\n\nFor each image, you will provide the following information:\n\n1. **Label**: Choose the predicted label for the disease from the following list: [\"nevus\", \"melanoma\"]. \n\n2. **Positive evidence**: \n   - List the observable features from the image that support the assigned category (either nevus or melanoma). \n   - You should describe specific features such as the type and arrangement of pigment network, presence of streaks, dots and globules, blue-whitish veil, regression structures.\n\n3. **Negative evidence**: \n   - Provide features from the image that rule out the reverse category (i.e., why the disease is not the opposite of the assigned label).\n   - This can include the absence of certain features or the presence of features that are inconsistent with the reverse category.\n\n4. **Summary**: \n   - Provide a concise rationale explaining why the assigned category is correct, based on the diagnostic criteria for nevus and melanoma.\n   - The rationale should be grounded in the features observed in the image.\n\n### Specific requirements:\n- **The possible labels for disease are**: \"nevus\", \"melanoma\".\n- The output MUST strictly follow the format below and include all required fields.\n\nThe expected output format is:\n\n<BEGIN_OUTPUT>\n{\n    \"label\": \"nevus\",   # or \"melanoma\"\n    \"positive evidence\": \"Explain what observable features from the image support the assigned category (nevus or melanoma).\",\n    \"negative evidence\": \"Explain what features from the image rule out the reverse category (why it is not the other disease).\",\n    \"summary\": \"Provide a concise rationale explaining why the assigned category is correct.\"\n}\n<END_OUTPUT>"
        # 初始化疾病统计变量
        disease_stats = {"total": 0, "correct": 0}
        # 初始化疾病错误统计 - 记录正确标签和错误预测分布
        disease_error_stats = defaultdict(lambda: defaultdict(int))
        # 初始化样本预测记录
        sample_predictions = []
    else:
        raise ValueError(f"不支持的任务类型: {args.task_type}. 请使用 'concepts' 或 'disease'.")
    
    # 初始化通用统计变量
    total_samples = len(test_data)
    processed_samples = 0
    failed_samples = 0
    
    # 开始评估
    print(f"开始评估 {total_samples} 个样本...")
    
    for idx, sample in enumerate(tqdm(test_data, desc="评估进度")):
        try:
            # 加载图像 - 直接使用样本中的完整图像路径
            image_path = sample["image"]
            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在: {image_path}")
                failed_samples += 1
                continue
            
            # 打开图像并转换为RGB
            image = Image.open(image_path).convert('RGB')
            
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
            
            # 计算max_new_tokens，
            max_context_length = getattr(model.config, 'max_position_embeddings', 4096)
            max_new_tokens_cap = 2048  # 与原代码保持一致
            max_new_tokens = 2048  # 默认值，与model_worker.py类似
            
            # 计算图像token数量
            num_image_tokens = 0
            if images is not None:
                image_token_count = prompt.count(DEFAULT_IMAGE_TOKEN)
                num_image_tokens = image_token_count * model.get_vision_tower().num_patches
            
            # 限制max_new_tokens，
            max_new_tokens = min(max_new_tokens, max_new_tokens_cap)
            max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)
            
            # 检查是否有足够空间生成token
            if max_new_tokens < 1:
                print(f"警告: 超出最大token长度限制，跳过此样本")
                failed_samples += 1
                continue
            
            temperature = args.temperature  # 从参数获取
            top_p = args.top_p  # 从参数获取
            do_sample = True  # 显式启用采样模式
            stop_str = tokenizer.eos_token if tokenizer.eos_token else "</s>"
            
            # 创建image_args字典，
            image_args = {"images": images} if images is not None else {}
            
            # 生成回复 - 添加异常处理
            try:
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
            except ValueError as e:
                print(f"生成过程中捕获ValueError: {e}")
                failed_samples += 1
                continue
            except torch.cuda.CudaError as e:
                print(f"生成过程中捕获CUDA错误: {e}")
                failed_samples += 1
                continue
            except Exception as e:
                print(f"生成过程中捕获未知错误: {e}")
                failed_samples += 1
                continue
            
            # 使用从streamer获取的文本作为输出
            output = generated_text.strip()
            
            # 根据任务类型处理结果
            if args.task_type == "concepts":
                # 提取预测的概念标签
                predicted_concepts = extract_concept_labels(output)
                
                # 获取真实标签
                ground_truth_concepts = {}
                if "conversations" in sample:
                    for conv in sample["conversations"]:
                        if conv.get("from") == "gpt" and "value" in conv:
                            ground_truth_concepts = extract_concept_labels(conv["value"])
                            break
                
                # 初始化当前样本的预测记录
                sample_pred = {
                    "image": image_path,
                    "concept_predictions": {}
                }
                
                # 计算概念准确率并记录预测结果
                for concept in ["pigment network", "streaks", "dots and globules", "blue-whitish veil", "regression structures"]:
                    concept_stats[concept]["total"] += 1
                    
                    # 初始化当前概念的预测记录
                    concept_pred = {"predicted": None, "ground_truth": None, "correct": False}
                    
                    if concept in predicted_concepts and concept in ground_truth_concepts:
                        pred_label = predicted_concepts[concept]["label"].lower()
                        true_label = ground_truth_concepts[concept]["label"].lower()
                        
                        concept_pred["predicted"] = pred_label
                        concept_pred["ground_truth"] = true_label
                        
                        if pred_label == true_label:
                            concept_stats[concept]["correct"] += 1
                            concept_pred["correct"] = True
                        else:
                            # 记录错误预测分布
                            concept_error_stats[concept][true_label][pred_label] += 1
                    
                    # 添加到样本预测记录
                    sample_pred["concept_predictions"][concept] = concept_pred
                
                # 添加当前样本到预测记录列表
                sample_predictions.append(sample_pred)
            
            elif args.task_type == "disease":
                # 提取预测的疾病标签
                predicted_label = extract_disease_label(output)
                
                # 获取真实标签
                ground_truth_label = None
                if "conversations" in sample:
                    for conv in sample["conversations"]:
                        if conv.get("from") == "gpt" and "value" in conv:
                            ground_truth_label = extract_disease_label(conv["value"])
                            break
                
                # 初始化当前样本的预测记录
                sample_pred = {
                    "image": image_path,
                    "predicted_label": predicted_label,
                    "ground_truth_label": ground_truth_label,
                    "correct": False
                }
                
                # 计算疾病分类准确率
                disease_stats["total"] += 1
                if predicted_label is not None and ground_truth_label is not None:
                    if predicted_label == ground_truth_label:
                        disease_stats["correct"] += 1
                        sample_pred["correct"] = True
                    else:
                        # 记录错误预测分布
                        disease_error_stats[ground_truth_label][predicted_label] += 1
                else:
                    print(f"样本 {idx} 解析失败: 预测={predicted_label}, 真实={ground_truth_label}")
                
                # 添加当前样本到预测记录列表
                sample_predictions.append(sample_pred)
            
            processed_samples += 1
                
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            failed_samples += 1
            continue
    
    # 计算准确率并输出结果
    print("\n评估结果:")
    print(f"总样本数: {total_samples}")
    print(f"成功处理: {processed_samples}")
    print(f"处理失败: {failed_samples}")
    
    # 保存结果字典
    results = {
        "total_samples": total_samples,
        "processed_samples": processed_samples,
        "failed_samples": failed_samples,
        "task_type": args.task_type,
        "parameters": {
            "temperature": temperature,
            "top_p": top_p,
            "lora_path": args.lora_path
        }
    }
    
    if args.task_type == "concepts":
        total_correct = 0
        total_processed = 0
        
        for concept, stats in concept_stats.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                total_correct += stats["correct"]
                total_processed += stats["total"]
                print(f"{concept}: {stats['correct']}/{stats['total']} = {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if total_processed > 0:
            average_accuracy = total_correct / total_processed
            print(f"\n平均准确率: {total_correct}/{total_processed} = {average_accuracy:.4f} ({average_accuracy*100:.2f}%)")
        
        # 添加概念准确率结果
        results.update({
            "concept_accuracies": {concept: {"correct": stats["correct"], "total": stats["total"], 
                                           "accuracy": stats["correct"]/stats["total"] if stats["total"] > 0 else 0}
                                  for concept, stats in concept_stats.items()},
            "average_accuracy": average_accuracy if total_processed > 0 else 0,
            "concept_error_stats": {concept: {true_label: dict(pred_counts) 
                                              for true_label, pred_counts in error_counts.items()}
                                   for concept, error_counts in concept_error_stats.items()},
            "sample_predictions": sample_predictions
        })
    
    elif args.task_type == "disease":
        if disease_stats["total"] > 0:
            accuracy = disease_stats["correct"] / disease_stats["total"]
            print(f"\n疾病分类准确率: {disease_stats['correct']}/{disease_stats['total']} = {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 添加疾病准确率结果
        results.update({
            "disease_accuracy": {
                "correct": disease_stats["correct"],
                "total": disease_stats["total"],
                "accuracy": disease_stats["correct"]/disease_stats["total"] if disease_stats["total"] > 0 else 0
            },
            "disease_error_stats": {true_label: dict(pred_counts) 
                                   for true_label, pred_counts in disease_error_stats.items()},
            "sample_predictions": sample_predictions
        })
    
    # 创建sample_predictions的保存路径
    output_dir = os.path.dirname(args.output_file)
    output_filename = os.path.basename(args.output_file)
    name_without_ext, ext = os.path.splitext(output_filename)
    predictions_file = os.path.join(output_dir, f"{name_without_ext}_predictions{ext}")
    
    # 保存sample_predictions到单独文件
    if 'sample_predictions' in results:
        predictions_data = {
            "total_samples": total_samples,
            "processed_samples": processed_samples,
            "task_type": args.task_type,
            "sample_predictions": results.pop('sample_predictions')
        }
        
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions_data, f, indent=2, ensure_ascii=False)
        
        print(f"样本预测详情已保存至: {predictions_file}")
    
    # 保存主结果（不包含sample_predictions）
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估统计结果已保存至: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估LLaVA-Med模型在皮肤病变分析任务上的性能")
    parser.add_argument("--task-type", type=str, choices=["concepts", "disease"], required=True, 
                        help="评估任务类型: concepts (概念预测) 或 disease (疾病分类)")
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/model/llava-med", help="基础模型路径")
    parser.add_argument("--lora-path", type=str, required=True, help="LoRA权重路径")
    parser.add_argument("--clip-path", type=str, default="/root/autodl-tmp/model/clip-vit-large-patch14-336", help="CLIP模型路径")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备")
    parser.add_argument("--dataset-path", type=str, required=True, help="测试数据集路径")
    parser.add_argument("--output-file", type=str, required=True, help="评估结果输出文件")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度参数")
    parser.add_argument("--top-p", type=float, default=0.9, help="生成top_p参数")
    
    args = parser.parse_args()
    evaluate_model(args)