#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并LoRA权重到基础模型的脚本
"""
import argparse
import os
import torch
import logging
from peft import PeftModel
from transformers import AutoTokenizer

# LLaVA imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_lora_weights(base_model_path, lora_path, output_path, clip_path=None):
    """
    合并LoRA权重到基础模型
    
    Args:
        base_model_path: 基础模型路径
        lora_path: LoRA权重路径
        output_path: 输出合并后模型的路径
        clip_path: 可选的CLIP模型路径
    """
    logger.info(f"开始合并权重...")
    logger.info(f"基础模型: {base_model_path}")
    logger.info(f"LoRA权重: {lora_path}")
    logger.info(f"输出路径: {output_path}")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    try:
        # 加载基础模型及组件
        logger.info(f"正在加载基础模型...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            base_model_path,
            None,  # model_base
            get_model_name_from_path(base_model_path),
            load_8bit=False,
            load_4bit=False,
            device_map='auto',
            clip_path=clip_path
        )
        
        # 加载LoRA权重
        logger.info(f"正在加载LoRA权重...")
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            device_map='auto'
        )
        
        # 合并权重
        logger.info(f"正在合并权重...")
        model = model.merge_and_unload()
        
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 保存合并后的模型
        logger.info(f"正在保存合并后的模型到 {output_path}...")
        
        # 保存模型
        model.save_pretrained(output_path, safe_serialization=True)
        
        # 保存tokenizer
        tokenizer.save_pretrained(output_path)
        
        # 保存image_processor
        image_processor.save_pretrained(output_path)
        
        logger.info(f"权重合并完成！")
        logger.info(f"合并后的模型保存在: {output_path}")
        
    except Exception as e:
        logger.error(f"合并过程中发生错误: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="合并LoRA权重到基础模型")
    
    parser.add_argument("--base-model-path", type=str, default="/root/autodl-tmp/model/llava-med",
                        help="基础模型的路径")
    parser.add_argument("--lora-path", type=str, required=True,
                        help="LoRA权重的路径")
    parser.add_argument("--output-path", type=str, required=True,
                        help="输出合并后模型的路径")
    parser.add_argument("--clip-path", type=str, default="",
                        help="可选的CLIP模型路径")
    
    args = parser.parse_args()
    
    merge_lora_weights(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        output_path=args.output_path,
        clip_path=args.clip_path if args.clip_path else None
    )

if __name__ == "__main__":
    main()