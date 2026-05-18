"""
Main training script for LLaVA-Med finetune
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
import logging
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# LLaVA imports
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from finetune.dataset import Derm7ptDataset, collate_fn
from finetune.trainer import train_model

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup wandb
os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"


def setup_model(args):
    """Setup model, tokenizer, and image processor"""
    logger.info(f"Loading model from {args.model_path}")
    
    # 清理GPU内存
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # 如果指定了本地CLIP路径，记录日志
    if args.clip_path:
        logger.info(f"Using local CLIP model from: {args.clip_path}")

    # 加载预训练模型及组件
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        get_model_name_from_path(args.model_path),
        args.load_8bit,
        args.load_4bit,
        device_map='auto',
        clip_path=args.clip_path
    )

    # 设置 padding 方式
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.config.use_cache = False
    # model.gradient_checkpointing_enable()
    
    # 如果不使用LoRA，则需要设置模型参数的requires_grad
    if not args.use_lora:
        for param in model.parameters():
            param.requires_grad = True

    # LoRA
    if args.use_lora:
        logger.info("Preparing model for LoRA training")
        if args.load_4bit or args.load_8bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM'
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer, image_processor, context_len


def setup_data(args, tokenizer, image_processor, model_config):
    """Setup datasets and dataloaders"""
    conv_template = conv_templates[args.conv_mode]
    
    # 训练数据集
    train_dataset = Derm7ptDataset(
        args.train_json,
        args.image_folder,
        tokenizer,
        conv_template,
        image_processor,
        model_config
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    # 测试数据集
    test_dataset = Derm7ptDataset(
        args.test_json,
        args.image_folder,
        tokenizer,
        conv_template,
        image_processor,
        model_config
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    return train_dataloader, test_dataloader, conv_template


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA-Med model on Derm7pt dataset")

    # model
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/model/llava-med")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--clip-path", type=str, default="", help="Path to local CLIP model directory")

    # data
    parser.add_argument("--train-json", type=str, default="/root/autodl-tmp/data/derm7pt_concepts_train_dataset.json")
    parser.add_argument("--test-json", type=str, default="/root/autodl-tmp/data/derm7pt_concepts_test_dataset.json")
    parser.add_argument("--image-folder", type=str, default="/root/autodl-tmp/data/Derm7pt")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")

    # train
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every-n-epochs", type=int, default=1, help="Evaluate on test set every N epochs")

    # LoRA
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-target-modules", type=str, nargs="+",
                        default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj","lm_head"])

    # output
    parser.add_argument("--output-dir", type=str, default="/root/autodl-tmp/output_concepts")
    
    # wandb
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="LLaVA-Med-finetune", help="Wandb project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="Wandb run name")

    args = parser.parse_args()

    # Set seed
    from transformers import set_seed
    set_seed(args.seed)

    disable_torch_init()

    # Setup model
    model, tokenizer, image_processor, context_len = setup_model(args)

    # Setup data
    train_dataloader, test_dataloader, conv_template = setup_data(args, tokenizer, image_processor, model.config)

    # Start training
    train_model(args, model, tokenizer, image_processor, conv_template, train_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
