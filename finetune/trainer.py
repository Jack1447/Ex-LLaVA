"""
Training functions for finetune
"""
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import logging

from .dataset import Derm7ptDataset, collate_fn
from .evaluation import evaluate

logger = logging.getLogger(__name__)


def train_model(args, model, tokenizer, image_processor, conv_template, train_dataloader, test_dataloader):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 余弦退火学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.num_epochs * len(train_dataloader))
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化 wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=args
        )

    # 开始训练
    model.train()
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            images = batch['images'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)

            # 确保图像数据类型与模型一致
            if hasattr(model, 'dtype'):
                images = images.to(dtype=model.dtype)
            elif hasattr(model, 'config') and hasattr(model.config, 'torch_dtype'):
                images = images.to(dtype=model.config.torch_dtype)
            
            # 强制保证input_ids有batch维度
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            
            fw_args = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=images
            )
            
            # 前向与损失计算
            outputs = model(**fw_args)
            loss = outputs.loss
            
            # 梯度累积：将损失除以累积步数以保持等效学习率
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            # 只有在指定的累积步数或最后一个批次时才更新参数
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                if args.gradient_clip_val and args.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item() * args.gradient_accumulation_steps)
            progress_bar.set_postfix({'loss': f"{(loss.item() * args.gradient_accumulation_steps):.4f}",
                                      'avg_loss': f"{(total_loss/(step+1)):.4f}"})
            
            # 记录 wandb
            if args.use_wandb and (step + 1) % args.gradient_accumulation_steps == 0:
                current_step = epoch * len(train_dataloader) + step
                wandb.log({
                    "train/loss": loss.item() * args.gradient_accumulation_steps,
                    "train/avg_loss": total_loss/(step+1),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/epoch": epoch + (step + 1) / len(train_dataloader),
                    "train/step": current_step
                })

        epoch_loss = total_loss / max(1, len(train_dataloader))
        logger.info(f"Epoch {epoch+1} train loss: {epoch_loss:.6f}")

        # 评估 - 只计算 test loss
        if (epoch + 1) % args.eval_every_n_epochs == 0 or (epoch + 1) == args.num_epochs:
            logger.info("开始测试集评估...")
            test_loss = evaluate(model, test_dataloader, device, tokenizer, "loss")
            logger.info(f"Epoch {epoch+1} test loss: {test_loss:.6f}")
            
            # 记录到wandb
            if args.use_wandb:
                wandb.log({
                    "test/loss": test_loss,
                    "test/epoch": epoch + 1
                })

        # 每个 epoch 保存
        checkpoint_path = os.path.join(args.output_dir, f"epoch{epoch+1}")
        if args.use_lora:
            model.save_pretrained(checkpoint_path)
        else:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, "model_state_dict.bin"))
        logger.info(f"Saved model after epoch {epoch+1} to {checkpoint_path}")

    # 最终保存
    if args.use_lora:
        lora_path = os.path.join(args.output_dir, "lora")
        model.save_pretrained(lora_path)
        logger.info(f"Training completed. Final LoRA model saved to {lora_path}")
    else:
        model_path = os.path.join(args.output_dir, "model_state_dict.bin")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Training completed. Final model saved to {model_path}")
    
    # 结束 wandb
    if args.use_wandb:
        wandb.finish()
