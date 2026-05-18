"""
Evaluation utilities for finetune - 完全仿照 model_worker.py 的方式
"""
import torch



def _align_images_dtype(model, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, 'dtype'):
        return images.to(dtype=model.dtype)
    if hasattr(model, 'config') and hasattr(model.config, 'torch_dtype'):
        return images.to(dtype=model.config.torch_dtype)
    return images


@torch.inference_mode()
def _compute_batch_loss(model, input_ids, attention_mask, labels, images) -> float:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        images=images,
    )
    return float(outputs.loss.item())




@torch.inference_mode()
def evaluate(model, dataloader, device, tokenizer, mode: str) -> float:
    """在测试集上评估：只计算 loss。"""
    import logging
    logger = logging.getLogger(__name__)
    
    model_was_training = model.training
    try:
        model.eval()
    except Exception:
        pass

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        try:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            images = batch['images'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)

            images = _align_images_dtype(model, images)

            batch_loss = _compute_batch_loss(model, input_ids, attention_mask, labels, images)
            total_loss += batch_loss
            num_batches += 1

        except Exception as e:
            logger.error(f"Failed to process batch {batch_idx + 1}: {e}")
            raise

    avg_loss = total_loss / max(1, num_batches)

    if model_was_training:
        try:
            model.train()
        except Exception:
            pass

    return avg_loss

