"""
Dataset classes for finetune
"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images

logger = logging.getLogger(__name__)


class Derm7ptDataset(Dataset):
    """Derm7pt dataset for finetune"""
    
    def __init__(self, json_file, image_folder, tokenizer, conv_template, image_processor, model_config):
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.image_processor = image_processor
        self.model_config = model_config

        logger.info(f"Loading data from {json_file}")
        with open(json_file, 'r') as f:
            dataset = json.load(f)

        self.data = dataset['data']
        self.global_config = dataset.get('global', {})
        self.system_prompt = self.global_config.get('role_system_prompt', '')

        logger.info(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # --- image ---
        image_path = sample['image']
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.image_folder, image_path)

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (336, 336), color='white')

        # optional
        orig_w, orig_h = image.size

        # 将图片转换成张量，包括预处理的操作
        try: 
            # 使用 LLaVA-Med 提供的工具函数  填充图像并预处理
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        except Exception as e:
            logger.warning(f"process_images() failed: {e}; fallback to image_processor.preprocess with padding=True")
            # 直接用 CLIP 的 preprocess
            # 指定 padding=True，保证 batchable
            fallback = self.image_processor.preprocess(
                [image],                      # batched list
                return_tensors='pt',
                padding=True                  # padding 与上面的 process_images 保持一致
            )
            image_tensor = fallback["pixel_values"][0]

        # --- text ---
        conversations = sample['conversations']
        human_message = ''
        gpt_message = ''
        for conv in conversations:
            if conv['from'] == 'human':
                human_message = conv['value']
            elif conv['from'] == 'gpt':
                gpt_message = conv['value']

        question = human_message.replace('<image>', '').strip()

        answer = gpt_message
        if '<BEGIN_OUTPUT>' in answer and '<END_OUTPUT>' in answer:
            answer = answer.split('<BEGIN_OUTPUT>')[1].split('<END_OUTPUT>')[0].strip()

        conv = self.conv_template.copy()
        if getattr(conv, "system", "") != "":
            conv.system = self.system_prompt

        if getattr(self.model_config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()

        # 将 prompt 文本中的 <image> 标记替换为 IMAGE_TOKEN_INDEX，并返回一个处理后的 token ID 一维张量
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        # 调试信息
        if input_ids is None:
            logger.error(f"tokenizer_image_token returned None for prompt: {prompt[:100]}...")
            # 创建一个空的input_ids作为fallback
            input_ids = torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long)
        
        # 转化为二维
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # build labels
        conv_question = self.conv_template.copy()
        if getattr(conv_question, "system", "") != "":
            conv_question.system = self.system_prompt
        conv_question.append_message(conv_question.roles[0], qs)
        conv_question.append_message(conv_question.roles[1], None)
        prompt_question = conv_question.get_prompt()
        input_ids_question = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        # 添加unsqueeze(0)将一维张量转为二维
        if input_ids_question.dim() == 1:
            input_ids_question = input_ids_question.unsqueeze(0)
        question_len = input_ids_question.shape[1]

        labels = input_ids.clone()
        # Ensure labels is 2D tensor for slicing operation
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        labels[:, :question_len] = -100

        # 从对话中提取真实标签
        ground_truth_concepts = None
        ground_truth_disease = None
        
        # 提取概念真实标签
        for conv in conversations:
            if conv['from'] == 'gpt' and '<BEGIN_OUTPUT>' in conv['value']:
                # 提取JSON格式的真实标签
                json_text = conv['value'].split('<BEGIN_OUTPUT>')[1].split('<END_OUTPUT>')[0].strip()
                try:
                    import json
                    concept_data = json.loads(json_text)
                    ground_truth_concepts = {}
                    for concept in ['pigment network', 'streaks', 'dots and globules', 'blue-whitish veil', 'regression structures']:
                        if concept in concept_data and 'label' in concept_data[concept]:
                            ground_truth_concepts[concept] = concept_data[concept]['label']
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Failed to parse concept labels: {e}")
                    ground_truth_concepts = None
                break
        
        # 提取疾病真实标签
        for conv in conversations:
            if conv['from'] == 'gpt' and 'label' in conv['value'] and 'positive evidence' in conv['value']:
                # 这是疾病预测的对话
                json_text = conv['value'].split('<BEGIN_OUTPUT>')[1].split('<END_OUTPUT>')[0].strip()
                try:
                    import json
                    disease_data = json.loads(json_text)
                    if 'label' in disease_data:
                        ground_truth_disease = disease_data['label']
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Failed to parse disease label: {e}")
                    ground_truth_disease = None
                break

        return {
            'input_ids': input_ids.squeeze(0),  # 保持1D张量
            'labels': labels.squeeze(0),        # 保持1D张量
            'images': image_tensor,
            'ground_truth_concepts': ground_truth_concepts,
            'ground_truth_disease': ground_truth_disease,
            'pad_token_id': self.tokenizer.pad_token_id,
        }


def collate_fn(batch):
    """保证批次中的所有的 input_ids labels 有相同的长度"""
    pad_token_id = batch[0].get('pad_token_id', 0)
    
    # 调试信息
    logger.debug(f"Collate_fn: batch size = {len(batch)}")
    for i, item in enumerate(batch):
        logger.debug(f"Item {i}: input_ids shape = {item['input_ids'].shape if item.get('input_ids') is not None else 'None'}")

    max_len = max(item['input_ids'].shape[0] for item in batch)
    B = len(batch)

    # 初始化
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    images = torch.stack([item['images'] for item in batch])

    for i, item in enumerate(batch):
        L = item['input_ids'].shape[0]
        input_ids[i, :L] = item['input_ids']
        labels[i, :L] = item['labels']
        attention_mask[i, :L] = 1

    # 收集真实标签
    ground_truth_concepts = [item.get('ground_truth_concepts', None) for item in batch]
    ground_truth_disease = [item.get('ground_truth_disease', None) for item in batch]

    batch_out = {
        'input_ids': input_ids,
        'labels': labels,
        'images': images,
        'attention_mask': attention_mask,
        'ground_truth_concepts': ground_truth_concepts,
        'ground_truth_disease': ground_truth_disease
    }

    return batch_out
