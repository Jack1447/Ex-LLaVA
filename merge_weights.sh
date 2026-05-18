#!/bin/bash

# 设置错误时退出
set -e

echo "======================================="
echo "开始合并LoRA权重到基础模型"
echo "======================================="

# 设置路径变量
BASE_MODEL_PATH="/root/autodl-tmp/model/llava-med"
CONCEPTS_LORA_PATH="/root/autodl-tmp/output_concepts/epoch6"
DISEASE_LORA_PATH="/root/autodl-tmp/output_disease/epoch3"
CONCEPTS_OUTPUT_PATH="/root/autodl-tmp/model/llava-med-concepts"
DISEASE_OUTPUT_PATH="/root/autodl-tmp/model/llava-med-disease"

# 确保输出目录存在
mkdir -p "$(dirname "$CONCEPTS_OUTPUT_PATH")"
mkdir -p "$(dirname "$DISEASE_OUTPUT_PATH")"

echo "\n1. 合并概念阶段的LoRA权重..."
echo "基础模型: $BASE_MODEL_PATH"
echo "LoRA权重: $CONCEPTS_LORA_PATH"
echo "输出路径: $CONCEPTS_OUTPUT_PATH"

python /root/LLaVA-Med/merge_lora_weights.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --lora-path "$CONCEPTS_LORA_PATH" \
    --output-path "$CONCEPTS_OUTPUT_PATH" \
    --clip-path /root/autodl-tmp/model/clip-vit-large-patch14-336

echo "\n2. 合并疾病阶段的LoRA权重..."
echo "基础模型: $BASE_MODEL_PATH"
echo "LoRA权重: $DISEASE_LORA_PATH"
echo "输出路径: $DISEASE_OUTPUT_PATH"

python /root/LLaVA-Med/merge_lora_weights.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --lora-path "$DISEASE_LORA_PATH" \
    --output-path "$DISEASE_OUTPUT_PATH" \
    --clip-path /root/autodl-tmp/model/clip-vit-large-patch14-336

echo "\n======================================="
echo "权重合并完成！"
echo "概念阶段模型: $CONCEPTS_OUTPUT_PATH"
echo "疾病阶段模型: $DISEASE_OUTPUT_PATH"
echo "======================================="