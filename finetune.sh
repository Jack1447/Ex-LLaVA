#!/bin/bash

# Use bash strict mode
source ~/.bashrc

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.9
export CUDA_LAUNCH_BLOCKING=1


# 检查参数
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <stage>"
    echo "支持的阶段:"
    echo "  concepts  - 概念微调阶段"
    echo "  disease   - 疾病微调阶段"
    echo "  both      - 运行两个阶段"
    echo ""
    echo "示例:"
    echo "  $0 concepts    # 只运行概念微调"
    echo "  $0 disease     # 只运行疾病微调"
    echo "  $0 both        # 运行两个阶段"
    exit 1
fi

STAGE=$1

# 基础配置
BASE_MODEL_PATH="/root/autodl-tmp/model/llava-med"
CLIP_PATH="/root/autodl-tmp/model/clip-vit-large-patch14-336"
IMAGE_FOLDER="/root/autodl-tmp/data/Derm7pt"
BASE_OUTPUT_DIR="/root/autodl-tmp"

# 训练参数 - 优化内存使用
NUM_EPOCHS=10
BATCH_SIZE=1
LEARNING_RATE=5e-5

# 运行微调函数
run_finetune() {
    local stage_name=$1
    local train_json=$2
    local test_json=$3
    local output_dir=$4
    local wandb_project=$5
    local model_path=$6
    
    echo "=========================================="
    echo "开始 ${stage_name} 微调"
    echo "训练数据: ${train_json}"
    echo "测试数据: ${test_json}"
    echo "输出目录: ${output_dir}"
    echo "模型路径: ${model_path}"
    echo "=========================================="
    
    # 创建输出目录
    mkdir -p "${output_dir}"
    
    # 使用单GPU训练避免分布式问题
    python finetune_llava_med.py \
        --model-path "${model_path}" \
        --train-json "${train_json}" \
        --test-json "${test_json}" \
        --image-folder "${IMAGE_FOLDER}" \
        --clip-path "${CLIP_PATH}" \
        --conv-mode "llava_v1" \
        --num-epochs ${NUM_EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --gradient-accumulation-steps 16 \
        --learning-rate ${LEARNING_RATE} \
        --weight-decay 0.01 \
        --gradient-clip-val 1.0 \
        --num-workers 4 \
        --seed 42 \
        --eval-every-n-epochs 1 \
        --use-lora \
        --lora-r 16 \
        --lora-alpha 32 \
        --lora-dropout 0.1 \
        --lora-target-modules q_proj v_proj k_proj o_proj gate_proj up_proj down_proj lm_head \
        --output-dir "${output_dir}" \
        --use-wandb \
        --wandb-project "${wandb_project}" \
        --wandb-name "lora-${stage_name}-$(date +%Y%m%d-%H%M%S)"
}

# 根据阶段执行相应的微调
case $STAGE in
    "concepts")
        run_finetune \
            "概念" \
            "/root/autodl-tmp/data/derm7pt_concepts_train_dataset.json" \
            "/root/autodl-tmp/data/derm7pt_concepts_test_dataset.json" \
            "${BASE_OUTPUT_DIR}/output_concepts" \
            "LLaVA-Med-Derm7pt-Concepts" \
            "${BASE_MODEL_PATH}"
        ;;
    "disease")
        run_finetune \
            "疾病" \
            "/root/autodl-tmp/data/derm7pt_disease_train_dataset.json" \
            "/root/autodl-tmp/data/derm7pt_disease_test_dataset.json" \
            "${BASE_OUTPUT_DIR}/output_disease" \
            "LLaVA-Med-Derm7pt-Disease" \
            "${BASE_MODEL_PATH}"
        ;;
    "both")
        echo "运行两阶段微调..."
        
        # 第一阶段：概念微调
        run_finetune \
            "概念" \
            "/root/autodl-tmp/data/derm7pt_concepts_train_dataset.json" \
            "/root/autodl-tmp/data/derm7pt_concepts_test_dataset.json" \
            "${BASE_OUTPUT_DIR}/output_concepts" \
            "LLaVA-Med-Derm7pt-Concepts" \
            "${BASE_MODEL_PATH}"
        
        # 第二阶段：疾病微调
        echo "开始第二阶段：疾病微调"
        run_finetune \
            "疾病" \
            "/root/autodl-tmp/data/derm7pt_disease_train_dataset.json" \
            "/root/autodl-tmp/data/derm7pt_disease_test_dataset.json" \
            "${BASE_OUTPUT_DIR}/output_disease" \
            "LLaVA-Med-Derm7pt-Disease" \
            "${BASE_MODEL_PATH}"
        ;;
    *)
        echo "错误: 未知的阶段 '$STAGE'"
        echo "支持的阶段: concepts, disease, both"
        exit 1
        ;;
esac

echo "微调完成！"


