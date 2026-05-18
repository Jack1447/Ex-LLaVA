#!/bin/bash

# 评估脚本 - 支持概念预测和疾病分类任务

echo "====================================="
echo "LLaVA-Med 评估脚本"
echo "支持概念预测(concepts)和疾病分类(disease)任务"
echo "====================================="

# 默认参数设置
DEFAULT_MODEL_PATH="/root/autodl-tmp/model/llava-med"
DEFAULT_CLIP_PATH="/root/autodl-tmp/model/clip-vit-large-patch14-336"
DEFAULT_DEVICE="cuda"
DEFAULT_TEMPERATURE="0.7"
DEFAULT_TOP_P="0.9"

# 任务特定默认参数
DEFAULT_CONCEPTS_LORA="/root/autodl-tmp/output_concepts/epoch6"
DEFAULT_CONCEPTS_DATASET="/root/autodl-tmp/data/derm7pt_concepts_test_dataset.json"
DEFAULT_CONCEPTS_OUTPUT="/root/autodl-tmp/concept_evaluation_results.json"

DEFAULT_DISEASE_LORA="/root/autodl-tmp/output_disease/epoch3"
DEFAULT_DISEASE_DATASET="/root/autodl-tmp/data/derm7pt_disease_test_dataset.json"
DEFAULT_DISEASE_OUTPUT="/root/autodl-tmp/disease_evaluation_results.json"

# 显示帮助信息
show_help() {
    echo "Usage: $0 [task_type] [options]"
    echo ""
    echo "位置参数:"
    echo "  task_type                   评估任务类型: concepts (概念预测) 或 disease (疾病分类)"
    echo ""
    echo "选项:"
    echo "  -t, --task <task_type>      评估任务类型 (与位置参数功能相同): concepts (概念预测) 或 disease (疾病分类)"
    echo "  -l, --lora <path>           LoRA权重路径"
    echo "  -d, --dataset <path>        测试数据集路径"
    echo "  -o, --output <path>         评估结果输出文件路径"
    echo "  -m, --model <path>          基础模型路径 (默认: $DEFAULT_MODEL_PATH)"
    echo "  -c, --clip <path>           CLIP模型路径 (默认: $DEFAULT_CLIP_PATH)"
    echo "  -g, --device <device>       运行设备 (默认: $DEFAULT_DEVICE)"
    echo "  --temp <value>              生成温度参数 (默认: $DEFAULT_TEMPERATURE)"
    echo "  --top-p <value>             生成top_p参数 (默认: $DEFAULT_TOP_P)"
    echo "  -h, --help                  显示帮助信息"
    echo ""
    echo "Examples:"
    echo "  # 简单方式运行概念预测任务评估"
    echo "  $0 concepts"
    echo ""
    echo "  # 简单方式运行疾病分类任务评估"
    echo "  $0 disease"
    echo ""
    echo "  # 详细方式运行概念预测任务评估"
    echo "  $0 concepts --lora /root/autodl-tmp/output_concepts/epoch6 \
      --dataset /root/autodl-tmp/data/derm7pt_concepts_test_dataset.json \
      --output /root/autodl-tmp/concept_evaluation_results.json"
    echo ""
    echo "  # 详细方式运行疾病分类任务评估"
    echo "  $0 disease --lora /root/autodl-tmp/output_disease/epoch3 \
      --dataset /root/autodl-tmp/data/derm7pt_disease_test_dataset.json \
      --output /root/autodl-tmp/disease_evaluation_results.json"
    echo ""
}

# 解析命令行参数
# 首先检查位置参数是否为任务类型
if [[ $1 == "concepts" || $1 == "disease" ]]; then
    TASK_TYPE="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--task)
            TASK_TYPE="$2"
            shift 2
            ;;
        -l|--lora)
            LORA_PATH="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -c|--clip)
            CLIP_PATH="$2"
            shift 2
            ;;
        -g|--device)
            DEVICE="$2"
            shift 2
            ;;
        --temp)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "错误: 未知参数 $1"
            show_help
            exit 1
            ;;
    esac
done

# 根据任务类型设置默认参数
if [[ "$TASK_TYPE" == "concepts" ]]; then
    LORA_PATH=${LORA_PATH:-$DEFAULT_CONCEPTS_LORA}
    DATASET_PATH=${DATASET_PATH:-$DEFAULT_CONCEPTS_DATASET}
    OUTPUT_FILE=${OUTPUT_FILE:-$DEFAULT_CONCEPTS_OUTPUT}
elif [[ "$TASK_TYPE" == "disease" ]]; then
    LORA_PATH=${LORA_PATH:-$DEFAULT_DISEASE_LORA}
    DATASET_PATH=${DATASET_PATH:-$DEFAULT_DISEASE_DATASET}
    OUTPUT_FILE=${OUTPUT_FILE:-$DEFAULT_DISEASE_OUTPUT}
fi

# 检查必需参数
if [ -z "$TASK_TYPE" ]; then
    echo "错误: 必须指定任务类型 (可作为位置参数或使用 --task)"
    show_help
    exit 1
fi

# 使用默认值设置未提供的参数
MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL_PATH}
CLIP_PATH=${CLIP_PATH:-$DEFAULT_CLIP_PATH}
DEVICE=${DEVICE:-$DEFAULT_DEVICE}
TEMPERATURE=${TEMPERATURE:-$DEFAULT_TEMPERATURE}
TOP_P=${TOP_P:-$DEFAULT_TOP_P}

# 验证任务类型
if [[ "$TASK_TYPE" != "concepts" && "$TASK_TYPE" != "disease" ]]; then
    echo "错误: 无效的任务类型 '$TASK_TYPE'. 任务类型必须是 'concepts' 或 'disease'"
    show_help
    exit 1
fi

# 验证文件路径
echo "正在验证文件路径..."

if [ ! -f "$DATASET_PATH" ]; then
    echo "错误: 数据集文件不存在: $DATASET_PATH"
    exit 1
fi

if [ ! -d "$LORA_PATH" ]; then
    echo "错误: LoRA权重目录不存在: $LORA_PATH"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 基础模型目录不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$CLIP_PATH" ]; then
    echo "错误: CLIP模型目录不存在: $CLIP_PATH"
    exit 1
fi

# 确保输出目录存在
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "警告: 输出目录不存在，将创建: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# 显示配置信息
echo ""
echo "配置信息:"
echo "任务类型: $TASK_TYPE"
echo "基础模型: $MODEL_PATH"
echo "LoRA权重: $LORA_PATH"
echo "CLIP模型: $CLIP_PATH"
echo "数据集: $DATASET_PATH"
echo "输出文件: $OUTPUT_FILE"
echo "运行设备: $DEVICE"
echo "生成参数: temperature=$TEMPERATURE, top_p=$TOP_P"
echo ""

# 运行评估
echo "开始运行评估..."
echo "====================================="

python /root/LLaVA-Med/evaluate.py \
    --task-type "$TASK_TYPE" \
    --model-path "$MODEL_PATH" \
    --lora-path "$LORA_PATH" \
    --clip-path "$CLIP_PATH" \
    --device "$DEVICE" \
    --dataset-path "$DATASET_PATH" \
    --output-file "$OUTPUT_FILE" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P"

# 检查执行状态
if [ $? -eq 0 ]; then
    echo "====================================="
    echo "✓ 评估完成！"
    echo "结果已保存至: $OUTPUT_FILE"
    echo ""
    echo "要查看结果，可使用:"
    echo "cat $OUTPUT_FILE"
else
    echo "====================================="
    echo "✗ 评估失败！"
    exit 1
fi