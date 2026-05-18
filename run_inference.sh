#!/bin/bash

# 设置默认值
DEFAULT_TASK="concepts"
DEFAULT_CONCEPT_MODEL="/root/autodl-tmp/model/llava-med-concepts"
DEFAULT_DISEASE_MODEL="/root/autodl-tmp/model/llava-med-disease"
DEFAULT_CLIP_PATH="/root/autodl-tmp/model/clip-vit-large-patch14-336"
DEFAULT_DEVICE="cuda"
DEFAULT_TEMPERATURE="0.7"
DEFAULT_TOP_P="0.9"
DEFAULT_OUTPUT_FILE=""

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --task <type>             任务类型: concepts (概念预测), disease (疾病分类) 或 both (依次执行两个模型)"
    echo "  --image-path <path>        单张图片路径（与--folder-path二选一）"
    echo "  --folder-path <path>       包含图片的文件夹路径（与--image-path二选一）"
    echo "  --concept-model <path>     概念阶段模型路径 [默认: $DEFAULT_CONCEPT_MODEL]"
    echo "  --disease-model <path>     疾病阶段模型路径 [默认: $DEFAULT_DISEASE_MODEL]"
    echo "  --clip-path <path>         CLIP模型路径 [默认: $DEFAULT_CLIP_PATH]"
    echo "  --device <device>          运行设备 [默认: $DEFAULT_DEVICE]"
    echo "  --temperature <value>      生成温度参数 [默认: $DEFAULT_TEMPERATURE]"
    echo "  --top-p <value>            生成top_p参数 [默认: $DEFAULT_TOP_P]"
    echo "  --output-file <path>       结果输出文件路径（不指定则自动生成）"
    echo "  -h, --help                 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 单张图片概念预测任务"
    echo "  $0 --task concepts --image-path /path/to/image.jpg"
    echo ""
    echo "  # 单张图片疾病分类任务并保存结果"
    echo "  $0 --task disease --image-path /path/to/image.jpg --output-file result.json"
    echo ""
    echo "  # 单张图片依次运行两个模型"
    echo "  $0 --task both --image-path /path/to/image.jpg"
    echo ""
    echo "  # 处理文件夹中的所有图片（概念预测）"
    echo "  $0 --task concepts --folder-path /path/to/images/"
    echo ""
    echo "  # 处理文件夹中的所有图片（两个模型）"
    echo "  $0 --task both --folder-path /path/to/images/"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --image-path)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --folder-path)
            FOLDER_PATH="$2"
            shift 2
            ;;
        --concept-model)
            CONCEPT_MODEL="$2"
            shift 2
            ;;
        --disease-model)
            DISEASE_MODEL="$2"
            shift 2
            ;;
        --clip-path)
            CLIP_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 设置未指定的参数为默认值
TASK=${TASK:-$DEFAULT_TASK}
CONCEPT_MODEL=${CONCEPT_MODEL:-$DEFAULT_CONCEPT_MODEL}
DISEASE_MODEL=${DISEASE_MODEL:-$DEFAULT_DISEASE_MODEL}
CLIP_PATH=${CLIP_PATH:-$DEFAULT_CLIP_PATH}
DEVICE=${DEVICE:-$DEFAULT_DEVICE}
TEMPERATURE=${TEMPERATURE:-$DEFAULT_TEMPERATURE}
TOP_P=${TOP_P:-$DEFAULT_TOP_P}

# 检查必需参数
if [ -z "$IMAGE_PATH" ] && [ -z "$FOLDER_PATH" ]; then
    echo "错误: 必须指定图片路径 --image-path 或文件夹路径 --folder-path"
    show_help
    exit 1
fi

# 检查图片或文件夹是否存在
if [ ! -z "$IMAGE_PATH" ]; then
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "错误: 图片文件不存在: $IMAGE_PATH"
        exit 1
    fi
    MODE="single"
elif [ ! -z "$FOLDER_PATH" ]; then
    if [ ! -d "$FOLDER_PATH" ]; then
        echo "错误: 文件夹不存在: $FOLDER_PATH"
        exit 1
    fi
    MODE="folder"
fi

# 验证任务类型
if [ "$TASK" != "concepts" ] && [ "$TASK" != "disease" ] && [ "$TASK" != "both" ]; then
    echo "错误: 无效的任务类型: $TASK. 必须是 'concepts', 'disease' 或 'both'"
    show_help
    exit 1
fi

# 构建Python命令
PYTHON_CMD="python /root/LLaVA-Med/inference.py \
    --task $TASK \
    --concept-model $CONCEPT_MODEL \
    --disease-model $DISEASE_MODEL \
    --clip-path $CLIP_PATH \
    --device $DEVICE \
    --temperature $TEMPERATURE \
    --top-p $TOP_P"

# 添加图片路径或文件夹路径参数
if [ "$MODE" == "single" ]; then
    PYTHON_CMD="$PYTHON_CMD \
    --image-path $IMAGE_PATH"
else
    PYTHON_CMD="$PYTHON_CMD \
    --folder-path $FOLDER_PATH"
fi

# 如果指定了输出文件，添加到命令中
if [ ! -z "$OUTPUT_FILE" ]; then
    PYTHON_CMD="$PYTHON_CMD \
    --output-file $OUTPUT_FILE"
fi

# 显示运行信息
echo "========================================"
echo "LLaVA-Med 推理脚本"
echo "========================================"
echo "任务类型: $TASK"
if [ "$MODE" == "single" ]; then
    echo "处理模式: 单张图片"
    echo "图片路径: $IMAGE_PATH"
else
    echo "处理模式: 文件夹遍历"
    echo "文件夹路径: $FOLDER_PATH"
fi
if [ "$TASK" == "concepts" ]; then
    echo "使用模型: $CONCEPT_MODEL"
elif [ "$TASK" == "disease" ]; then
    echo "使用模型: $DISEASE_MODEL"
elif [ "$TASK" == "both" ]; then
    echo "使用模型: $CONCEPT_MODEL 和 $DISEASE_MODEL"
fi
echo "CLIP模型: $CLIP_PATH"
echo "运行设备: $DEVICE"
echo "温度参数: $TEMPERATURE"
echo "Top-P参数: $TOP_P"
if [ ! -z "$OUTPUT_FILE" ]; then
    echo "输出文件: $OUTPUT_FILE"
else
    if [ "$MODE" == "single" ]; then
        # 获取图片ID（文件名无后缀）用于显示预期的输出文件名
        IMAGE_FILENAME=$(basename "$IMAGE_PATH")
        IMAGE_ID=${IMAGE_FILENAME%.*}
        echo "输出文件: (自动生成: inference_single_${TASK}_${IMAGE_ID}.json)"
    else
        echo "输出文件: (自动生成: inference_${TASK}.json)"
    fi
fi
echo "========================================"
echo

# 执行Python脚本
exec $PYTHON_CMD