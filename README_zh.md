[English](README.md) | 中文

---

# Ex-LLaVA: 基于大语言模型的可解释医学图像分析

我们设计了一个基于大语言模型的医学图像分析系统，专注于皮肤疾病诊断，特别是痣(nevus)和黑色素瘤(melanoma)的区分。本项目实现了一个创新的两阶段训练框架，首先进行皮肤病变概念预测（提取关键临床特征），然后基于这些特征进行疾病分类，模拟皮肤科医生的诊断思路。

## 项目概览

![architecture](asset/architecture.png)

## 安装环境

```
pip install torch==2.1.0+cu121 torchaudio==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
    
pip install -r requirements.txt
```

------

## 目录结构

```
LLaVA-Med/
├── evaluate.py          # 模型评估脚本
├── evaluate.sh          # 评估执行脚本
├── finetune/            # 微调相关模块
│   ├── constants.py     # 常量定义
│   ├── dataset.py       # 数据集处理
│   ├── evaluation.py    # 评估工具
│   └── trainer.py       # 训练器实现
├── finetune.sh          # 微调执行脚本
├── finetune_llava_med.py # 微调主入口
├── inference.py         # 推理脚本
├── llava/               # 核心模型代码
│   ├── model/           # 模型架构实现
│   ├── serve/           # 服务相关代码
│   └── utils.py         # 工具函数
├── merge_lora_weights.py # LoRA权重合并脚本
├── merge_weights.sh     # 权重合并执行脚本
└── run_inference.sh     # 推理执行脚本
```

## 数据集

### 图像数据
- 位置：`/root/autodl-tmp/data/Derm7pt`
- 包含Derm7pt皮肤病变数据集的图像文件

### 概念阶段数据
- 训练集：`/root/autodl-tmp/data/derm7pt_concepts_train_dataset.json`
- 测试集：`/root/autodl-tmp/data/derm7pt_concepts_test_dataset.json`

```
clinical_concepts_mapping={
    "pigment network": {0: "absent", 1: "typical", 2: "atypical"},
    "streaks": {0: "absent", 1: "regular", 2: "irregular"},
    "dots and globules": {0: "absent", 1: "regular", 2: "irregular"},
    "blue-whitish veil": {0: "absent", 1: "present"},
    "regression structures": {0: "absent", 1: "present"},
}
```

#### 数据格式
```json
{
    "image": "/root/autodl-tmp/data/Derm7pt/{image_id}.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nAccording to this picture of skin disease, provide labels and rationales for five clinical concepts."
        },
        {
            "from": "gpt",
            "value": gpt_output
        }
    ],
    "meta": {
        "image_id": image_id,
        "split": split_name
    }
}
```

#### 概念输出格式
```json
<BEGIN_OUTPUT>
{
    "pigment network": {
        "label": "choose one from [\"absent\", \"typical\", \"atypical\"]",
        "rationale": "Explain why you chose the label for pigment network."
    },
    "streaks": {
        "label": "choose one from [\"absent\", \"regular\", \"irregular\"]",
        "rationale": "Explain why you chose the label for streaks."
    },
    "dots and globules": {
        "label": "choose one from [\"absent\", \"regular\", \"irregular\"]",
        "rationale": "Explain why you chose the label for dots and globules."
    },
    "blue-whitish veil": {
        "label": "choose one from [\"absent\", \"present\"]",
        "rationale": "Explain why you chose the label for blue-whitish veil."
    },
    "regression structures": {
        "label": "choose one from [\"absent\", \"present\"]",
        "rationale": "Explain why you chose the label for regression structures."
    }
}
<END_OUTPUT>
```

### 疾病阶段数据
- 训练集：`/root/autodl-tmp/data/derm7pt_concepts_train_dataset.json`
- 测试集：`/root/autodl-tmp/data/derm7pt_disease_test_dataset.json`

```
clinical_class_mapping={0: "nevus", 1: "melanoma"}
```

#### 数据格式
```json
{
    "image": "/root/autodl-tmp/data/Derm7pt/{image_id}.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nAccording to this picture of skin disease,provide [label, positive evidence,negative evidence,summary].Here are the concepts and their respective rationales:{information}."
        },
        {
            "from": "gpt",
            "value": gpt_output_parts
        }
    ],
    "meta": {
        "image_id": image_id,
        "split": split_name
    }
}
```

#### 疾病输出格式
```json
<BEGIN_OUTPUT>
{
    "label": "nevus",   # or "melanoma"
    "positive evidence": "Explain what observable features from the image support the assigned category (nevus or melanoma).",
    "negative evidence": "Explain what features from the image rule out the reverse category (why it is not the other disease).",
    "summary": "Provide a concise rationale explaining why the assigned category is correct."
}
<END_OUTPUT>
```

## 模型架构

### 系统组件
- **视觉编码器**：CLIP ViT-Large (patch14-336)，用于提取医学图像特征
- **语言模型**：经过LoRA微调得到的llava-med-concepts和llava-med-disease，处理文本和视觉特征
- **连接层**：将视觉特征投影到语言模型的嵌入空间

### 基础模型
- CLIP模型：`/root/autodl-tmp/model/clip-vit-large-patch14-336`

### 微调后模型
- 概念阶段模型：`/root/autodl-tmp/model/llava-med-concepts`
- 疾病阶段模型：`/root/autodl-tmp/model/llava-med-disease`

## 使用方法

### 1. 模型微调

#### 执行微调

```bash
cd /root/LLaVA-Med

# 概念阶段微调（学习皮肤病变特征）
bash finetune.sh concepts

# 疾病阶段微调（学习疾病分类）
bash finetune.sh disease

# 运行两个阶段的微调（概念+疾病）
bash finetune.sh both
```

#### 自定义微调参数

微调脚本支持多种参数配置，可以通过修改finetune.sh文件中的变量进行调整：

- **数据路径**：
  - 基础模型：`/root/autodl-tmp/model/llava-med`
  - CLIP模型：`/root/autodl-tmp/model/clip-vit-large-patch14-336`
  - 图像文件夹：`/root/autodl-tmp/data/Derm7pt`
  - 训练/测试数据：自动根据任务类型选择对应的JSON文件

#### 微调输出

微调完成后，每个阶段的LoRA权重会保存在以下位置：
- 概念阶段：`/root/autodl-tmp/output_concepts`
- 疾病阶段：`/root/autodl-tmp/output_disease`

每个输出目录包含按轮次组织的权重文件，如epoch1, epoch2等，便于选择最佳模型进行评估。

### 2. 权重合并

将LoRA权重与基础模型合并，便于部署使用：

```bash
cd /root/LLaVA-Med
bash merge_weights.sh
```

此脚本会自动合并两个阶段的LoRA权重，并保存到以下路径：
- 概念阶段模型：`/root/autodl-tmp/model/llava-med-concepts`
- 疾病阶段模型：`/root/autodl-tmp/model/llava-med-disease`

### 3. 模型评估

我们提供了全面的模型评估功能，支持对概念预测和疾病分类两个阶段的模型进行详细评估。评估过程会计算各种性能指标，并将结果保存为结构化的JSON文件，方便进一步分析。

#### 评估功能概述

- **概念预测评估**：计算5个皮肤病变特征的预测准确率，包括：
  - 色素网络(pigment network)
  - 条纹(streaks)
  - 点和球(dots and globules)
  - 蓝白色面纱(blue-whitish veil)
  - 退行性结构(regression structures)
- **疾病分类评估**：计算良恶性肿瘤(nevus/melanoma)分类的准确率

#### 执行评估

```bash
cd /root/LLaVA-Med
# 评估概念预测模型
bash evaluate.sh --task concepts --model-path /root/autodl-tmp/model/llava-med-concepts

# 评估疾病分类模型
bash evaluate.sh --task disease --model-path /root/autodl-tmp/model/llava-med-disease
```

#### 自定义评估参数

评估脚本支持多种参数自定义，可以根据需要调整：

```bash
# 使用完整参数进行评估
bash evaluate.sh --task concepts \
  --lora /path/to/lora/weights \
  --dataset /path/to/test/dataset.json \
  --output /path/to/output/results.json \
  --model /path/to/base/model \
  --clip /path/to/clip/model \
  --device cuda \
  --temp 0.7 \
  --top-p 0.9
```

#### 评估结果说明

评估完成后，将生成两个JSON文件：

1. **主结果文件**（如`concept_evaluation_results.json`或`disease_evaluation_results.json`）：
   - 总样本数、成功处理数和失败数统计
   - 评估参数记录（温度、top_p、LoRA路径等）
   - 各概念/疾病的准确率数据
   - 错误预测分布统计

2. **预测详情文件**（如`concept_evaluation_results_predictions.json`）：
   - 每个测试样本的详细预测结果
   - 预测标签与真实标签的对比
   - 预测正确性标记

### 4. 图像推理

LLaVA-Med提供了便捷的命令行推理工具，支持单张图像分析和文件夹批量处理：

#### 单张图像概念预测
```bash
cd /root/LLaVA-Med
./run_inference.sh --task concepts --image-path /path/to/image.jpg
```

#### 单张图像疾病分类
```bash
cd /root/LLaVA-Med
./run_inference.sh --task disease --image-path /path/to/image.jpg
```

#### 单张图像全流程分析（概念+疾病）
```bash
cd /root/LLaVA-Med
./run_inference.sh --task both --image-path /path/to/image.jpg
```

**注意：** 在both阶段，疾病分类模型会利用概念阶段的分析结果作为额外输入。系统会自动将概念阶段生成的皮肤病变特征及其解释（包括色素网络、条纹、点和球、蓝白色面纱、退行性结构等）整合到疾病分类的提示词中，以提升疾病诊断的准确性。具体来说，系统会在疾病分类提示中添加："Here are the concepts and their respective rationales:{概念阶段生成的内容}."，使疾病分类能够基于更全面的信息进行分析。

#### 文件夹批量处理
```bash
# 批量处理文件夹中的所有图片
cd /root/LLaVA-Med
./run_inference.sh --task both --folder-path /path/to/images/folder
```

#### 保存结果到文件
```bash
# 自定义输出文件名
cd /root/LLaVA-Med
./run_inference.sh --task concepts --image-path /path/to/image.jpg --output-file result.json
```

## 输出格式说明

### JSON输出结构

推理结果以结构化JSON格式保存，字段顺序如下：

```json
{
  "stage": "both",  // 任务类型：concepts, disease 或 both
  "concepts_model_path": "/root/autodl-tmp/model/llava-med-concepts",  // 概念模型路径
  "disease_model_path": "/root/autodl-tmp/model/llava-med-disease",    // 疾病模型路径
  "clip_path": "/root/autodl-tmp/model/clip-vit-large-patch14-336",    // CLIP模型路径
  "generate_text": {
    "image_id": {    // 图片ID（无扩展名的文件名）
      "image_path": "/path/to/image.jpg",  // 图片完整路径
      "concepts": { ... },  // 概念预测结果（仅concepts和both任务）
      "disease": { ... }    // 疾病分类结果（仅disease和both任务）
    }
    // 多个图片时会有更多条目（文件夹模式）
  }
}
```
