import json
import os
from collections import OrderedDict
import random
from tqdm import tqdm

with open('images_concepts_rationale.json', 'r') as f:
    images_concepts_rationale = json.load(f, object_pairs_hook=OrderedDict)

with open('images_concepts_label.json', 'r') as f:
    images_concepts_label = json.load(f, object_pairs_hook=OrderedDict)

# 定义全局信息
global_block = {
    "domain": "dermatology",
    "role_system_prompt": """
You are a professional dermoscopy analysis assistant specialized in feature-level interpretation. Your task is to analyze skin disease images and predict the label of certain clinical concepts. For each concept, you will provide two pieces of information:

1. **Label**: Choose the predicted label for the concept from the possible options provided below.
2. **Rationale**: A brief explanation that justifies the label based on the features observed in the image.

The possible labels for each concept are as follows:
- **pigment network**: "absent", "typical", "atypical"
- **streaks**: "absent", "regular", "irregular"
- **dots and globules**: "absent", "regular", "irregular"
- **blue-whitish veil**: "absent", "present"
- **regression structures**: "absent", "present"

The output MUST strictly follow the format below. Choose one label for each concept from the possible options:

<BEGIN_OUTPUT>
{
    "pigment network": {
        "label": "choose one from ["absent", "typical", "atypical"]",
        "rationale": "Explain why you chose the label for pigment network."
    },
    "streaks": {
        "label": "choose one from ["absent", "regular", "irregular"]",
        "rationale": "Explain why you chose the label for streaks."
    },
    "dots and globules": {
        "label": "choose one from ["absent", "regular", "irregular"]",
        "rationale": "Explain why you chose the label for dots and globules."
    },
    "blue-whitish veil": {
        "label": "choose one from ["absent", "present"]",
        "rationale": "Explain why you chose the label for blue-whitish veil."
    },
    "regression structures": {
        "label": "choose one from ["absent", "present"]",
        "rationale": "Explain why you chose the label for regression structures."
    }
}
<END_OUTPUT>

Please make sure to choose one label from the list of possible labels for each concept. Provide a rationale for each label based on visual features that can be observed in the image. Ensure that the rationale justifies your choice clearly and concisely. The output MUST strictly follow the JSON structure, and each concept must include both the "label" and "rationale" sections.
"""
}


valid_image_id_list=list(images_concepts_label.keys())
length_data=len(valid_image_id_list)
print(f"数据集的大小为：{length_data}\n")
random.seed(42)
random.shuffle(valid_image_id_list)
train_size = int(length_data * 0.8)
train_image_id_list=valid_image_id_list[:train_size]
print(f"训练集的大小为：{len(train_image_id_list)}\n")
test_image_id_list=valid_image_id_list[train_size:]
print(f"测试集的大小为：{len(test_image_id_list)}\n")


def build_samples(image_id_list,split_name):
    # 初始化 data 数组
    data = []
    for image_id in image_id_list:
        concepts = images_concepts_label[image_id]
        rationale = images_concepts_rationale[image_id]
        
        # 构建字典形式的输出
        output_dict = {}
        
        order = [
            "pigment network",
            "streaks",
            "dots and globules",
            "blue-whitish veil",
            "regression structures"
        ]
        
        for concept in order:
            label = concepts[concept]
            rationale_text = rationale.get(concept, "")
            output_dict[concept] = {
                "label": label,
                "rationale": rationale_text
            }

        # 将字典转换为JSON字符串
        gpt_output = json.dumps(output_dict, ensure_ascii=False, indent=4)
        gpt_output = f"<BEGIN_OUTPUT>\n{gpt_output}\n<END_OUTPUT>"

        sample = {
            "image": f"/root/autodl-tmp/data/Derm7pt/{image_id}.jpg",
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
        data.append(sample)

    return data


print("开始构造训练集。\n")
train_data = build_samples(train_image_id_list, "train")
train_dataset = {
    "task_name": "Dermoscopic_Concepts_With_Rationales",
    "global": global_block,
    "data": train_data
}
with open("derm7pt_concepts_train_dataset.json", "w", encoding="utf-8") as f:
    json.dump(train_dataset, f, ensure_ascii=False, indent=2)
print("训练集已保存到 derm7pt_concepts_train_dataset.json\n")

print("开始构造测试集。\n")
test_data = build_samples(test_image_id_list, "train")
test_dataset = {
    "task_name": "Dermoscopic_Concepts_With_Rationales",
    "global": global_block,
    "data": test_data
}
with open("derm7pt_concepts_test_dataset.json", "w", encoding="utf-8") as f:
    json.dump(test_dataset, f, ensure_ascii=False, indent=2)
print("测试集已保存到 derm7pt_concepts_test_dataset.json\n")

print("数据集构造完成。\n")