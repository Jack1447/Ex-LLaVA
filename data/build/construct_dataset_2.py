import json
import os
from collections import OrderedDict
import random
from tqdm import tqdm
import pandas as pd


# 定义全局信息
global_block = {
    "domain": "dermatology",
    "role_system_prompt": """
You are a professional dermoscopy analysis assistant specialized in feature-level interpretation of skin disease images. Your task is to analyze skin disease images and predict the disease label, based on detailed visual cues. 

For each image, you will provide the following information:

1. **Label**: Choose the predicted label for the disease from the following list: ["nevus", "melanoma"]. 
   - Make your choice based on the observable features visible in the image and the diagnostic concepts information provided below.

2. **Positive evidence**: 
   - List the observable features from the image that support the assigned category (either nevus or melanoma). 
   - You should describe specific features such as the type and arrangement of pigment network, presence of streaks, dots and globules, and any other relevant dermoscopic characteristics.

3. **Negative evidence**: 
   - Provide features from the image that rule out the reverse category (i.e., why the disease is not the opposite of the assigned label).
   - This can include the absence of certain features or the presence of features that are inconsistent with the reverse category.

4. **Summary**: 
   - Provide a concise rationale explaining why the assigned category is correct, based on the diagnostic criteria for nevus and melanoma. 
   - The rationale should be grounded in the features and diagnostic concepts information provided below.

### Specific requirements:
- **The possible labels for disease are**: "nevus", "melanoma".
- The output MUST strictly follow the format below and include all required fields.

The expected output format is:

<BEGIN_OUTPUT>
{
    "label": "nevus",   # or "melanoma"
    "positive evidence": "Explain what observable features from the image support the assigned category (nevus or melanoma).",
    "negative evidence": "Explain what features from the image rule out the reverse category (why it is not the other disease).",
    "summary": "Provide a concise rationale explaining why the assigned category is correct."
}
<END_OUTPUT>

### Notes:
- Your responses must be based **only** on observable features in the image.
- Do **not** include any external information or general knowledge that is not visible in the image.
- The **JSON structure** MUST be followed strictly, with each field properly filled.

"""
}


clinical_class_mapping={0: "nevus", 1: "melanoma"}


with open('images_category_evidences.json', 'r',encoding='utf-8') as f:
    images_category_evidences = json.load(f, object_pairs_hook=OrderedDict)
df=pd.read_csv("derm7pt_disease.csv")

with open('derm7pt_concepts_train_dataset.json','r',encoding='utf-8') as f:
    derm7pt_concepts_train_dataset=json.load(f)

with open('derm7pt_concepts_test_dataset.json','r',encoding='utf-8') as f:
    derm7pt_concepts_test_dataset=json.load(f)

valid_image_id_list=df['image_id'].tolist()
length_data=len(valid_image_id_list)
print(f"数据集的大小为：{length_data}\n")
train_image_id_list = [sample["meta"]["image_id"] for sample in derm7pt_concepts_train_dataset["data"]]
test_image_id_list  = [sample["meta"]["image_id"] for sample in derm7pt_concepts_test_dataset["data"]]
print(f"训练集的大小为：{len(train_image_id_list)}\n")
print(f"测试集的大小为：{len(test_image_id_list)}\n")


def build_samples(image_id_list,split_name):
    data=[]
    for image_id in tqdm(image_id_list, desc="Generate disease rationale dataset", unit="image"):
        row=df.index[df['image_id']==image_id][0]
        disease_label=clinical_class_mapping[df.loc[row,'category']]

        if split_name=="train":
            data_set=derm7pt_concepts_train_dataset
        else:
            data_set=derm7pt_concepts_test_dataset
        for sample in data_set["data"]:
            if sample["meta"]["image_id"] == image_id:
                for conv in sample["conversations"]:
                    if conv["from"] == "gpt":
                        information=conv["value"]

        gpt_output_parts = {}
        gpt_output_parts['label']=disease_label

        for key, value in images_category_evidences[image_id].items():
            gpt_output_parts['positive evidence'] = images_category_evidences[image_id]['positive evidence']
            gpt_output_parts['negative evidence'] = images_category_evidences[image_id]['negative evidence']
            gpt_output_parts['summary'] = images_category_evidences[image_id]['summary']
            

        gpt_output_parts = json.dumps(gpt_output_parts, ensure_ascii=False, indent=4)
        gpt_output_parts = f"<BEGIN_OUTPUT>\n{gpt_output_parts}\n<END_OUTPUT>"

        sample = {
            "image": f"/root/autodl-tmp/data/Derm7pt/{image_id}.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\nAccording to this picture of skin disease,provide [label, positive evidence,negative evidence,summary].Here are the concepts and their respective rationales:{information}."
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
        data.append(sample)

    return data



print("开始构造训练集。\n")
train_data = build_samples(train_image_id_list, "train")
train_dataset = {
    "format_version": "1.0",
    "task_name": "Dermoscopic_Concepts_With_Rationales",
    "global": global_block,
    "data": train_data
}
with open("derm7pt_disease_train_dataset.json", "w", encoding="utf-8") as f:
    json.dump(train_dataset, f, ensure_ascii=False, indent=2)
print("训练集已保存到 derm7pt_disease_train_dataset.json\n")

print("开始构造测试集。\n")
test_data = build_samples(test_image_id_list, "test")
test_dataset = {
    "format_version": "1.0",
    "task_name": "Dermoscopic_Concepts_With_Rationales",
    "global": global_block,
    "data": test_data
}
with open("derm7pt_disease_test_dataset.json", "w", encoding="utf-8") as f:
    json.dump(test_dataset, f, ensure_ascii=False, indent=2)
print("测试集已保存到 derm7pt_disease_test_dataset.json\n")

print("数据集构造完成。\n")