import pandas as pd
import os,base64,json
from openai import OpenAI
from collections import OrderedDict
import pprint
from tqdm import tqdm

clinical_concepts_mapping={
    "pigment network": {0: "absent", 1: "typical", 2: "atypical"},
    "streaks": {0: "absent", 1: "regular", 2: "irregular"},
    "dots and globules": {0: "absent", 1: "regular", 2: "irregular"},
    "blue-whitish veil": {0: "absent", 1: "present"},
    "regression structures": {0: "absent", 1: "present"},
}

df=pd.read_csv("dermoscopic_concepts_Derm7pt_all.csv")  # 包含各张图片的概念
image_id_list=df['image_id'].tolist()
# 初始化一个字典
images_concepts_label=OrderedDict.fromkeys(image_id_list)

print(f"数据集的大小为：{len(image_id_list)}，开始生成全部图片的各个概念的标签。\n")

# 开始向里面填充内容 使用全部的数据集
for image_id in tqdm(image_id_list, desc="Generate concept label", unit="image"):
    # 初始化各个image_id的字典
    concepts_label=dict.fromkeys(clinical_concepts_mapping.keys())
    row = df.index[df['image_id'] == image_id][0]
    concepts_label['pigment network']=clinical_concepts_mapping['pigment network'][df.loc[row,'pigment network']]
    concepts_label['streaks']=clinical_concepts_mapping['streaks'][df.loc[row,'streaks']]
    concepts_label['dots and globules']=clinical_concepts_mapping['dots and globules'][df.loc[row,'dots and globules']]
    concepts_label['blue-whitish veil']=clinical_concepts_mapping['blue-whitish veil'][df.loc[row,'blue-whitish veil']]
    concepts_label['regression structures']=clinical_concepts_mapping['regression structures'][df.loc[row,'regression structures']]
    images_concepts_label[image_id]=concepts_label

print("全部图片的各个概念的标签已经生成完毕，保存在 'images_concepts_label.json' 中，请查看。\n")
with open("images_concepts_label.json",'w') as f:
    json.dump(images_concepts_label,f,indent=4)
