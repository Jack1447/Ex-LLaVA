import pandas as pd
import os,base64,json
from openai import OpenAI
from collections import OrderedDict
import pprint
from tqdm import tqdm

OPENAI_API_KEY="Your Key"  

# 读取 csv 文件
df=pd.read_csv("derm7pt_concepts.csv")  # 包含各张图片的概念

# 一些关于概念的基本的信息
clinical_concepts=[
    "pigment network",
    "streaks",
    "dots and globules",
    "blue-whitish veil",
    "regression structures"
]
clinical_concepts_mapping={
    "pigment network": {0: "absent", 1: "typical", 2: "atypical"},
    "streaks": {0: "absent", 1: "regular", 2: "irregular"},
    "dots and globules": {0: "absent", 1: "regular", 2: "irregular"},
    "blue-whitish veil": {0: "absent", 1: "present"},
    "regression structures": {0: "absent", 1: "present"},
}

concept_instructions = {
    "pigment network": (
        "Definition: A grid-like pattern of intersecting brown lines that form a reticular network, commonly seen in dermoscopy. "
        "It is a crucial feature for evaluating melanocytic lesions.\n"
        "Label criteria:\n"
        " - absent: No visible intersecting brown lines or grid-like network. The lesion may appear uniform in color.\n"
        " - typical: Fine, regular brown lines evenly spaced to form a consistent grid with uniform hole sizes, often seen in benign lesions.\n"
        " - atypical: Lines are uneven in thickness or irregularly spaced, and the holes between the lines vary in size and shape, disrupting the regular pattern, often seen in malignant lesions.\n"
        "Visual cues to focus on:\n"
        " - Pattern symmetry: Look for the symmetry of the grid, evenly spaced lines, and consistent hole size. Typical networks show clear uniformity, while atypical ones show irregularities.\n"
        " - Line thickness: Typical networks have thin, evenly spaced lines, while atypical ones may show thicker or more variable lines.\n"
        " - Hole size: Typical networks show holes of a similar size, while atypical networks may have irregularly shaped or sized gaps.\n"
        "Example of typical: A lesion with fine, even brown lines forming a uniform mesh with holes of similar size.\n"
        "Example of atypical: A lesion with thicker, uneven lines and varying hole sizes, with some areas missing lines completely.\n"
    ),
    
    "streaks": (
        "Definition: Linear or bulbous pigmented projections extending from the periphery of a melanocytic lesion, often seen in melanoma.\n"
        "Label criteria:\n"
        " - absent: No peripheral projections. The lesion appears smooth and uniform at the edges.\n"
        " - regular: Symmetric projections, similar in length, thickness, and spacing around the periphery of the lesion.\n"
        " - irregular: Asymmetric projections, varying in length, thickness, and/or spacing. The projections may not follow any clear pattern.\n"
        "Visual cues to focus on:\n"
        " - Symmetry: Regular streaks maintain symmetry, whereas irregular streaks appear asymmetric.\n"
        " - Length and thickness consistency: Regular streaks maintain a consistent size throughout the lesion's perimeter.\n"
        " - Distribution: Regular streaks are evenly spaced around the lesion, while irregular streaks may be clustered in some regions.\n"
        "Example of regular: A lesion with evenly spaced, symmetrical streaks radiating outward from the periphery.\n"
        "Example of irregular: A lesion with streaks of different lengths and thicknesses, irregularly spaced along the border.\n"
    ),
    
    "dots and globules": (
        "Definition: Small, round pigmented structures within a melanocytic lesion, often indicative of melanocytic activity.\n"
        "Label criteria:\n"
        " - absent: No dots or globules visible within the lesion. The lesion appears uniform without distinct pigmented areas.\n"
        " - regular: Dots and globules are uniform in size, shape, and evenly distributed across the lesion.\n"
        " - irregular: Dots and globules vary in size, shape, and distribution. The structures may be unevenly spaced or clustered.\n"
        "Visual cues to focus on:\n"
        " - Size and shape: Regular dots are small and round; irregular ones may be larger or more irregular in shape.\n"
        " - Distribution: Regular dots are evenly spaced throughout the lesion, while irregular ones may be clustered or scattered unevenly.\n"
        " - Density: Check the density of the dots—whether they are uniformly distributed or concentrated in certain areas.\n"
        "Example of regular: A lesion with small, uniform dots distributed evenly across the surface.\n"
        "Example of irregular: A lesion with dots and globules of varying sizes, clustered in some areas, with scattered spots elsewhere.\n"
    ),
    
    "blue-whitish veil": (
        "Definition: A bluish-white cloud-like area often observed in melanocytic lesions, especially associated with melanoma in dermoscopy.\n"
        "Label criteria:\n"
        " - absent: No bluish or white haze is present. The lesion appears uniform without any cloudy or opaque overlay.\n"
        " - present: A bluish-white veil is clearly visible, with a cloud-like or opaque appearance blending into the surrounding tissue.\n"
        "Visual cues to focus on:\n"
        " - Hue and opacity: Look for a bluish or whitish tint. The presence of opacity or cloudiness in the area is crucial.\n"
        " - Blending: Check if the bluish-white veil blends seamlessly into the surrounding tissue or if it stands out distinctly.\n"
        " - Size and coverage: Assess how large the veil area is and whether it covers the entire lesion or only parts of it.\n"
        "Example of present: A lesion with a distinct bluish-white cloud-like area blending into the surrounding skin.\n"
        "Example of absent: A lesion that appears uniformly dark without any bluish or white overlay."
    ),
    
    "regression structures": (
        "Definition: Depigmented, scar-like areas within a melanocytic lesion, indicating partial regression of pigment cells.\n"
        "Label criteria:\n"
        " - absent: No visible depigmented or scar-like areas in the lesion. The lesion appears uniform in color.\n"
        " - present: Clear, whitish or grayish depigmented areas that suggest tissue regression.\n"
        "Visual cues to focus on:\n"
        " - Color: Look for whitish or grayish discoloration, as this typically signals regression.\n"
        " - Texture: Pay attention to any scarring, texture changes, or irregularities in the lesion.\n"
        " - Distribution: Check whether the depigmented areas are scattered or localized within the lesion.\n"
        "Example of present: A lesion with white or grayish depigmented areas indicating partial regression.\n"
        "Example of absent: A lesion with no depigmented or scar-like areas, appearing uniform in color and texture."
    )
}


class Image:
    # 图片文件夹的根路径
    root_path="D:\\Study\\CBVLM\\数据集制作\\Derm7pt"

    def __init__(self,image_id):
        self.image_id=image_id
        # 初始化一部字典来存储每各个特定的概念值对应的推理过程
        self.image_concepts_rationale = dict.fromkeys(clinical_concepts_mapping.keys())

    # 生成某图片某具体概念的推理过程
    def generate_concept_rationale(
            self,
            image_path: str,
            concept: str,
            label: str,
            model: str = "gpt-4o",
            max_words: int = 50,
            temperature: float = 0.2,
    ) -> str:
        # 合法性检查
        if not concept or not label:
            raise ValueError("concept 与 label 不能为空")

        # 转换本地图片
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{b64}"

        # 构造 prompt
        prompt = f"""
        You are a professional dermoscopy analysis assistant specialized in feature-level interpretation.
        You are given a skin lesion image and asked to provide an explanation for one specific dermoscopic concept.

        Concept: "{concept}"
        Definition and visual criteria:
        {concept_instructions[concept]}

        The image has been assigned the label: "{label}". Your task is to explain why this specific label is the correct choice for this image.

        Your task:
        - Explain WHY this label is appropriate based only on what is visible in the image.
        - Mention specific observable visual patterns (color, structure, shape, distribution, or symmetry).
        - Do NOT restate the label or concept name.
        - Do NOT speculate about diagnosis or unseen features.
        - Keep the explanation factual and concise (≤ {max_words} words).

        Output format (JSON only):
        {{
            "rationale": "your short justification here"
        }}
        """


        # 调用模型，强制 json
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
        )

        # 从响应中提取 rationale
        output_content = (response.choices[0].message.content or "").strip()
        try:
            data = json.loads(output_content)
            rationale = (data.get("rationale") or "").strip()
        except Exception:
            # 兜底
            rationale = output_content

        words = rationale.split()
        if len(words) > max_words:
            rationale = " ".join(words[:max_words])

        return rationale


    # 生成某图片全部概念的推理过程
    def generate_concepts_rationale(self):
        # 得到图片所对应的行号
        row=df.index[df['image_id']==self.image_id][0]
        # 得到图片对应的各个概念的值
        PN=clinical_concepts_mapping['pigment network'][df.loc[row,'pigment network']]
        STR=clinical_concepts_mapping['streaks'][df.loc[row,'streaks']]
        DG=clinical_concepts_mapping['dots and globules'][df.loc[row,'dots and globules']]
        BWV=clinical_concepts_mapping['blue-whitish veil'][df.loc[row,'blue-whitish veil']]
        RS=clinical_concepts_mapping['regression structures'][df.loc[row,'regression structures']]
        # 建立一个字典存储这些概念值
        concepts_values=[PN,STR,DG,BWV,RS]
        concepts_specific_values=dict(zip(clinical_concepts,concepts_values))
        # 得到图片的完整路径
        image_path=os.path.join(self.root_path,self.image_id+'.jpg')

        # 遍历全部的概念得到推理过程字典
        for concept in concepts_specific_values:
            label=concepts_specific_values[concept]
            rationale=self.generate_concept_rationale(
                image_path=image_path,
                concept=concept,
                label=label,
                model="gpt-4o",
                max_words=80,
                temperature=0.2
            )
            self.image_concepts_rationale[concept]=rationale

        return self.image_concepts_rationale


def main():
    # 按照顺序，初始化一部字典存储全部图片的全部概念的推理过程
    image_id_list = df['image_id'].tolist()[:3]
    images_concepts_rationale = OrderedDict.fromkeys(image_id_list)
    print(f"数据集的大小为：{len(image_id_list)}，开始生成全部图片的各个概念的推理过程。\n")
    # 使用全部的数据集
    for image_id in tqdm(image_id_list, desc="Generate concept rationale", unit="image"):
        image=Image(image_id)
        image_concepts_rationale=image.generate_concepts_rationale()
        images_concepts_rationale[image_id]=image_concepts_rationale

    print("全部图片的各个概念的推理过程已经生成完毕，保存在 'images_concepts_rationale.json' 中，请查看。\n ")

    with open("images_concepts_rationale.json",'w') as f:
        json.dump(images_concepts_rationale,f,indent=4)

if __name__ == "__main__":
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://api.302.ai/v1",
    )
    main()










