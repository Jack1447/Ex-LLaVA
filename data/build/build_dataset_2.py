import pandas as pd
import os,base64,json
from openai import OpenAI
from collections import OrderedDict
import pprint
from tqdm import tqdm
import logging

OPENAI_API_KEY="Your Key"

# 疾病分类映射
clinical_class_mapping={0: "nevus", 1: "melanoma"}

instructions = {
    "nevus": {
        "definition": "A nevus, commonly known as a mole, is a benign (non-cancerous) skin growth composed of melanocytes (pigment cells). Nevi can appear at birth or later in life.",
        "diagnostic_criteria": "The diagnosis of a nevus is based on the following dermoscopic features:",
        "features": {
            "pigment_network": "Typically regular, with thin, evenly spaced brown lines forming a uniform grid. The gaps between lines are consistent in size and shape.",
            "streaks": "Usually absent. If present, streaks are symmetric and evenly spaced around the lesion’s perimeter.",
            "dots_and_globules": "Usually regular, with small, round pigmented structures distributed evenly throughout the lesion.",
            "blue_whitish_veil": "Typically absent. There is no bluish or whitish cloud-like overlay.",
            "regression_structures": "Typically absent. No depigmented or scar-like areas are visible."
        },
        "general_characteristics": "A benign nevus is generally symmetrical, with well-defined borders, and consistent color and size. Nevi typically do not evolve into melanoma."
    },
    "melanoma": {
        "definition": "Melanoma is a malignant (cancerous) skin lesion originating from melanocytes. It is the deadliest form of skin cancer due to its ability to metastasize to other parts of the body.",
        "diagnostic_criteria": "The diagnosis of melanoma is based on the following dermoscopic features:",
        "features": {
            "pigment_network": "Typically atypical, with irregularly spaced, unevenly thick lines. The network’s symmetry is disrupted by irregular gaps and varying line thickness.",
            "streaks": "Typically irregular. Streaks are asymmetric with varying lengths, thicknesses, and spacing at the lesion’s periphery.",
            "dots_and_globules": "Typically irregular, with significant variation in size and shape. Some dots and globules may appear misshapen or larger than others.",
            "blue_whitish_veil": "Typically present. A bluish-white, cloud-like structure overlays the lesion, giving it a hazy appearance, which is a key melanoma indicator.",
            "regression_structures": "Typically present. Depigmented, scar-like areas within the lesion indicate regression of pigment cells."
        },
        "general_characteristics": "Melanoma lesions are often irregular in shape, with uneven color distribution and poorly defined borders. The presence of features like a blue-whitish veil or regression structures supports its malignancy."
    }
}



# 得到针对于某张图片的合并的概念推理字典
def images_concepts_rationale_merged(images_concepts_rationale):
    images_rationale_merged=dict()
    for image_id, concepts_rationale in list(images_concepts_rationale.items()):
        string_values=[str(value) for value in concepts_rationale.values()]
        concatenated_string = ','.join(string_values)
        images_rationale_merged[image_id]=concatenated_string
    return images_rationale_merged


class Image:
    # 图片文件夹的根路径
    root_path="D:\\Study\\CBVLM\\数据集制作\\Derm7pt"

    def __init__(self,image_id,image_rationale_merged):
        self.image_id=image_id
        self.image_category_evidences = dict.fromkeys(["positive evidence", "negative evidence","summary"])
        self.image_rationale_merged=image_rationale_merged

    def generate_category_evidence(
            self,
            image_path:str,
            image_category:str,
            image_reverse_category:str,
            image_rationale_merged:str,
            model:str="gpt-4o",
            max_words:int=70,
            temperature: float = 0.2,
    ) -> dict:
        # 合法性检查
        if not image_category:
            raise ValueError("image_category不能为空")

        # 转换本地图片
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{b64}"

        # 构造 prompt
        prompt = (
            "You are a dermoscopy assistant analyzing a skin lesion image.\n"
            f"Image Disease Category: '{image_category}'.\n"
            f"Concepts Rationale: '{image_rationale_merged}'.\n"
            f"Diagnostic criteria for diseases:\n"
            f"- **Nevus**: {instructions['nevus']['definition']} {instructions['nevus']['diagnostic_criteria']} {instructions['nevus']['features']}\n"
            f"- **Melanoma**: {instructions['melanoma']['definition']} {instructions['melanoma']['diagnostic_criteria']} {instructions['melanoma']['features']}\n"
            
            "Task: Provide a detailed explanation for why the category '{image_category}' is correct, and why the reverse category '{image_reverse_category}' is not correct.\n"
            
            "Your explanation must include:\n"
            "- **Positive evidence**: What observable features from the image support the assigned category (nevus or melanoma)?\n"
            "- **Negative evidence**: What features from the image rule out the reverse category?\n"
            "- **Summary**: A concise rationale for why the assigned category is correct, based on the diagnostic criteria.\n"
            
            "Guidelines:\n"
            "- **Only use observable visual cues** from the image.\n"
            "- Provide your explanation in **JSON format only**. No extra text or markdown.\n"
            "- If evidence is insufficient, explicitly state that.\n"
            f"- The explanation should not exceed {max_words} words.\n\n"
            "Return a single JSON object with the following keys:\n"
            '"positive evidence": "<why the image supports {image_category}>"\n'
            '"negative evidence": "<why the image does NOT support {image_reverse_category}>"\n'
            '"summary": "<reasoning for disease category>"\n'
            "Do not include anything else."
        )



        # 预定义的空响应
        EMPTY_RESPONSE = {"positive evidence": "", "negative evidence": "", "summary": ""}

        # 配置日志
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
        logger = logging.getLogger(__name__)

        try:
            # 调用模型，强制 JSON 输出
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

            # 获取模型响应并处理空值
            raw = (response.choices[0].message.content or "").strip()

            if not raw:
                logger.warning(f"Empty response for image: {image_url}")
                result = EMPTY_RESPONSE

            else:
                # 尝试解析 JSON
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as e:
                    logger.error(f"JSONDecodeError for image {image_url}: {e}")
                    result = EMPTY_RESPONSE
                else:
                    # 正常解析字段
                    pos = (data.get("positive evidence") or "").strip()
                    neg = (data.get("negative evidence") or "").strip()
                    summary = (data.get("summary") or "").strip()

                    result = {"positive evidence": pos, "negative evidence": neg, "summary": summary}

        except Exception as e:
            # 捕获所有异常，记录日志，并返回空响应
            logger.error(f"Exception while processing image {image_url}: {e}", exc_info=True)
            result = EMPTY_RESPONSE

        return result



    # 生成某图片的疾病分类的正向，负向证据
    def generate_category_evidences(self):
        # 得到图片所对应的行号
        row=df.index[df['image_id']==self.image_id][0]
        # 得到图片的完整路径
        image_path=os.path.join(self.root_path,self.image_id+'.jpg')
        # 得到图片的分类
        image_category=clinical_class_mapping[df.loc[row,'category']]
        image_reverse_category=clinical_class_mapping[abs(df.loc[row,'category']-1)]

        # 得到正向证据 负向证据 总结字典
        self.image_category_evidences=self.generate_category_evidence(
            image_path=image_path,
            image_category=image_category,
            image_reverse_category=image_reverse_category,
            image_rationale_merged=self.image_rationale_merged,
            model="gpt-4o",
            max_words=150,
            temperature=0.2,
        )

        return self.image_category_evidences


def main():
    # 按照顺序，初始化一部字典存储全部图片的全部概念的推理过程
    image_id_list = df['image_id'].tolist()
    images_category_evidences = OrderedDict.fromkeys(image_id_list)
    print("开始合并概念推理。\n")
    images_rationale_merged=images_concepts_rationale_merged(images_concepts_rationale)

    print(f"数据集的大小为：{len(image_id_list)}，开始生成全部图片的各个疾病的推理过程。\n")
    for image_id in tqdm(image_id_list,desc="Generate disease rationale",unit="image"):
        image = Image(image_id,images_rationale_merged[image_id])
        image_category_evidences=image.generate_category_evidences()
        images_category_evidences[image_id]=image_category_evidences

    with open("images_category_evidences_error.json",'w') as f:
        json.dump(images_category_evidences,f,indent=4)

    print("全部图片的各个疾病的推理过程已经生成完毕，保存在 'images_category_evidences_error.json' 中，请查看。\n ")
    

if __name__ == "__main__":
    with open('images_concepts_rationale.json', 'r') as f:
        images_concepts_rationale = json.load(f, object_pairs_hook=OrderedDict)

    # 读取 csv 文件，包含了各张图片的疾病的分类
    df=pd.read_csv("derm7pt_disease.csv")

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://api.302.ai/v1",
    )

    main()