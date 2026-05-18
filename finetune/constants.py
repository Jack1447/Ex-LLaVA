"""
Constants for finetune
"""

# 概念
CONCEPT_NAMES = [
    "pigment network",
    "streaks", 
    "dots and globules",
    "blue-whitish veil",
    "regression structures"
]

# 概念标签
CONCEPTS_LABELS = {
    "pigment network":["absent","typical","atypical"],
    "streaks": ["absent","regular","irregular"],
    "dots and globules": ["absent","regular","irregular"],
    "blue-whitish veil": ["absent","present"],
    "regression structures": ["absent","present"],
}

# 疾病评估相关
DISEASE_LABELS = ["nevus", "melanoma"]
