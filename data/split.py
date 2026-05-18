import json

def make_small_dataset(train_path, test_path, train_out, test_out, train_size=10, test_size=5):
    """
    构造小样本数据集（通用函数）

    参数:
        train_path: 原始训练集路径
        test_path: 原始测试集路径
        train_out: 输出训练集文件名
        test_out: 输出测试集文件名
        train_size: 训练集样本数
        test_size: 测试集样本数
    """
    # 读取原始数据
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    # 构造小样本数据
    train_small = {
        "task_name": train_data["task_name"],
        "global": train_data["global"],
        "data": train_data["data"][:train_size]
    }

    test_small = {
        "task_name": test_data["task_name"],
        "global": test_data["global"],
        "data": test_data["data"][:test_size]
    }

    # 保存结果
    with open(train_out, 'w') as f:
        json.dump(train_small, f, indent=2)
    with open(test_out, 'w') as f:
        json.dump(test_small, f, indent=2)

    print(f"✅ 已生成小数据集: {train_out}, {test_out}")
    print(f"   训练样本数: {len(train_small['data'])}, 测试样本数: {len(test_small['data'])}\n")


# === 构造 derm7pt_concepts 小数据集 ===
make_small_dataset(
    'derm7pt_concepts_train_dataset.json',
    'derm7pt_concepts_test_dataset.json',
    'derm7pt_concepts_train_dataset_small.json',
    'derm7pt_concepts_test_dataset_small.json',
    train_size=10,
    test_size=5
)

# === 构造 derm7pt_disease 小数据集 ===
make_small_dataset(
    'derm7pt_disease_train_dataset.json',
    'derm7pt_disease_test_dataset.json',
    'derm7pt_disease_train_dataset_small.json',
    'derm7pt_disease_test_dataset_small.json',
    train_size=10,
    test_size=5
)
