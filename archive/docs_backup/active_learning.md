# 主动学习模块使用指南

## 功能概述

主动学习模块允许您：
1. **检查数据**：输入SMILES，检查是否存在于当前训练数据中
2. **智能预测**：对新分子进行预测
3. **专家标注**：添加专家标注（BBB+或BBB-）
4. **数据集管理**：创建自定义数据集
5. **模型重训**：使用新数据重新训练所有模型

## 快速开始

### Web界面使用

启动应用后，访问 "Active Learning" 页面（页面4）：

```bash
streamlit run app_bbb_predict.py
```

访问: http://localhost:8503

### 使用流程

#### 第一步：检查和标注

1. 输入SMILES字符串（例如：`CCO`）
2. 点击"检查"按钮
3. 查看结果：
   - **如果存在**：显示标签和数据划分
   - **如果不存在**：显示预测结果，然后询问是否添加标注

4. 添加标注：
   - 点击"✅ 标注为 BBB+" 或 "❌ 标注为 BBB-"
   - 或点击"⏭️ 跳过"不添加标注

#### 第二步：管理新数据集

1. 累积多个新标注后，访问"管理新数据集"标签
2. 查看所有新标注
3. 输入数据集名称（例如：`my_custom_dataset`）
4. 点击"保存数据集"

数据集将保存在：`data/custom_datasets/{dataset_name}/`

保存的内容：
- `full_dataset.csv` - 完整数据集（原始+新标注）
- `new_annotations.csv` - 只包含新标注
- `metadata.json` - 数据集元数据

#### 第三步：重新训练模型

1. 在"重新训练"标签中选择已保存的数据集
2. 输入模型版本号（例如：`v2`, `custom_001`）
3. 选择是否跳过GNN训练
4. 点击"开始训练"或手动运行命令

训练完成后，新模型保存在：`artifacts/models/{model_version}_seed_{seed}/`

## 编程接口使用

### 基本使用

```python
from src.active_learning import create_active_learning_manager

# 创建管理器
manager = create_active_learning_manager(seed=0)

# 1. 检查SMILES
result = manager.check_smiles('CCO')
if result.exists:
    print(f"标签: {result.label_str}")
    print(f"划分: {result.split}")
else:
    print("不在训练数据中")

    # 2. 预测
    pred_label, pred_prob, individual = manager.predict_smiles('CCO')
    print(f"预测: {'BBB+' if pred_label == 1 else 'BBB-'}")
    print(f"概率: {pred_prob:.3f}")

    # 3. 添加标注
    manager.add_annotation(
        smiles='CCO',
        label=1,  # 1 for BBB+, 0 for BBB-
        predicted_label=pred_label,
        predicted_probability=pred_prob
    )
```

### 保存数据集

```python
# 保存为新数据集
manager.save_new_annotations(dataset_name='my_dataset')

# 准备重新训练的划分
train_file, val_file, test_file = manager.prepare_retrain_splits(
    dataset_name='my_dataset',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

### 重新训练模型

```bash
# 命令行训练
python scripts/retrain_with_new_data.py \
    --dataset_name my_dataset \
    --model_version v2 \
    --seed 0 \
    --skip_gnn
```

参数说明：
- `--dataset_name`: 自定义数据集名称
- `--model_version`: 新模型版本号
- `--seed`: 随机种子
- `--skip_gnn`: 跳过GNN训练（可选）

## 完整示例

### 场景：发现新分子并添加到训练集

```python
from src.active_learning import create_active_learning_manager

# 初始化
manager = create_active_learning_manager(seed=0)

# 检查新分子
new_molecule = "C1=CC=CC=C1C=O"  # 苯甲醛
result = manager.check_smiles(new_molecule)

if not result.exists:
    print(f"{new_molecule} 不在训练集中")

    # 预测
    pred, prob, _ = manager.predict_smiles(new_molecule)
    print(f"模型预测: {'BBB+' if pred == 1 else 'BBB-'} (概率: {prob:.3f})")

    # 添加专家标注（假设专家确定它是BBB+）
    manager.add_annotation(
        smiles=new_molecule,
        label=1,
        predicted_label=pred,
        predicted_probability=prob
    )
    print("已添加专家标注")

# 保存数据集
manager.save_new_annotations(dataset_name='benzene_derivatives')
print("数据集已保存")

# 查看统计
stats = manager.get_statistics()
print(f"原始训练集: {stats['original_train_size']}")
print(f"新标注数: {stats['new_annotations_count']}")
print(f"潜在新训练集: {stats['potential_new_train_size']}")
```

## 批量标注示例

```python
# 批量检查多个分子
molecules = {
    'C1=CC=CC=C1C=O': '苯甲醛',
    'NCC1=CC=C(O)C=C1O': '多巴胺',
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C': '咖啡因'
}

manager = create_active_learning_manager(seed=0)

for smiles, name in molecules.items():
    result = manager.check_smiles(smiles)

    if not result.exists:
        print(f"\n{name} ({smiles}):")
        pred, prob, _ = manager.predict_smiles(smiles)
        print(f"  预测: {'BBB+' if pred == 1 else 'BBB-'} ({prob:.3f})")

        # 根据预测添加标注（或根据专家知识）
        # 这里示例：如果概率>0.7，则按预测标注
        if prob > 0.7:
            label = pred
            manager.add_annotation(smiles, label, pred, prob)
            print(f"  已自动添加标注: {'BBB+' if label == 1 else 'BBB-'}")
        else:
            print(f"  概率较低，需要专家判断")

# 保存所有标注
manager.save_new_annotations('batch_annotations_v1')
```

## 目录结构

```
bbb_project/
├── data/
│   └── custom_datasets/
│       └── {dataset_name}/
│           ├── full_dataset.csv          # 完整数据集
│           ├── new_annotations.csv       # 新标注记录
│           ├── metadata.json             # 元数据
│           └── splits/
│               ├── train.csv
│               ├── val.csv
│               ├── test.csv
│               └── split_info.json
├── artifacts/
│   └── models/
│       └── {model_version}_seed_{seed}/
│           ├── baseline/
│           │   ├── RF_seed{seed}.joblib
│           │   ├── XGB_seed{seed}.joblib
│           │   └── LGBM_seed{seed}.joblib
│           └── training_summary.json
├── src/
│   └── active_learning.py                # 主动学习核心模块
├── scripts/
│   └── retrain_with_new_data.py          # 重新训练脚本
└── pages/
    └── 4_active_learning.py              # Streamlit页面
```

## 数据格式

### 原始数据格式

训练数据使用以下格式：
- `SMILES`: SMILES字符串
- `y_cls`: 标签（1 for BBB+, 0 for BBB-）
- `split`: 数据划分（train/val/test）

### 新标注数据格式

`new_annotations.csv`包含：
- `SMILES`: SMILES字符串
- `BBB+/BBB-`: 标签字符串
- `y_cls`: 标签数值
- `timestamp`: 标注时间
- `predicted_label`: 模型预测
- `predicted_probability`: 预测概率
- `annotation_source`: 标注来源

## 常见问题

### Q: 如何查看当前有多少新标注？
A: 使用 `manager.get_statistics()` 或访问Web界面的统计信息。

### Q: 可以修改已添加的标注吗？
A: 可以。如果再次添加相同SMILES的标注，会更新现有标注。

### Q: 重新训练需要多长时间？
A:
- RF/XGB/LGBM: 通常5-15分钟（取决于数据量）
- GNN: 需要30-60分钟（可选）

### Q: 新模型可以和原模型同时使用吗？
A: 可以。新模型保存在不同的目录（使用model_version区分），可以同时加载使用。

### Q: 如何使用新训练的模型？
A: 更新模型路径配置：
```python
from src.multi_model_predictor import MultiModelPredictor, ModelConfig

custom_models = {
    'Custom RF': ModelConfig(
        name='Custom RF',
        path='artifacts/models/v2_seed_0/baseline/RF_seed0.joblib',
        model_type='rf'
    )
}

predictor = MultiModelPredictor(models=custom_models)
```

## 最佳实践

1. **累积标注**：建议累积至少10-20个新标注后再重新训练
2. **质量优先**：只对您确定或验证过的分子添加标注
3. **记录来源**：记录标注来源（实验数据、文献、专家判断等）
4. **版本管理**：使用清晰的model_version（如`v2_expert_2024_01_26`）
5. **性能对比**：训练后对比新旧模型的性能
6. **备份重要**：重要的自定义数据集请备份

## 高级用法

### 自定义数据划分策略

```python
from sklearn.model_selection import train_test_split

# 加载完整数据
full_data = pd.read_csv('data/custom_datasets/my_dataset/full_dataset.csv')

# 自定义划分
X = full_data['SMILES'].values
y = full_data['y_cls'].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, train_size=0.7, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, train_size=0.5, test_size=0.5, random_state=42, stratify=y_temp
)
```

### 集成多个数据集

```python
import pandas as pd

# 加载多个自定义数据集
datasets = ['dataset1', 'dataset2', 'dataset3']

all_data = []
for ds in datasets:
    df = pd.read_csv(f'data/custom_datasets/{ds}/full_dataset.csv')
    all_data.append(df)

# 合并
combined = pd.concat(all_data, ignore_index=True)

# 去重（保留最新标注）
combined = combined.drop_duplicates(subset='SMILES', keep='last')
```

## 反馈和问题

如有问题或建议，请提交Issue或Pull Request。
