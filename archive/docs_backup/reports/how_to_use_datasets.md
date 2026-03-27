# 如何使用不同数据集��模型

## 快速开始

### 1. 查看已训练的模型

所有模型已保存在 `artifacts/models/` 目录下：

```bash
# 查看所有可用模型
ls artifacts/models/seed_0_*/baseline/*.joblib
```

输出：
```
artifacts/models/seed_0_A/baseline/RF_seed0.joblib
artifacts/models/seed_0_A/baseline/XGB_seed0.joblib
artifacts/models/seed_0_A_B/baseline/RF_seed0.joblib
artifacts/models/seed_0_A_B/baseline/XGB_seed0.joblib
artifacts/models/seed_0_A_B_C/baseline/RF_seed0.joblib
artifacts/models/seed_0_A_B_C/baseline/XGB_seed0.joblib
artifacts/models/seed_0_A_B_C_D/baseline/RF_seed0.joblib
artifacts/models/seed_0_A_B_C_D/baseline/XGB_seed0.joblib
```

### 2. 选择数据集进行预测

#### 方法1: 修改多模型预测模块配置

```python
from src.multi_model_predictor import MultiModelPredictor, ModelConfig, create_ensemble_predictor

# 使用A+B组模型（推荐）
predictor = create_ensemble_predictor(
    strategy='hard_voting',
    threshold=0.5
)

# 或者指定使用特定数据集的模型
from pathlib import Path

models_A_B = {
    'Random Forest (A+B)': ModelConfig(
        name='Random Forest (A+B)',
        path='artifacts/models/seed_0_A_B/baseline/RF_seed0.joblib',
        model_type='rf',
        auc=0.968,
        precision=0.904
    ),
    'XGBoost (A+B)': ModelConfig(
        name='XGBoost (A+B)',
        path='artifacts/models/seed_0_A_B/baseline/XGB_seed0.joblib',
        model_type='rf',
        auc=0.969,
        precision=0.925
    )
}

predictor_custom = MultiModelPredictor(models=models_A_B)

# 预测
results = predictor_custom.predict(['CCO', 'CC(=O)OC1=CC=C(C=C)C=C1'])
print(results.ensemble_prediction)
```

#### 方法2: 直接使用单个模型

```python
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy import sparse
import numpy as np

# 加载模型
model = joblib.load('artifacts/models/seed_0_A_B/baseline/RF_seed0.joblib')

# 计算Morgan指纹
def compute_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return sparse.csr_matrix(arr.reshape(1, -1))

# 预测
smiles = 'CCO'
X = compute_fp(smiles)
prob = model.predict_proba(X)[0, 1]
pred = 'BBB+' if prob >= 0.5 else 'BBB-'

print(f'SMILES: {smiles}')
print(f'预测: {pred}')
print(f'概率: {prob:.3f}')
```

### 3. 在Web应用中切换数据集

访问Streamlit应用后，可以创建一个新的页面来选择数据集。或者修改现有页面添加数据集选择功能。

示例代码：

```python
import streamlit as st

# 侧边栏选择数据集
st.sidebar.title("选择数据集")

dataset_options = {
    'A组 (高质量)': 'A',
    'A+B组 (默认推荐)': 'A_B',
    'A+B+C组 (扩展)': 'A_B_C',
    'A+B+C+D组 (完整)': 'A_B_C_D'
}

selected = st.sidebar.selectbox(
    "选择训练数据集",
    list(dataset_options.keys()),
    index=1  # 默认A+B
)

dataset_code = dataset_options[selected]

# 显示数据集信息
st.info(f"当前使用: **{selected}**")

# 使用对应的模型
import joblib

if selected == 'A组 (高质量)':
    model_path = 'artifacts/models/seed_0_A/baseline/RF_seed0.joblib'
elif selected == 'A+B组 (默认推荐)':
    model_path = 'artifacts/models/seed_0_A_B/baseline/RF_seed0.joblib'
# ... 其他数据集

# 加载模型并预测
model = joblib.load(model_path)
# ... 进行预测
```

### 4. 批量预测并对比

```python
from pathlib import Path
import joblib
import pandas as pd

# 测试SMILES列表
test_smiles = ['CCO', 'CC(=O)OC1=CC=C(C=C)C=C1', 'c1ccccc1']

# 所有数据集的模型
datasets = ['A', 'A_B', 'A_B_C', 'A_B_C_D']
models = ['RF', 'XGB']

results = []

for dataset in datasets:
    for model_name in models:
        model_path = f'artifacts/models/seed_0_{dataset}/baseline/{model_name}_seed0.joblib'
        model = joblib.load(model_path)

        # 预测
        probs = []
        for smi in test_smiles:
            # 计算特征并预测
            X = compute_fp(smi)
            prob = model.predict_proba(X)[0, 1]
            probs.append(prob)

        results.append({
            'Dataset': dataset,
            'Model': model_name,
            'Avg_Prob': np.mean(probs)
        })

df = pd.DataFrame(results)
print(df)
```

## 数据集性能对比

根据训练结果：

| 数据集 | 训练样本 | RF AUC | RF Precision | XGB AUC | XGB Precision | 推荐 |
|--------|---------|--------|--------------|---------|---------------|------|
| **A** | 846 | 0.926 | 0.947 | 0.921 | 0.939 | 高Precision |
| **A+B** | 3743 | 0.968 | 0.904 | 0.969 | 0.925 | ⭐ 默认推荐 |
| **A+B+C** | 6204 | 0.951 | 0.861 | 0.950 | 0.875 | 探索性 |
| **A+B+C+D** | 6245 | 0.943 | 0.855 | 0.945 | 0.866 | 完整数据 |

### 选择建议

- **生产环境**: A+B组（最佳综合性能）
- **药物研发**: A组（最高Precision）
- **学术研究**: A+B+C组（大数据集）
- **极限探索**: A+B+C+D组（所有数据）

## 高级用法

### 1. 集成不同数据集的模型

```python
from src.multi_model_predictor import MultiModelPredictor, ModelConfig

# 混合使用不同数据集的模型
mixed_models = {
    'RF (A+B)': ModelConfig(
        name='RF (A+B)',
        path='artifacts/models/seed_0_A_B/baseline/RF_seed0.joblib',
        model_type='rf'
    ),
    'XGB (A)': ModelConfig(
        name='XGB (A)',
        path='artifacts/models/seed_0_A/baseline/XGB_seed0.joblib',
        model_type='rf'
    ),
    # ... 更多模型
}

predictor = MultiModelPredictor(models=mixed_models)
results = predictor.predict(['CCO'])
```

### 2. 动态选择模型

```python
def select_best_model(smiles, threshold=0.95):
    """根据预测置信度选择模型"""

    # A组：高Precision
    model_A = joblib.load('artifacts/models/seed_0_A/baseline/RF_seed0.joblib')

    # A+B组：高AUC
    model_AB = joblib.load('artifacts/models/seed_0_A_B/baseline/RF_seed0.joblib')

    # 预测
    X = compute_fp(smiles)
    prob_A = model_A.predict_proba(X)[0, 1]
    prob_AB = model_AB.predict_proba(X)[0, 1]

    # 如果A组模型非常确信（prob > 0.95 或 < 0.05），使用A组
    if prob_A > 0.95 or prob_A < 0.05:
        return 'A', prob_A
    else:
        # 否则使用A+B组
        return 'A+B', prob_AB
```

## 注意事项

1. **模型路径**: 确保路径正确，模型在 `baseline/` 子目录下
2. **特征计算**: 所有模型使用相同的Morgan指纹（2048位，半径2）
3. **阈值**: 默认使用0.5作为分类阈值
4. **性能**: 不同数据集训练的模型性能不同，根据需求选择

## 下一步

1. 查看性能详细报告: `docs/dataset_training_results.md`
2. 在Web应用中集成数据集选择功能
3. 对比不同数据集的预测结果
4. 根据应用场景选择最合适的数据集
