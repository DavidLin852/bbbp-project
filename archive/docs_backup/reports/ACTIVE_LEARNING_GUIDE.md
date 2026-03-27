# Active Learning 使用指南

## 功能概述

Active Learning模块允许你：
1. 输入新的SMILES分子
2. 验证分子格式并检查是否已在训练数据中
3. 使用16个模型进行预测
4. 手动标注真实标签
5. 保存到数据库
6. 使用新数据重新训练模型

---

## 快速开始

### 1. 访问Active Learning页面

启动Streamlit后，在左侧导航栏点击 **"Active Learning"**

```bash
streamlit run app_bbb_predict.py --server.port 8502
```

访问: http://localhost:8502

### 2. 输入新分子

在 **"Input & Predict"** 标签页：

1. **选择数据集**: 选择基础数据集 (A, A+B, A+B+C, A+B+C+D)
2. **选择模型**: 选择用于预测的模型 (RF, XGB, LGBM, GAT+SMARTS)
3. **输入SMILES**: 输入或粘贴SMILES字符串
4. **点击验证**: 检查SMILES格式和重复性
5. **查看预测**: 模型会预测BBB渗透性
6. **手动标注**: 根据实际知识确认或修改预测结果
7. **保存到数据库**: 点击保存按钮

### 3. 查看数据库

在 **"View Database"** 标签页：

- 浏览所有已保存的分子
- 按标签筛选 (BBB+, BBB-, 未标注)
- 查看分子详情
- 导出数据为CSV

### 4. 重新训练模型

当积累了足够的新数据后（建议至少20个分子）：

**方法1: 在Streamlit界面重新训练**
- 切换到 **"Retrain Models"** 标签页
- 查看数据库统计信息
- 点击 **"重新训练模型"** 按钮
- 等待训练完成

**方法2: 使用命令行重新训练**
```bash
# 例如：使用A_B数据集的新数据重新训练
python scripts/retrain_with_new_data.py \
    --dataset_name active_learning_A_B \
    --model_version v2_al \
    --seed 0

# 使用A_B_C数据集
python scripts/retrain_with_new_data.py \
    --dataset_name active_learning_A_B_C \
    --model_version v3_al \
    --seed 0
```

---

## 数据集说明

### 支持的基础数据集

| 数据集 | 描述 | 原始训练样本数 |
|--------|------|----------------|
| A | 仅高质量数据 (Group A) | ~4,700 |
| A_B | 高质量 + 文献数据 (Group A+B) | ~6,200 |
| A_B_C | 高质量 + 文献 + 计算 (Group A+B+C) | ~7,000 |
| A_B_C_D | 所有数据 (Group A+B+C+D) | ~7,800 |

### 数据保存位置

新分子保存在：
```
artifacts/active_learning_cache/new_molecules_{dataset}.csv
```

例如：
- `new_molecules_A.csv`
- `new_molecules_A_B.csv`
- `new_molecules_A_B_C.csv`
- `new_molecules_A_B_C_D.csv`

---

## 完整工作流程示例

### 场景: 为A_B数据集添加新分子并重新训练

#### Step 1: 添加分子

1. 打开 http://localhost:8502
2. 点击左侧 **"Active Learning"**
3. 在 **"Input & Predict"** 标签页：
   - Dataset: **A_B**
   - Model: **RF** (或任意模型)
   - 输入SMILES: `CCO` (乙醇)
   - 点击 **"验证 & 检查重复"**
   - 查看预测结果
   - 确认标签: **BBB+** (如果预测正确)
   - 点击 **"保存到数据库"**

4. 重复上述步骤添加更多分子

#### Step 2: 查看数据库

1. 切换到 **"View Database"** 标签页
2. 选择Dataset: **A_B**
3. 查看已保存的所有分子
4. 检查标签分布

#### Step 3: 重新训练

**方法A: 使用Streamlit界面**

1. 切换到 **"Retrain Models"** 标签页
2. 选择Dataset: **A_B**
3. 查看统计信息（新增分子数等）
4. 点击 **"重新训练RF/XGB/LGBM"** 按钮
5. 等待训练完成
6. 查看训练结果

**方法B: 使用命令行**

```bash
# 停止Streamlit (Ctrl+C)

# 运行重新训练脚本
python scripts/retrain_with_new_data.py \
    --dataset_name active_learning_A_B \
    --model_version v2_al \
    --seed 0

# 查看输出：
# - 原始训练集大小
# - 新增分子数
# - 合并后训练集大小
# - 各模型性能指标
```

---

## 重新训练后的模型

### 模型保存位置

```
artifacts/models/
└── {model_version}_seed_{seed}/
    ├── baseline/
    │   ├── RF_seed0.joblib
    │   ├── XGB_seed0.joblib
    │   └── LGBM_seed0.joblib
    ├── gnn_info.json
    └── training_summary.json
```

例如：
```
artifacts/models/v2_al_seed_0/
├── baseline/
│   ├── RF_seed0.joblib      ← 使用新数据训练的RF
│   ├── XGB_seed0.joblib     ← 使用新数据训练的XGB
│   └── LGBM_seed0.joblib    ← 使用新数据训练的LGBM
├── gnn_info.json
└── training_summary.json    ← 训练摘要和性能指标
```

### 在预测页面使用新模型

默认情况下，预测页面使用原始模型。要使用重新训练的模型：

1. 修改 `pages/0_prediction.py` 中的模型路径
2. 或创建新的预测配置

---

## 数据格式

### Active Learning数据库格式

```csv
smiles,canonical_smiles,label,prediction,probability,model,dataset,confidence,timestamp,user_added
CCO,CCO,1,1,0.95,RF,A_B,0.95,2025-01-26 22:00:00,True
c1ccccc1,c1ccccc1,0,0,0.12,XGB,A_B,0.88,2025-01-26 22:05:00,True
```

字段说明：
- `smiles`: 原始输入的SMILES
- `canonical_smiles`: RDKit标准化的SMILES
- `label`: 手动标注的标签 (1=BBB+, 0=BBB-)
- `prediction`: 模型预测结果
- `probability`: 预测概率
- `model`: 使用的模型
- `dataset`: 基础数据集
- `confidence`: 预测置信度
- `timestamp`: 保存时间
- `user_added`: 是否为用户添加 (True=已标注, False=仅预测)

---

## 常见问题

### Q1: 为什么我的新分子没有保存到数据库？

**检查**:
- 确认点击了"保存到数据库"按钮
- 检查是否有错误提示
- 确认选择了正确的数据集

### Q2: 重新训练需要多长时间？

**时间估计** (16个模型):
- RF: ~2-5分钟
- XGB: ~3-8分钟
- LGBM: ~1-3分钟
- GAT: ~10-30分钟 (可选)

总计约 10-20分钟 (不包括GAT)

### Q3: 需要多少新分子才能重新训练？

**建议**:
- 最少: 20个分子
- 推荐: 50+ 个分子
- 最佳: 100+ 个分子，且BBB+和BBB-分布相对均衡

### Q4: 新数据会覆盖原始数据吗？

**不会**。重新训练时会：
1. 加载原始训练数据
2. 加载Active Learning新增数据
3. 合并两个数据集（去除重复）
4. 使用合并后的数据训练新模型

原始数据不会被修改。

### Q5: 如何删除误添加的分子？

**方法1: 编辑CSV文件**
```bash
# 打开数据库文件
notepad artifacts/active_learning_cache/new_molecules_A_B.csv

# 删除误添加的行，保存
```

**方法2: 使用pandas**
```python
import pandas as pd

# 加载数据
df = pd.read_csv('artifacts/active_learning_cache/new_molecules_A_B.csv')

# 删除特定行 (例如最后一行)
df = df.iloc[:-1]

# 保存
df.to_csv('artifacts/active_learning_cache/new_molecules_A_B.csv', index=False)
```

### Q6: GNN模型会重新训练吗？

**不会自动训练**。由于GNN训练需要较长时间，重新训练脚本只会：

1. 保存GNN训练信息到 `gnn_info.json`
2. 提示手动运行GNN训练命令

要重新训练GAT+SMARTS模型：
```bash
python run_gnn_pipeline.py --custom_dataset v2_al --seed 0
```

---

## 高级功能

### 批量导入分子

如果已有大量SMILES需要标注：

1. 准备CSV文件：
```csv
smiles,label
CCO,1
c1ccccc1,0
CC(C)O,1
```

2. 在Python中导入：
```python
import pandas as pd
from pathlib import Path

# 读取现有数据库
db_path = Path('artifacts/active_learning_cache/new_molecules_A_B.csv')
if db_path.exists():
    df = pd.read_csv(db_path)
else:
    # 创建新数据库
    df = pd.DataFrame(columns=[
        'smiles', 'canonical_smiles', 'label', 'prediction', 'probability',
        'model', 'dataset', 'confidence', 'timestamp', 'user_added'
    ])

# 读取新数据
new_df = pd.read_csv('my_new_molecules.csv')

# 处理并添加
from rdkit import Chem
for _, row in new_df.iterrows():
    mol = Chem.MolFromSmiles(row['smiles'])
    if mol:
        canonical = Chem.MolToSmiles(mol)
        new_row = {
            'smiles': row['smiles'],
            'canonical_smiles': canonical,
            'label': row['label'],
            'prediction': row['label'],  # 假设预测正确
            'probability': 1.0,
            'model': 'Manual',
            'dataset': 'A_B',
            'confidence': 1.0,
            'timestamp': pd.Timestamp.now(),
            'user_added': True
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# 保存
df.to_csv(db_path, index=False)
```

### 导出数据用于其他工具

```python
import pandas as pd

# 加载数据库
df = pd.read_csv('artifacts/active_learning_cache/new_molecules_A_B.csv')

# 导出为不同格式
# 1. 简单SMILES和标签
df[['canonical_smiles', 'label']].to_csv('for_other_tool.csv', index=False)

# 2. 包含置信度
df[['canonical_smiles', 'label', 'confidence']].to_csv('with_confidence.csv', index=False)
```

---

## 最佳实践

### 1. 逐步添加分子

- ✅ 先添加10-20个分子测试流程
- ✅ 验证预测结果合理
- ✅ 确认数据库保存正确
- ✅ 然后批量添加更多分子

### 2. 保持标签平衡

- 尽量保持BBB+和BBB-数量相近
- 避免某一类标签过多

### 3. 选择多样性分子

- 添加不同类型的分子结构
- 覆盖不同的化学空间
- 避免重复或相似分子过多

### 4. 定期重新训练

- 每添加50-100个新分子后重新训练
- 对比新旧模型性能
- 保存性能提升的模型版本

### 5. 记录实验

为每次重新训练记录：
- 日期
- 新增分子数
- 数据集来源
- 模型版本号
- 性能指标

---

## 联系与反馈

如果遇到问题或有改进建议，请：
1. 查看终端错误信息
2. 检查 `artifacts/active_learning_cache/` 目录
3. 提供错误信息和复现步骤
