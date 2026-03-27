# B3DB数据集与穿透机制分析指南

## 概述

本指南说明如何使用B3DB数据集来研究血脑屏障(BBB)穿透机制。

## 数据集对比

### B3DB vs Cornelissen 2022

| 特性 | B3DB | Cornelissen 2022 |
|------|------|------------------|
| **样本数** | 7,807 | 8,658 |
| **BBB+率** | 63.5% | 77.7% |
| **机制标签** | ❌ 无 | ✅ 有 (Influx/Efflux/PAMPA/CNS) |
| **数据质量** | 高质量 | 高质量 |
| **重叠样本** | - | 2,018个 |

### 关键发现

**1. 重叠分子**
- 两个数据集有 **2,018个重叠分子**
- 这为交叉验证提供了机会！
- 可以用Cornelissen的机制标签来标注B3DB的部分分子

**2. B3DB的数据分组**

| 组别 | 样本数 | BBB+率 | 特点 |
|------|--------|--------|------|
| **A** | 1,058 | 87.9% | 高精度，低假阳性 |
| **B** | 3,621 | 68.7% | 最佳平衡 ⭐ |
| **C** | 3,075 | 50.0% | 大规模 |
| **D** | 51 | 7.8% | 最大覆盖 |

**3. 关键理化性质差异**

基于B3DB数据集，BBB+和BBB-分子的关键差异：

| 性质 | BBB+ | BBB- | 差值 | 文献一致性 |
|------|------|------|------|-----------|
| **TPSA** | 53-66 Å² | 101-141 Å² | -68~-81 | ✅ 一致 |
| **MW** | 311-355 Da | 383-471 Da | -71~-130 | ✅ 一致 |
| **LogP** | 2.8-2.9 | 1.1-2.0 | +0.8~+1.8 | ✅ 一致 |
| **HBA** | 3.7-4.3 | 6.1-8.2 | -2.4~-4.1 | ✅ 一致 |
| **HBD** | 1.1-1.4 | 2.5-3.5 | -1.3~-2.3 | ✅ 一致 |

## 使用策略

### 策略1: 使用重叠分子传递标签

```python
# 使用Cornelissen的机制标签标注B3DB的重叠分子
import pandas as pd

# 加载数据
b3db = pd.read_csv('outputs/b3db_analysis/b3db_with_features.csv')
cornelissen = pd.read_csv('data/transport_mechanisms/cornelissen_2022/cornelissen_2022_processed.csv')

# 找到重叠分子
overlap = set(b3db['SMILES']) & set(cornelissen['SMILES'])

# 为重叠分子添加机制标签
for smiles in overlap:
    b3db_loc = b3db[b3db['SMILES'] == smiles].index
    cornelissen_row = cornelissen[cornelissen['SMILES'] == smiles].iloc[0]

    # 添加机制标签
    b3db.loc[b3db_loc, 'label_Influx'] = cornelissen_row['label_Influx']
    b3db.loc[b3db_loc, 'label_Efflux'] = cornelissen_row['label_Efflux']
    b3db.loc[b3db_loc, 'label_PAMPA'] = cornelissen_row['label_PAMPA']
    b3db.loc[b3db_loc, 'label_CNS'] = cornelissen_row['label_CNS']
```

### 策略2: 基于启发式规则标注

对于没有实验标签的分子，可以使用基于理化性质的启发式规则：

```python
# 基于Cornelissen et al. 2022的规则
df['suggested_passive'] = (
    (df['TPSA'] < 90) &
    (df['LogP'] >= 1) &
    (df['LogP'] <= 3) &
    (df['MW'] < 500)
).astype(int)

df['suggested_influx'] = (
    (df['TPSA'] > 100) &
    (df['HBA'] > 5)
).astype(int)

df['suggested_efflux'] = (
    (df['MW'] > 500)
).astype(int)
```

### 策略3: 混合训练方法

```python
# 1. 在Cornelissen 2022上训练模型（有真实标签）
# 2. 在B3DB上微调（只有BBB标签）
# 3. 使用B3DB作为测试集验证BBB预测
```

## 生成的文件

运行 `explore_b3db_mechanisms.py` 后，会生成以下文件：

```
outputs/b3db_analysis/
├── b3db_with_features.csv              # B3DB + 提取的特征
└── b3db_with_mechanism_suggestions.csv # B3DB + 启发式机制标签
```

## 下一步建议

### 1. 数据合并
```bash
# 合并B3DB和Cornelissen数据集
python scripts/mechanism_training/merge_datasets.py
```

### 2. 模型训练选项

**选项A: 仅使用Cornelissen数据**
- ✅ 有真实机制标签
- ❌ 样本量较小（886-2474个）

**选项B: 仅使用B3DB + 启发式标签**
- ✅ 样本量大（7,807个）
- ❌ 标签不精确（基于规则）

**选项C: 混合方法（推荐）**
- ✅ 2018个重叠分子有真实标签
- ✅ 可用B3DB测试BBB预测
- ✅ 可用B3DB数据微调模型

### 3. 分析现有结果

查看生成的文件：

```bash
# 查看B3DB特征
cat outputs/b3db_analysis/b3db_with_features.csv | head -20

# 查看建议的机制标签
cat outputs/b3db_analysis/b3db_with_mechanism_suggestions.csv | head -20
```

## 重要提醒

1. **B3DB没有实验性的机制标签**
   - 所有机制标签都需要从其他来源获取
   - 或者基于启发式规则推断

2. **Cornelissen 2022是最好的机制标签来源**
   - 有实验验证的标签
   - 与B3DB有2018个重叠分子
   - 可以用于传递学习

3. **建议的验证策略**
   - 在Cornelissen上训练
   - 在B3DB的重叠分子上测试机制预测
   - 在B3DB的所有分子上测试BBB预测

## 代码示例

完整的例子可以在以下文件中找到：
- `scripts/mechanism_training/explore_b3db_mechanisms.py` - B3DB探索
- `scripts/mechanism_training/train_cornelissen_models.py` - Cornelissen训练
- `src/path_prediction/mechanism_predictor_cornelissen.py` - 预测器使用
