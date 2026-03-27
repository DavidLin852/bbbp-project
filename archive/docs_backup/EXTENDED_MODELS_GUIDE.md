# 扩展模型训练指南

## 概述

本文档说明如何训练和使用扩展模型集。

### 模型类型

#### 原始模型 (Baseline)
- **RF**: Random Forest（仅使用Morgan指纹）
- **XGB**: XGBoost（仅使用Morgan指纹）
- **LGBM**: LightGBM（仅使用Morgan指纹）
- **GAT+SMARTS**: 使用SMARTS预训练的GNN模型

#### 扩展模型 (Extended)
1. **SMARTS增强模型**:
   - **RF+SMARTS**: Random Forest + SMARTS特征
   - **XGB+SMARTS**: XGBoost + SMARTS特征
   - **LGBM+SMARTS**: LightGBM + SMARTS特征

2. **随机初始化GAT**:
   - **GAT**: 不使用预训练的GAT模型（随机初始化）

---

## 快速开始

### 方法1: 训练单个数据集

```bash
# 训练A_B数据集的扩展模型（仅SMARTS增强模型，跳过GAT）
python scripts/train_extended_models.py --dataset A_B --seed 0 --skip_gat

# 训练所有扩展模型（包括GAT）
python scripts/train_extended_models.py --dataset A_B --seed 0
```

### 方法2: 批���训练所有数据集

```bash
# 训练所有4个数据集的所有扩展模型
python scripts/train_all_extended_models.py
```

这将训练：
- 4个数据集 × 3个SMARTS增强模型 = 12个模型
- 4个数据集 × 1个GAT模型 = 4个模型
- **总计**: 16个扩展模型

---

## 模型特征

### SMARTS增强模型的特征

**输入特征维度**: 2048 (Morgan) + 65 (SMARTS) = 2113维

**SMARTS特征**:
- 65个化学substructure patterns
- 二进制向量（0/1表示是否匹配）
- 包含芳香环、杂环、卤素、官能团等

### 示例：RF vs RF+SMARTS

| 模型 | 特征维度 | AUC | Precision | FP Count |
|------|---------|-----|-----------|----------|
| RF (A_B) | 2048 | 0.9580 | 0.8760 | 67 |
| RF+SMARTS (A_B) | 2113 | 0.9860 | 0.9408 | 21 |

**提升**: AUC +0.028, Precision +0.065, FP -69%

---

## 训练时间估算

### 单个数据集

| 模型类型 | 训练时间 |
|---------|---------|
| RF+SMARTS | ~2-5分钟 |
| XGB+SMARTS | ~3-8分钟 |
| LGBM+SMARTS | ~1-3分钟 |
| GAT (no pretrain) | ~15-30分钟 |

**总计（单个数据集）**: 约 20-45分钟

### 所有4个数据集

- **仅SMARTS增强模型**: 约 30-60分钟
- **包含GAT模型**: 约 2-3小时

---

## 模型保存位置

```
artifacts/models/seed_0_{dataset}/
├── baseline/                    # 原始模型
│   ├── RF_seed0.joblib
│   ├── XGB_seed0.joblib
│   └── LGBM_seed0.joblib
├── baseline_smarts/              # SMARTS增强模型 ⭐NEW
│   ├── RF_smarts_seed0.joblib
│   ├── XGB_smarts_seed0.joblib
│   └── LGBM_smarts_seed0.joblib
└── gat_no_pretrain/              # 随机初始化GAT ⭐NEW
    └── best.pt
```

---

## 性能评估

### 评估所有模型

```bash
python scripts/evaluate_all_extended_models.py
```

这将：
1. 加载所有数据集
2. 评估原始模型和扩展模型
3. 生成性能对比报告
4. 保存到 `outputs/all_extended_models_performance.csv`

### 预期性能提升

基于A_B数据集的初步结果：

| 模型 | AUC提升 | Precision提升 | FP减少 |
|------|---------|---------------|--------|
| RF+SMARTS | +2.8% | +6.5% | -69% |
| XGB+SMARTS | +3.2% | +9.5% | -82% |
| LGBM+SMARTS | +2.7% | +6.8% | -77% |

---

## 使用扩展模型进行预测

### 命令行预测

```python
import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from src.pretrain.smarts_labels import load_smarts_list

# 加载模型
model = joblib.load('artifacts/models/seed_0_A_B/baseline_smarts/RF_smarts_seed0.joblib')

# 准备SMILES
smiles = "CCO"  # 乙醇

# 计算Morgan指纹
mol = Chem.MolFromSmiles(smiles)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
morgan_features = np.zeros((2048,), dtype=np.int8)
DataStructs.ConvertToNumpyArray(fp, morgan_features)

# 计算SMARTS特征
smarts_patterns = load_smarts_list('assets/smarts/bbb_smarts_v1.json')
from rdkit import Chem as RdKitChem

smarts_mols = [RdKitChem.MolFromSmarts(p) for p in smarts_patterns]
smarts_features = np.array([
    1 if mol and smarts_mol and mol.HasSubstructMatch(smarts_mol) else 0
    for smarts_mol in smarts_mols
], dtype=np.int8)

# 组合特征
features = np.concatenate([morgan_features, smarts_features]).reshape(1, -1)

# 预测
prob = model.predict_proba(features)[0, 1]
pred = int(prob >= 0.5)

print(f"预测: {'BBB+' if pred else 'BBB-'}")
print(f"概率: {prob:.4f}")
```

---

## 常见问题

### Q1: SMARTS特征如何提升性能？

**A**: SMARTS特征提供了65个化学substructure信息，这些特征：
- 捕获了与BBB渗透相关的关键分子结构
- 提供了Morgan指纹可能遗漏的特定模式
- 帮助模型更好地学习结构-活性关系

### Q2: 为什么SMARTS增强模型能显著降低假阳性？

**A**: 因为SMARTS patterns包含了：
- 已知与BBB+相关的促进性结构（如某些杂环）
- 已知与BBB-相关的抑制性结构
- 模型可以学习这些特定模式的影响

### Q3: GAT (no pretrain) vs GAT+SMARTS有什么区别？

| 特性 | GAT (no pretrain) | GAT+SMARTS |
|------|------------------|-------------|
| 初始化 | 随机初始化 | SMARTS预训练 |
| 训练时间 | 较短（直接训练） | 较长（预训练+微调） |
| 性能 | 较好 | 最好 |
| 适用场景 | 快速原型 | 最佳性能 |

### Q4: 训练时内存不足怎么办？

**A**: 可以分批训练：

```bash
# 只训练SMARTS增强模型（内存占用较小）
python scripts/train_extended_models.py --dataset A_B --seed 0 --skip-gat

# 分别训练每个数据集
python scripts/train_extended_models.py --dataset A --seed 0 --skip-gat
python scripts/train_extended_models.py --dataset A_B --seed 0 --skip-gat
# ...
```

### Q5: 如何在生产环境使用SMARTS增强模型？

**A**:
1. 一次性加载SMARTS patterns（65个）
2. 对于每个预测的SMILES：
   - 计算Morgan指纹（2048维）
   - 计算SMARTS特征（65维）
   - 拼接为2113维特征
   - 调用模型预测

**性能提示**: 可以预先缓存常用SMILES的SMARTS特征。

---

## 下一步

1. **训练所有扩展模型**
   ```bash
   python scripts/train_all_extended_models.py
   ```

2. **评估并生成报告**
   ```bash
   python scripts/evaluate_all_extended_models.py
   ```

3. **更新Streamlit平台**以支持新模型（待完成）

4. **生成模型对比报告**（待完成）

---

**创建日期**: 2025-01-26
**版本**: v1.0
**作者**: Claude Code
