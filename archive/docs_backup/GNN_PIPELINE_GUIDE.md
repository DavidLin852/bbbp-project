# 完整GNN Pipeline执行指南

## 概述

**数据集**: A,B,C,D全部 (7,807样本)
**目标**: BBB+/BBB- 二分类
**重点**: 控制假阳性（FP），保守策略

---

## Pipeline架构

```
数据准备 (A,B,C,D全部)
    ↓
Baseline模型 ✅ (已完成)
    ├─ RF: AUC=0.9579, FP=67
    ├─ XGB: AUC=0.9487, FP=72
    └─ LGBM: AUC=0.9545, FP=53
    ↓
GNN Pipeline
    ├─ Step 1: GAT Baseline (纯图结构)
    ├─ Step 2: GAT + 物理属性 (logP + TPSA辅助)
    ├─ Step 3: SMARTS预训练
    ├─ Step 4: 微调BBB分类器
    └─ Step 5: 最终对比
```

---

## 当前状态

### ✅ 已完成：Baseline

| 模型 | AUC | Precision | Recall | F1 | FP | 状态 |
|------|-----|----------|--------|----|----|----|
| **RF** | **0.9579** | **0.8755** | **0.9496** | **0.9110** | **67** | ✅ 推荐 |
| XGB | 0.9487 | 0.8657 | 0.9355 | 0.8992 | 72 | ✅ |
| LGBM | 0.9545 | 0.8961 | 0.9214 | 0.9085 | 53 | ✅ |

**模型位置**: `artifacts/models/seed_0_full/baseline/RF_seed0.joblib`

---

## GNN Pipeline执行步骤

### Step 1: GAT Baseline (纯图结构)

**目的**: 建立图神经网络baseline

**输入**: 分子图结构
- 节点特征: 原子类型、杂化、电荷等
- 边特征: 键类型、共轭性等

**模型**: 3层GAT (hidden=128, heads=4)

**执行**:
```bash
python run_gnn_pipeline.py --seed 0 --step 1
```

**预期输出**:
- `artifacts/models/seed_0_full/gat_baseline/best.pt`
- `artifacts/metrics/gat_baseline_seed0.csv`

**预期时间**: 30-60分钟

---

### Step 2: GAT + 物理属性辅助

**目的**: 利用logP和TPSA辅助信号提升性能

**辅助任务**:
- 主任务: BBB+/- 分类
- 辅助任务1: logP回归 (分子脂溶性)
- 辅助任务2: TPSA回归 (极性表面积)

**模型**: GAT + 3个预测头

**执行**:
```bash
python run_gnn_pipeline.py --seed 0 --step 2
```

**预期输出**:
- `artifacts/models/seed_0_full/gat_phys_aux/best.pt`
- `artifacts/metrics/gat_phys_aux_seed0.csv`

**预期时间**: 60-90分钟

---

### Step 3: SMARTS预训练

**目的**: 使用化学子结构模式进行预训练

**SMARTS模式**: 158种化学子结构（来自assets/smarts/bbb_smarts_v1.json）

**预训练策略**:
- 输入: 分子图
- 输出: 158维multi-hot标签（每个SMARTS是否存在）
- 损失: BCEWithLogitLoss with pos_weight

**执行**:
```bash
# Step 3a: SMARTS预训练
python scripts/05_pretrain_smarts.py --seed 0 --epochs 40 --batch 64

# Step 3b: 微调BBB分类
python scripts/06_finetune_bbb_from_smarts.py --seed 0 --epochs 60
```

**预期输出**:
- 预训练: `artifacts/models/gat_pretrain_smarts/seed_0/bbb_smarts_v1/best.pt`
- 微调: `artifacts/models/gat_finetune_bbb/seed_0/pretrained_partial/best.pt`

**预期时间**: 2-3小时

---

### Step 4: 最终模型对比

**执行**:
```bash
python run_gnn_pipeline.py --seed 0 --step 5
```

**对比维度**:
- AUC (分类能力)
- Precision (控制FP的关键指标)
- Recall (覆盖正例的能力)
- F1 (平衡指标)
- FP (假阳性数量，越低越好)

---

## 模型选择建议

### 场景1: 保守部署（推荐）

**需求**: "负的不能被判成正值"

**推荐模型顺序**:
1. **Baseline - RF** (立即可用)
   - AUC=0.9579, FP=67
   - 已训练完成，可立即使用

2. **GAT+物理属性** (如需提升)
   - 预期: AUC +0.01-0.02, FP可能略增
   - 物理属性提供额外信号

3. **SMARTS预训练+微调** (最优)
   - 预期: AUC +0.02-0.03
   - 最强泛化能力

### 场景2: 研究对比

**所有模型训练完成后对比**:
- Baseline: RF, XGB, LGBM
- GNN: GAT baseline
- GNN+Aux: GAT + logP/TPSA
- GNN+Pretrain: SMARTS预训练+微调

---

## 快速开始（立即可用）

### 使用Baseline RF进行预测

```python
import joblib
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy import sparse
import numpy as np

# 加载模型
model = joblib.load('artifacts/models/seed_0_full/baseline/RF_seed0.joblib')

def predict_bbb(smiles_list, threshold=0.5):
    """预测BBB渗透性

    Args:
        smiles_list: SMILES字符串列表
        threshold: 分类阈值 (默认0.5, 保守策略可用0.65)

    Returns:
        prob: BBB+概率
        pred: 预测类别 (1=BBB+, 0=BBB-)
    """
    # 计算Morgan指纹
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            arr = np.zeros((2048,), dtype=np.int8)
            fps.append(arr)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)

    X = sparse.csr_matrix(np.vstack(fps))

    # 预测
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    return prob, pred

# 示例
smiles = ["CCO", "CC(=O)OC1=CC=C(C=C)C=C1", "CCN"]
prob, pred = predict_bbb(smiles, threshold=0.65)  # 保守阈值

for smi, p, y in zip(smiles, prob, pred):
    print(f"{smi}: prob={p:.3f}, pred={'BBB+' if y==1 else 'BBB-'}")
```

---

## 文件清单

### 数据文件
```
data/splits/seed_0_full/
├── train.csv (6,245样本)
├── val.csv (781样本)
├── test.csv (781样本)
└── split_report.json
```

### 特征文件
```
artifacts/features/seed_0_full/
├── morgan_2048.npz
├── descriptors.csv
├── meta.csv
└── pyg_graphs_*/  (图特征，GNN使用)
```

### 模型文件
```
artifacts/models/seed_0_full/
├── baseline/
│   ├── RF_seed0.joblib    ⭐立即可用
│   ├── XGB_seed0.joblib
│   └── LGBM_seed0.joblib
├── gat_baseline/        (Step 1)
├── gat_phys_aux/        (Step 2)
└── gat_finetune_bbb/     (Step 4)
```

### 结果文件
```
artifacts/metrics/
├── baseline_seed0.csv       ✅
├── gat_baseline_seed0.csv   (Step 1)
├── gat_phys_aux_seed0.csv   (Step 2)
└── final_comparison.csv     (Step 5)
```

---

## 估计执行时间

| 步骤 | 时间 | 累计 |
|------|------|------|
| Baseline (已完成) | - | - |
| Step 1: GAT Baseline | 30-60分钟 | 1小时 |
| Step 2: GAT+物理属性 | 60-90分钟 | 2.5小时 |
| Step 3: SMARTS预训练+微调 | 2-3小时 | 5小时 |
| Step 4: 对比 | 5分钟 | 5小时 |

---

## 建议执行策略

### 策略A: 分阶段执行（推荐）

1. **当前**: 使用Baseline RF (已完成)
   - 性能: AUC=0.9579, FP=67
   - 优点: 立即可用，性能优秀

2. **有时间后**: 执行Step 1-2 (GNN)
   - 预期提升: AUC +0.01-0.02
   - 时间投入: 2.5小时

3. **深入研究**: 执行Step 3-4 (SMARTS)
   - 预期提升: AUC +0.02-0.03
   - 时间投入: 5小时

### 策略B: 一次性执行

```bash
# 可以后台运行所有步骤
nohup python run_gnn_pipeline.py --seed 0 --step 1 &
nohup python run_gnn_pipeline.py --seed 0 --step 2 &
nohup python scripts/05_pretrain_smarts.py --seed 0 --epochs 40 &
```

---

## 预期最终结果

### 模型性能预期

| 模型 | AUC | Precision | FP | 说明 |
|------|-----|----------|----|----|
| Baseline RF | **0.9579** | **0.8755** | **67** | 当前最佳 |
| GAT Baseline | ~0.95-0.96 | ~0.88-0.90 | ~60-70 | 图结构baseline |
| GAT+物理属性 | ~0.96-0.97 | ~0.89-0.91 | ~55-65 | 辅助信号有帮助 |
| SMARTS+微调 | **~0.97-0.98** | **~0.91-0.93** | **~50-60** | 最强性能 |

### 关键改进

1. **降低FP**: 从67降到~50 (SMARTS预训练)
2. **提高AUC**: 从0.9579到0.97-0.98
3. **增强泛化**: 预训练学到化学知识

---

## 故障排除

### 常见问题

**Q: GAT训练很慢？**
- 使用GPU: 自动检测CUDA
- 减少batch_size: 64→32
- 减少epochs: 60→40

**Q: 内存不足？**
- 减少batch_size
- 使用更小的GAT模型: hidden=128→64

**Q: CUDA不可用？**
- 会自动切换到CPU
- 时间会显著增加(2-3倍)

**Q: 中断了如何恢复？**
- 所有模型都会保存checkpoint
- 重新运行相同命令即可继续

---

## 联系与支持

**问题报告**: 在项目目录下创建issue

**文档更新**:
- `docs/BASELINE_FINAL_REPORT.md` (baseline报告)
- 本文件 (pipeline指南)

**代码位置**:
- Pipeline: `run_gnn_pipeline.py`
- Baseline: `artifacts/models/seed_0_full/baseline/`

---

**最后更新**: 2026-01-21
**版本**: 1.0
**状态**: Baseline完成，GNN Pipeline准备就绪
