# 实验1总结：对比 A,B vs A,B,C 分组

## 实验完成时间
2026-01-21

## 实验目的
测试添加 Group C 数据是否会提升 BBB 渗透性分类性能

## 数据对比

| 指标       | A,B   | A,B,C | 变化     |
|-----------|-------|-------|----------|
| 总样本数   | 4679  | 7756  | +65.8%   |
| 训练集     | 3743  | 6204  | +65.7%   |
| 测试集     | 468   | 776   | +65.8%   |
| **正例率** | 0.730 | 0.638 | -0.092   |

**关键发现**：Group C 的正例率明显更低（更多 BBB- 样本）

## 实验结果

### 测试集性能对比

| 模型 | 数据集 | AUC    | AUPRC  | F1     |
|------|--------|--------|--------|--------|
| RF   | A,B    | 0.9724 | 0.9885 | 0.9388 |
| RF   | A,B,C  | 0.9681 | 0.9781 | 0.9213 |
| **变化** | - | **-0.004** | **-0.010** | **-0.018** |
| | | | | |
| XGB  | A,B    | **0.9705** | **0.9880** | **0.9498** |
| XGB  | A,B,C  | 0.9577 | 0.9692 | 0.9146 |
| **变化** | - | **-0.013** | **-0.019** | **-0.035** |
| | | | | |
| LGBM | A,B    | 0.9657 | 0.9852 | 0.9495 |
| LGBM | A,B,C  | 0.9610 | 0.9735 | 0.9238 |
| **变化** | - | **-0.005** | **-0.012** | **-0.026** |

### 结论
❌ **添加 Group C 反而降低了所有模型的性能**

## 原因分析

### 1. 标签分布偏移
- Group A,B: 正例率 73%
- Group C: 正例率约 64%
- 添加 C 后，整体正例率降至 63.8%
- 模型需要学习新的决策边界

### 2. Group C 可能是"困难样本"
- 3077 个 C group 分子在化学性质上可能不同
- 可能包含渗透性边界模糊的化合物
- 这些样本增加了分类难度

### 3. 实验条件差异
- 不同 group 可能来自不同实验批次
- 测量方法或条件可能有系统差异
- 导致 domain shift 问题

## 建议

### 短期（立即执行）
- ✅ **继续使用 A,B groups 作为主数据集**
- ✅ **最佳模型**: XGB on A,B (AUC=0.9705, F1=0.9498)

### 中期（进一步探索）
1. **化学分析**:
   - 分析 Group C 分子的分子描述符分布
   - 可视化 PCA/t-SNE 看是否有聚类差异

2. **尝试不同组合**:
   - Group A,C (跳过 B)
   - Group A only (高质量数据)

3. **多任务学习**:
   - 分类任务使用 A,B,C
   - 回归任务使用 A (有 logBB)

### 长期（高级策略）
1. **Domain Adaptation**: 在 A,B 上训练，针对 C 做适应
2. **Group-Aware Models**: 将 group 作为特征输入
3. **Ensemble**: 分别训练 A,B 和 C 模型，然后集成

## 文件清单

### 数据文件
```
data/splits/
├── seed_0_original_AB/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── seed_0_extended_ABC/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

### 特征文件
```
artifacts/features/
├── seed_0_original_AB/
│   ├── morgan_2048.npz
│   └── meta.csv
└── seed_0_extended_ABC/
    ├── morgan_2048.npz
    └── meta.csv
```

### 模型文件
```
artifacts/models/exp1/
├── A,B/
│   ├── RF_seed0.joblib
│   ├── XGB_seed0.joblib
│   └── LGBM_seed0.joblib
└── A,B,C/
    ├── RF_seed0.joblib
    ├── XGB_seed0.joblib
    └── LGBM_seed0.joblib
```

### 结果文件
```
artifacts/metrics/
└── exp1_baseline_comparison.csv
```

## 下一步

1. **运行实验 2**: 回归基线测试
   ```bash
   python scripts/01b_prepare_splits_extended.py --task regression --groups A,B --seed 0
   python scripts/03c_run_regression.py --seed 0 --feature morgan --model tree
   ```

2. **化学分析 Group C**:
   - 计算分子描述符
   - 可视化分布差异

3. **多任务学习**:
   - 同时优化分类和回归

## 关键经验教训

> **更多数据 ≠ 更好性能**
>
> 当新数据与原数据分布不同时（如标签分布偏移），
> 盲目合并可能损害性能。需要：
> 1. 分析新数据的特性
> 2. 理解分布差异的原因
> 3. 选择合适的融合策略（如 domain adaptation）
