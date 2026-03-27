# 回归数据集利用指南

## 数据集结构分析

### Classification (7807样本)
| Group | 样本数 | 有logBB | BBB+/-标签 |
|-------|--------|---------|-----------|
| A     | 1058   | ✓       | ✓         |
| B     | 3621   | ✗       | ✓         |
| C     | 3077   | ✗       | ✓         |
| D     | 51     | ✗       | ✓         |

**关键发现**: C group没有logBB值，但有BBB+/-标签！可以大幅扩充分类任务数据。

### Regression (1058样本)
| Group | 样本数 | 说明                   |
|-------|--------|------------------------|
| A     | 243    | 高质量logBB测量         |
| B     | 663    | 中等质量logBB测量       |
| C     | 3      | 低质量（基本可忽略）   |
| D     | 149    | 特殊类别               |

**关键发现**: Regression = Classification的A group，但按新标准重新分组。

---

## 推荐策略

### 策略1: 扩展分类任务 (A,B,C)

**适用**: 纯分类任务，追求最大数据量

```bash
# 使用A,B,C三个group (7756样本，比A,B的4679多66%)
python scripts/01b_prepare_splits_extended.py \
    --task classification \
    --groups A,B,C \
    --seed 0

# 后续特征化和训练与原来相同
python scripts/02_featurize_all.py --seed 0
python scripts/03_run_baselines.py --seed 0 --feature morgan
```

**优势**:
- 数据量增加66% (4679→7756)
- C group虽然无logBB，但有BBB+/-标签
- 可能提高分类模型泛化能力

**劣势**:
- 无法评估logBB预测性能

---

### 策略2: 独立回归任务

**适用**: 需要预测连续logBB值

```bash
# 使用regression数据集的A,B groups (906样本)
python scripts/01b_prepare_splits_extended.py \
    --task regression \
    --groups A,B \
    --seed 0

# 训练回归模型
python scripts/03c_run_regression.py \
    --seed 0 \
    --feature morgan \
    --model tree    # 或 gat

# 评估指标: MAE, RMSE, R², ±0.5准确率
```

**输出**: `artifacts/metrics/regression_results.csv`

**评估指标**:
- MAE: 平均绝对误差
- RMSE: 均方根误差
- R²: 决定系数
- acc_within_05: 预测值在±0.5范围内的比例

---

### 策略3: 领域迁移实验

**适用**: 测试模型跨group定义的泛化能力

```bash
# 准备领域迁移split
python scripts/01b_prepare_splits_extended.py \
    --task domain_shift \
    --seed 0

# 结构:
# train/val: Classification group A (原定义)
# test: Regression A/B/D (新定义)
```

**用途**:
- 验证模型是否过拟合特定group定义
- 评估真实场景下的泛化性能

**测试集划分**:
- `test_reg_group_A.csv`: Regression A (243样本)
- `test_reg_group_B.csv`: Regression B (663样本)
- `test_reg_group_D.csv`: Regression D (149样本)
- `test_reg_all.csv`: 全部regression (1055样本)

---

### 策略4: 多任务学习 (推荐)

**适用**: 同时优化分类和回归

**设计**:
```
输入: Classification A,B groups
任务1: BBB+/- 分类 (A,B 全部样本)
任务2: logBB 回归 (仅 A group，B masked)
```

```bash
# 训练多任务GAT
python scripts/04b_train_multitask_cls_reg.py \
    --seed 0 \
    --lambda_cls 1.0 \
    --lambda_reg 0.5

# 输出:
# - artfiacts/models/gat_multitask_cls_reg/
# - artifacts/metrics/gat_multitask_cls_reg.csv
```

**优势**:
- 共享特征表示
- 分类任务从回归信号中学习
- 回归任务从更多样本中受益

**超参数**:
- `--lambda_cls`: 分类损失权重 (默认1.0)
- `--lambda_reg`: 回归损失权重 (默认0.5)

---

## 实验对比矩阵

| 策略            | 训练集大小 | 分类指标 | 回归指标 | 推荐场景          |
|-----------------|-----------|----------|----------|-------------------|
| 原始(A,B)       | 4679      | ✓        | ✗        | 基线对比          |
| 扩展(A,B,C)     | 7756      | ✓        | ✗        | 纯分类任务        |
| 独立回归        | 906       | ✗        | ✓        | 纯回归任务        |
| 多任务(A,B)     | 4679      | ✓        | ✓        | 综合性能 (推荐)   |
| 领域迁移        | 1058      | ✓        | ✗        | 泛化性测试        |

---

## 完整实验流程

### 实验1: 对比不同group范围的影响

```bash
# (1) 原始: A,B only
python scripts/01_prepare_splits.py --seed 0
python scripts/02_featurize_all.py --seed 0
python scripts/03_run_baselines.py --seed 0 --feature morgan

# (2) 扩展: A,B,C
python scripts/01b_prepare_splits_extended.py --task classification --groups A,B,C --seed 0
python scripts/02_featurize_all.py --seed 0  # 需要修改以支持新的split路径
python scripts/03_run_baselines.py --seed 0 --feature morgan  # 同上

# 对比 AUC, AUPRC, F1
```

### 实验2: 回归基准测试

```bash
# 准备回归数据
python scripts/01b_prepare_splits_extended.py --task regression --groups A,B --seed 0

# Tree-based回归
python scripts/03c_run_regression.py --seed 0 --feature morgan --model tree
python scripts/03c_run_regression.py --seed 0 --feature desc --model tree

# GAT回归
python scripts/03c_run_regression.py --seed 0 --feature graph --model gat

# 对比 MAE, R²
```

### 实验3: 多任务vs单任务

```bash
# 单任务分类
python scripts/06_finetune_bbb_from_smarts.py --seed 0

# 多任务
python scripts/04b_train_multitask_cls_reg.py --seed 0 --lambda_cls 1.0 --lambda_reg 0.5

# 对比分类性能: 是否从回归信号中获益？
```

### 实验4: 领域迁移

```bash
# 准备domain shift splits
python scripts/01b_prepare_splits_extended.py --task domain_shift --seed 0

# 在classification A上训练
python scripts/06_finetune_bbb_from_smarts.py --seed 0

# 在regression A/B/D上测试
# (需要修改评估脚本以支持多个test set)
```

---

## 代码修改说明

### `02_featurize_all.py` 需要支持新路径

当前版本固定使用 `data/splits/seed_{seed}/`，需要修改以支持:

```python
# 添加参数
ap.add_argument("--split_dir", type=str, default=None)

# 修改路径逻辑
if args.split_dir:
    split_dir = Path(args.split_dir)
else:
    split_dir = P.data_splits / f"seed_{args.seed}"
```

### `03_run_baselines.py` 同上

---

## 结果解读

### 分类指标
- **AUC**: ROC曲线下面积，越大越好
- **AUPRC**: PR曲线下面积，适合不平衡数据
- **F1**: precision和recall的调和平均

### 回归指标
- **MAE < 0.5**: 预测误差在半对数单位内 (不错)
- **MAE < 0.3**: 预测误差在0.3对数单位内 (很好)
- **R² > 0.6**: 解释60%以上方差 (良好)

---

## 推荐实验顺序

1. **快速验证**: 扩展到A,B,C，看分类性能提升
2. **回归基线**: 建立logBB预测基线 (tree models)
3. **多任务学习**: 尝试联合优化
4. **领域迁移**: 验证泛化性

---

## 文件清单

新增文件:
- `scripts/01b_prepare_splits_extended.py`: 数据准备
- `scripts/03c_run_regression.py`: 回归模型
- `scripts/04b_train_multitask_cls_reg.py`: 多任务训练
- `src/pretrain/train_gat_multitask_cls_reg.py`: 多任务GAT

输出目录:
```
data/splits/
├── cls_extended_groupsABC_seed_0/    # 扩展分类
├── reg_groupsAB_seed_0/              # 回归
└── domain_shift_seed_0/              # 领域迁移

artifacts/models/
├── regression/
│   ├── seed_0/morgan/tree/
│   └── seed_0/gat/
└── gat_multitask_cls_reg/
    └── seed_0/cls1.0_reg0.5/

artifacts/metrics/
├── regression_results.csv
└── gat_multitask_cls_reg.csv
```
