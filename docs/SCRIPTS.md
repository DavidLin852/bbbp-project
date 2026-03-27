# Scripts 目录说明

## 核心脚本 (10个)

### 训练流程 (6个)

| 编号 | 脚本 | 功能 |
|------|------|------|
| 01 | `01_prepare_splits.py` | 数据划分 (train/val/test, 80:10:10) |
| 02 | `02_featurize_all.py` | 特征提取 (Morgan, MACCS, AtomPairs, FP2, Graph) |
| 03 | `03_run_baselines.py` | 训练基础模型 (RF, XGB, LGBM) |
| 04 | `04_run_gat_aux.py` | 训练GAT模型 (带辅助任务) |
| 05 | `05_pretrain_smarts.py` | SMARTS预训练 |
| 06 | `06_finetune_bbb_from_smarts.py` | BBB微调 (迁移学习) |

### 可视化脚本 (4个)

| 脚本 | 功能 | 输出 |
|------|------|------|
| `generate_final_comprehensive_heatmap.py` | 综合热力图 (DPI 600) | `FINAL_COMPREHENSIVE_HEATMAP.png` |
| `create_auc_f1_scatter.py` | AUC-F1 散点图 | `AUC_vs_F1_SCATTER_OPTIMIZED.png` |
| `visualize_molecule_predictions.py` | 分子预测柱状图 | `molecule_predictions_bar_chart.png` |
| `draw_all_smarts_from_json.py` | SMARTS结构可视化 | `smarts_viz/single/*.png` (70+张) |

---

## 完整训练流程

```bash
# 1. 数据准备
python scripts/01_prepare_splits.py --seed 0 --keep_groups "A,B"

# 2. 特征提取
python scripts/02_featurize_all.py --seed 0

# 3. 训练基础模型
python scripts/03_run_baselines.py --seed 0 --feature morgan

# 4. 训练GAT (可选)
python scripts/04_run_gat_aux.py --seed 0 --dataset A_B

# 5. SMARTS预训练 (可选)
python scripts/05_pretrain_smarts.py --seed 0 --dataset A_B

# 6. BBB微调 (可选)
python scripts/06_finetune_bbb_from_smarts.py --seed 0 --dataset A_B
```

---

## 生成核心图片

```bash
# 1. 综合热力图 (DPI 600)
python scripts/generate_final_comprehensive_heatmap.py

# 2. AUC-F1 散点图
python scripts/create_auc_f1_scatter.py

# 3. 分子预测图
python scripts/visualize_molecule_predictions.py

# 4. SMARTS结构图
python scripts/draw_all_smarts_from_json.py
```

---

## 归档脚本

其他脚本已归档到 `archive/scripts_backup/`，包括：
- 重复功能的训练脚本
- 解释性分析脚本 (SHAP, atom attribution)
- 消融实验脚本
- 降维分析脚本
- 辅助生成脚本

需要时可从归档中恢复。
