# 图片生成清单

> 生成所有图片所需的脚本和命令

---

## 一、核心图片 (必生成)

### 1. 模型对比热力图

| 图片 | 脚本 | 命令 |
|------|------|------|
| FINAL_COMPREHENSIVE_HEATMAP.png (DPI 600) | `generate_final_comprehensive_heatmap.py` | `python scripts/generate_final_comprehensive_heatmap.py` |

### 2. AUC-F1 散点图

| 图片 | 脚本 | 命令 |
|------|------|------|
| AUC_vs_F1_SCATTER_OPTIMIZED.png | `create_auc_f1_scatter.py` | `python scripts/create_auc_f1_scatter.py` |

### 3. 分子预测图

| 图片 | 脚本 | 命令 |
|------|------|------|
| molecule_predictions_bar_chart.png | `visualize_molecule_predictions.py` | `python scripts/visualize_molecule_predictions.py` |

---

## 二、可选图片

### 1. SMARTS 相关

| 图片 | 脚本 | 命令 |
|------|------|------|
| all_smarts_grid.png | `draw_all_smarts_from_json.py` | `python scripts/draw_all_smarts_from_json.py` |
| smarts_viz/single/*.png (70+张) | `draw_all_smarts_from_json.py` | `python scripts/draw_all_smarts_from_json.py` |
| smarts_global_heatmap.png | `12_plot_global_smarts_importance.py` | `python scripts/12_plot_global_smarts_importance.py` |
| smarts_global_heatmap_with_freq.png | `12_plot_smarts_heatmap_with_freq.py` | `python scripts/12_plot_smarts_heatmap_with_freq.py` |
| fig4_smarts_interaction_heatmap.png | `13_plot_smarts_interactions.py` | `python scripts/13_plot_smarts_interactions.py` |
| fig5_top_smarts_interactions.png | `13_plot_smarts_interactions.py` | `python scripts/13_plot_smarts_interactions.py` |

### 2. 降维可视化

| 图片 | 脚本 | 命令 |
|------|------|------|
| tsne_*.png | `16_tsne_analysis.py` | `python scripts/16_tsne_analysis.py --feature morgan` |
| pca_*.png | `17_advanced_dim_reduction.py` | `python scripts/17_advanced_dim_reduction.py --methods pca` |

### 3. ROC曲线

| 图片 | 脚本 | 命令 |
|------|------|------|
| roc_baselines_seed0_morgan.png | `03_run_baselines.py` | `python scripts/03_run_baselines.py --seed 0 --feature morgan` |
| final_roc_bbb.png | `07_plot_final_roc.py` | `python scripts/07_plot_final_roc.py` |

### 4. 消融实验

| 图片 | 脚本 | 命令 |
|------|------|------|
| ablation_mask_*.png | `15_ablate_smarts_on_model.py` | `python scripts/15_ablate_smarts_on_model.py` |
| ablation_cut_*.png | `15_ablate_smarts_on_model.py` | `python scripts/15_ablate_smarts_on_model.py` |

### 5. 其他

| 图片 | 脚本 | 命令 |
|------|------|------|
| comparison_seed*.png | `18_compare_methods.py` | `python scripts/18_compare_methods.py` |
| complete_model_heatmap.png | `generate_model_heatmap.py` | `python scripts/generate_model_heatmap.py` |
| multi_metric_heatmap.png | `generate_multi_metric_heatmap.py` | `python scripts/generate_multi_metric_heatmap.py` |

---

## 三、快速生成命令

### 生成所有核心图片

```bash
# 1. 综合热力图 (DPI 600)
python scripts/generate_final_comprehensive_heatmap.py

# 2. AUC-F1 散点图
python scripts/create_auc_f1_scatter.py

# 3. 分子预测图
python scripts/visualize_molecule_predictions.py
```

### 生成所有图片

```bash
# SMARTS 结构可视化
python scripts/draw_all_smarts_from_json.py

# 降维分析
python scripts/16_tsne_analysis.py --feature morgan
python scripts/17_advanced_dim_reduction.py --methods pca tsne

# ROC 曲线
python scripts/03_run_baselines.py --seed 0 --feature morgan
python scripts/07_plot_final_roc.py

# 消融实验
python scripts/15_ablate_smarts_on_model.py

# 其他分析
python scripts/18_compare_methods.py
python scripts/generate_model_heatmap.py
python scripts/generate_multi_metric_heatmap.py
```

---

## 四、输出目录

所有图片输出到 `outputs/images/` 目录：

```
outputs/images/
├── model_comparison/    # 模型对比图
├── analysis/            # 分析图
├── smarts_viz/         # SMARTS可视化
│   └── single/         # 单个SMARTS结构
├── metrics/           # 指标图 (ROC等)
├── training/          # 训练过程图
└── IMAGE_INDEX.md     # 图片索引
```

---

## 五、注意事项

1. **DPI设置**: 核心图片使用 DPI 600，其他默认 DPI 300
2. **字体**: 使用 Times New Roman
3. **依赖**: 确保模型训练完成，artifacts 目录有数据
4. **顺序**: 建议先运行模型训练，再生成图片
