# BBB项目清理建议

## 📋 检查结果

项目中共发现：
- **根目录Python文件**: 32个
- **scripts目录**: 54个脚本
- ** outputs/model_comparison**: 多个重复的summary文件

---

## 🗑️ 建议删除的文件（分类）

### A. 根目录 - 临时/重复脚本（建议删除）

**可视化相关（已被generate_plots_from_export.py取代）:**
1. `generate_additional_plots.py` - 旧版可视化
2. `generate_final_plots.py` - 旧版可视化（数据有问题）
3. `generate_selected_plots.py` - 旧版可视化（数据有问题）
4. `visualize_model_comparison.py` - 旧版可视化

**分析脚本（一次性使用）:**
5. `analyze_smarts_enhanced.py` - 临时分析
6. `analyze_smarts_importance.py` - 临时分析
7. `evaluate_missing_baselines.py` - 已完成评估任务

**测试脚本:**
8. `test_ensemble.py` - 临时测试
9. `test_prediction.py` - 临时测试

**SMOTE相关（如不需要）:**
10. `plot_smote_comparison.py` - SMOTE临时脚本
11. `plot_smote_comprehensive.py` - SMOTE临时脚本

**其他临时脚本:**
12. `plot_complete_story.py` - 临时绘图
13. `plot_fixed_story.py` - 临时绘图
14. `ionizable_lipds.py` - 不相关脚本
15. `sdf_csv.py` - 临时转换脚本

### B. outputs/model_comparison - 重复文档（建议删除）

**重复的summary文件（保留最新版本）:**
16. `FINAL_SUMMARY.md` - 旧版summary
17. `FINAL_SUMMARY_v2.md` - 旧版summary
18. `all_model_performance.csv` - 旧版数据
19. `complete_model_performance.csv` - 旧版数据（旧版本）
20. `verified_model_performance.csv` - 旧版数据
21. `model_comparison_report.html` - 旧报告
22. `README.md` - 旧README

**保留的文件:**
- ✅ `CHARTS_GUIDE.md` - 最新图表说明
- ✅ `DATASET_SIZES_EXPLANATION.md` - 数据集说明
- ✅ `EXPORT_DATA_ANALYSIS.md` - 最新分析报告
- ✅ `performance_summary.csv` - 最新性能数据
- ✅ `fig*.png` - 所有6张图表

### C. 可视化模板（建议保留到独立目录）

这些文件是独立的可视化工具，建议移到独立目录：

23. `visualization_template.html`
24. `visualization_template.py`
25. `QUICKSTART.md`
26. `VISUALIZATION_TEMPLATE_README.md`
27. `TEMPLATE_CREATION_SUMMARY.md`

**建议操作**: 移动到 `tools/visualization_template/` 目录

### D. scripts目录 - 过时脚本（建议删除）

**重复的实验脚本:**
28. `scripts/01b_prepare_splits_extended.py` - 被01_prepare_splits.py取代
29. `scripts/01c_prepare_balanced_splits.py` - 平衡数据集（如不需要）
30. `scripts/02_featurize_dataset.py` - 被02_featurize_all.py取代
31. `scripts/03_run_baselines_shuffle.py` - shuffle版本（如不需要）
32. `scripts/03b_run_xgb_only.py` - 被03_run_baselines.py取代
33. `scripts/03c_run_regression.py` - 回归实验（如不需要）
34. `scripts/04b_train_multitask_cls_reg.py` - 多任务实验（如不需要）

**SMOTE相关（如不需要）:**
35. `scripts/08_train_smote.py`
36. `scripts/09_train_gat_smote.py`

**旧的SMRTS训练脚本（已被train_extended_models.py取代）:**
37. `scripts/train_all_extended_models.py`
38. `scripts/train_extended_models.py`
39. `scripts/train_gat_smarts_all_datasets.py`
40. `scripts/train_gat_smarts_in_memory.py`
41. `scripts/train_gat_smarts_remaining.py`
42. `scripts/train_gat_smarts_with_filtering.py`
43. `scripts/train_gat_baseline_remaining.py`

**评估/诊断脚本（已完成任务）:**
44. `scripts/diagnose_streamlit.py`
45. `scripts/evaluate_all_extended_models.py`
46. `scripts/evaluate_all_models.py`
47. `scripts/verify_platform_update.py`

**其他临时脚本:**
48. `scripts/batch_train_all_datasets.py`
49. `scripts/batch_train_simple.py`
50. `scripts/create_full_dataset.py`
51. `scripts/generate_missing_graphs.py`
52. `scripts/restart_streamlit.py`
53. `scripts/retrain_with_new_data.py`
54. `scripts/test_all_16_models.py`
55. `scripts/train_lgbm_all_datasets.py`
56. `scripts/15_dump_smarts_presence.py` - 被后续脚本取代
57. `scripts/16_counterfactual_attach_fragments.py` - 临时实验
58. `scripts/16_predict_lipidmaps_batch.py` - 临时预测
59. `scripts/17_filter_lipid_hits.py` - 临时过滤
60. `scripts/18_xgb_baselines.py` - 被pipeline取代

---

## ✅ 保留的核心文件

### 根目录（核心功能）:
- `app_bbb_predict.py` - ✅ Streamlit主应用
- `run_gnn_pipeline.py` - ✅ GNN训练流程
- `generate_plots_from_export.py` - ✅ 最新可视化脚本
- `CLAUDE.md` - ⚠️ 需要重写
- `README.md` - ⚠️ 需要更新
- `environment.yml` - ✅ 环境配置

### scripts目录（核心流程）:
- `scripts/01_prepare_splits.py` - ✅ 数据分割
- `scripts/02_featurize_all.py` - ✅ 特征计算
- `scripts/03_run_baselines.py` - ✅ Baseline模型
- `scripts/04_run_gat_aux.py` - ✅ GNN辅助任务
- `scripts/05_pretrain_smarts.py` - ✅ SMARTS预训练
- `scripts/06_finetune_bbb_from_smarts.py` - ✅ BBB微调
- `scripts/07_plot_final_roc.py` - ✅ ROC曲线
- `scripts/08_explain_atoms.py` - ✅ 原子解释
- `scripts/09_explain_smarts.py` - ✅ SMARTS解释
- `scripts/10_global_smarts_importance.py` - ✅ 全局重要性
- `scripts/10_global_smarts_importance_full.py` - ✅ 完整版本
- `scripts/11_global_smarts_interactions.py` - ✅ 交互分析
- `scripts/12_plot_global_smarts_importance.py` - ✅ 可视化
- `scripts/12_plot_smarts_heatmap_with_freq.py` - ✅ 热图
- `scripts/13_plot_smarts_interactions.py` - ✅ 交互可视化
- `scripts/14_predict_smiles.py` - ✅ SMILES预测
- `scripts/14_predict_smiles_cli.py` - ✅ CLI预测
- `scripts/15_ablate_smarts_on_model.py` - ✅ 消融实验

### pages目录:
- `pages/0_prediction.py` - ✅ 预测页面
- `pages/1_smarts_analysis.py` - ✅ SMARTS分析
- `pages/2_model_comparison.py` - ✅ 模型对比
- `pages/3_active_learning.py` - ✅ 主动学习

### src目录:
- 所有文件保留 ✅

---

## 📊 清理统计

**建议删除**: 60个文件
**建议移动**: 5个文件（到独立目录）
**需要重写**: 2个文件（CLAUDE.md, README.md）

---

## 🎯 清理后项目结构

```
bbb_project/
├── app_bbb_predict.py              # Streamlit主应用
├── run_gnn_pipeline.py             # GNN训练流程
├── generate_plots_from_export.py   # 模型对比可视化
├── CLAUDE.md                       # 项目文档（重写）
├── README.md                       # 项目说明（更新）
├── environment.yml                 # 环境配置
│
├── scripts/                        # 核心流程脚本 (20个)
│   ├── 01_prepare_splits.py
│   ├── 02_featurize_all.py
│   ├── 03_run_baselines.py
│   ├── ...
│
├── pages/                          # Streamlit页面
│   ├── 0_prediction.py
│   ├── 1_smarts_analysis.py
│   ├── 2_model_comparison.py
│   └── 3_active_learning.py
│
├── src/                            # 核心模块
│   ├── config.py
│   ├── baseline/
│   ├── featurize/
│   └── ...
│
├── outputs/
│   └── model_comparison/           # 清理后的输出
│       ├── fig1-fig6.png           # 6张图表
│       ├── performance_summary.csv # 性能数据
│       ├── CHARTS_GUIDE.md         # 图表说明
│       ├── DATASET_SIZES_EXPLANATION.md
│       └── EXPORT_DATA_ANALYSIS.md
│
└── tools/                          # 新建：独立工具目录
    └── visualization_template/     # 可视化模板
        ├── visualization_template.html
        ├── visualization_template.py
        ├── QUICKSTART.md
        └── ...
```

---

## ❓ 请确认

**我建议执行以下操作：**

1. ✅ 删除根目录的15个临时/重复脚本
2. ✅ 删除scripts目录的45个过时脚本
3. ✅ 清理outputs/model_comparison中的7个旧文件
4. ✅ 将5个可视化模板文件移动到tools/visualization_template/
5. ✅ 重写CLAUDE.md
6. ✅ 更新README.md

**是否同意执行这些清理操作？**

您可以说：
- "全部执行" - 执行所有清理
- "分批执行" - 我先执行部分，您确认后再继续
- "修改方案" - 告诉我哪些要保留/删除

请告诉我您的决定！
