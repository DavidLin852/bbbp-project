# 归档目录说明

> 本目录存放项目历史中积累的文档和图片，供后续整理或参考。

## 目录结构

```
archive/
├── root_backup/                # 根目录归档脚本
│
├── docs_backup/               # 文档归档
│   ├── outputs_docs/           # 原outputs/docs目录的归档
│   │   └── reports/            # 28+ 个历史报告文档
│   └── *.md                    # 原docs/目录的归档文档
│
├── scripts_backup/            # 脚本归档
│
├── images_backup/             # 图片归档
│   ├── analysis_nokey/         # 大量分析图表（已归档）
│   ├── figures/                # 早期分析图表
│   ├── metrics/               # 指标图表
│   └── training/              # 训练过程图表
│
└── pretraining_analysis/      # 预训练分析数据
```

## 归档内容

### 根目录脚本 (root_backup/)
| 文件 | 说明 |
|------|------|
| app_bbb_predict_complete.py | 备份的主应用 |
| download_chembl.py | ChEMBL数据下载 |
| download_zinc20_real.py | ZINC数据下载 |
| explore_pretraining_data.py | 预训练数据探索 |
| generate_diverse_molecules.py | 分子生成 |
| generate_plots_from_export.py | 绘图脚本 |
| get_real_zinc_data.py | ZINC数据获取 |
| pretrain_zinc20.py | ZINC预训练 |
| run_gnn_pipeline.py | GNN流水线 |
| test_prediction.py | 测试脚本 |

### 已删除的文件
- SiC晶体材料相关Excel文件 (已删除)
- 翻译缓存文件 (已删除)

## 归档原因

1. **文档冗余**: 历史报告中很多内容重复，已整理到关键文档中
2. **图片过多**: 大量中间结果图、分析图已归档
3. **后续整理**: 保留归档文件，可供后续选择性恢复

## 保留的关键内容

### 文档 (docs/)
- `USER_GUIDE.md` - 完整使用指南
- `IMPROVEMENTS.md` - 改进建议
- `DOCKER_GUIDE.md` - Docker部署指南

### 文档 (outputs/docs/)
- `PROJECT_STRUCTURE.md` - 项目结构说明
- `ORGANIZATION_SUMMARY.md` - 目录组织总结

### 图片 (outputs/images/)
- `analysis/molecule_predictions_bar_chart.png` - 19个分子预测柱状图
- `model_comparison/FINAL_COMPREHENSIVE_HEATMAP.png` - DPI 600综合热力图
- `model_comparison/AUC_vs_F1_SCATTER_OPTIMIZED.png` - AUC-F1散点图
- `smarts_viz/` - 70+ SMARTS子结构可视化
- `metrics/` - ROC曲线等指标图
- `training/` - 训练过程图

## 恢复归档文件

如需恢复归档的文件，可从本目录复制回原位置：

```bash
# 示例：恢复某个归档的文档
cp archive/docs_backup/active_learning.md docs/

# 示例：恢复归档的图片
cp archive/images_backup/analysis_nokey/some_chart.png outputs/images/analysis/
```

## 清理建议

后续可以考虑：
1. 删除超过6个月的历史报告
2. 只保留最终版本的图片
3. 将关键内容整合到主文档中

---

*最后更新: 2026-02-24*
