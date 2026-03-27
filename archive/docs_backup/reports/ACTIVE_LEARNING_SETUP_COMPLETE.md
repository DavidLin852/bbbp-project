# Active Learning模块部署完成

## ✅ 已完成的工作

### 1. 创建了Active Learning页面
**文件**: `pages/3_active_learning.py`

包含三个主要标签页：
- **Input & Predict**: 输入SMILES，验证，预测，手动标注，保存
- **View Database**: 浏览已保存的分子，筛选，导出
- **Retrain Models**: 查看统计信息，重新训练模型

### 2. 更新了重新训练脚本
**文件**: `scripts/retrain_with_new_data.py`

新增功能：
- 支持从Active Learning缓存加载数据
- 自动合并原始训练数据和新标注数据
- 去除重复分子
- 支持数据集：A, A_B, A_B_C, A_B_C_D

### 3. 创建了使用文档
**文件**: `docs/ACTIVE_LEARNING_GUIDE.md`

包含：
- 快速开始指南
- 完整工作流程示例
- 常见问题解答
- 最佳实践

---

## 🚀 如何使用

### 启动Streamlit

```bash
# 方法1: 使用默认��口8501
streamlit run app_bbb_predict.py

# 方法2: 使用端口8502 (如果8501被占用)
streamlit run app_bbb_predict.py --server.port 8502
```

### 访问Active Learning页面

1. 打开浏览器访问 http://localhost:8501 (或 8502)
2. 在左侧导航栏点击 **"Active Learning"**
3. 开始添加新分子！

### 典型工作流程

```
1. 输入SMILES
   ↓
2. 验证格式 & 检查重复
   ↓
3. 模型预测
   ↓
4. 手动确认/修改标签
   ↓
5. 保存到数据库
   ↓
6. 重复1-5步骤，积累20-100个新分子
   ↓
7. 重新训练模型
   ↓
8. 评估新模型性能
```

---

## 📊 数据存储位置

### Active Learning数据库
```
artifacts/active_learning_cache/new_molecules_{dataset}.csv
```

示例：
- `new_molecules_A.csv`
- `new_molecules_A_B.csv`
- `new_molecules_A_B_C.csv`
- `new_molecules_A_B_C_D.csv`

### 重新训练后的模型
```
artifacts/models/{model_version}_seed_0/
├── baseline/
│   ├── RF_seed0.joblib
│   ├── XGB_seed0.joblib
│   └── LGBM_seed0.joblib
├── gnn_info.json
└── training_summary.json
```

---

## 📝 重新训练命令

### 使用Active Learning数据重新训练

```bash
# A_B数据集
python scripts/retrain_with_new_data.py \
    --dataset_name active_learning_A_B \
    --model_version v2_al \
    --seed 0

# A_B_C数据集
python scripts/retrain_with_new_data.py \
    --dataset_name active_learning_A_B_C \
    --model_version v3_al \
    --seed 0

# A_B_C_D数据集
python scripts/retrain_with_new_data.py \
    --dataset_name active_learning_A_B_C_D \
    --model_version v4_al \
    --seed 0
```

---

## 🎯 支持的数据集和模型

### 数据集 (4个)
- A (Group A only)
- A_B (Group A + B)
- A_B_C (Group A + B + C)
- A_B_C_D (Group A + B + C + D)

### 每个数据集的模型 (4个)
- RF (Random Forest)
- XGB (XGBoost)
- LGBM (LightGBM)
- GAT+SMARTS (Graph Attention Network)

**总计**: 16个模型组合

---

## ✨ 主要功能

### 1. SMILES验证
- RDKit格式检查
- 分子标准化
- 计算分子属性 (MW, logP, TPSA等)

### 2. 重复检查
- 检查是否已在训练数据中
- 使用Canonical SMILES进行精确匹配
- 显示重复分子所在数据集

### 3. 多模型预测
- 支持16个模型组合
- 显示预测结果和置信度
- 支持模型选择和数据集选择

### 4. 手动标注
- 可以确认或修改预测结果
- 添加注释说明
- 保存时自动记录时间戳

### 5. 数据库管理
- 查看所有已保存分子
- 按标签筛选 (BBB+, BBB-)
- 导出为CSV格式
- 显示统计信息

### 6. 模型重新训练
- 自动合并原始数据和新数据
- 去除重复分子
- 训练RF/XGB/LGBM三个模型
- 评估并保存新模型

---

## 📖 相关文档

- **Active Learning使用指南**: `docs/ACTIVE_LEARNING_GUIDE.md`
- **Streamlit修复指南**: `docs/STREAMLIT_FIX_GUIDE.md`
- **项目说明**: `CLAUDE.md`

---

## ⚠️ 注意事项

### 1. 端口使用
- 如果8501端口被占用，使用8502端口
- 或者先停止旧的Streamlit进程

### 2. 数据备份
- 定期备份 `artifacts/active_learning_cache/` 目录
- 重新训练前备份现有模型

### 3. GNN模型训练
- 重新训练脚本不会自动训练GNN模型
- 需要手动运行: `python run_gnn_pipeline.py --custom_dataset v2_al --seed 0`
- GNN训练时间较长 (10-30分钟)

### 4. 新分子数量
- 建议至少积累20个新分子再重新训练
- 最佳: 50-100个分子
- 保持BBB+和BBB-的平衡

---

## 🐛 故障排除

### 问题1: Active Learning页面不显示

**解决**:
```bash
# 清除Streamlit缓存
python -c "from pathlib import Path; import shutil; shutil.rmtree(Path.home() / '.streamlit')"

# 重新启动
streamlit run app_bbb_predict.py --server.port 8502
```

### 问题2: 保存到数据库失败

**检查**:
- 确保选择了正确的数据集
- 查看终端错误信息
- 检查是否有写入权限

### 问题3: 重新训练后模型没有更新

**原因**: Streamlit可能需要重启才能加载新模型

**解决**:
1. 停止Streamlit (Ctrl+C)
2. 重新启动
3. 或在代码中修改模型路径

---

## 🎉 开始使用

现在你可以在8502端口使用完整的Active Learning功能了！

```bash
streamlit run app_bbb_predict.py --server.port 8502
```

访问: http://localhost:8502

点击左侧导航栏的 **"Active Learning"** 开始添加新分子！

---

**创建日期**: 2025-01-26
**版本**: v1.0
**作者**: Claude Code
