# Streamlit平台启动指南

## ✅ 页面结构已修复

现在平台有正确的多页面结构：

```
bbb_project/
├── app_bbb_predict.py          # 主页（入口）
└── pages/
    ├── 0_prediction.py          # 预测页面（16个模型）
    ├── 1_smarts_analysis.py     # SMARTS分析页面
    └── 2_model_comparison.py    # 模型对比页面（16个模型）
```

---

## 🚀 启动步骤

### 1. 启动Streamlit

```bash
cd E:\PythonProjects\bbb_project
streamlit run app_bbb_predict.py
```

### 2. 访问地址

浏览器自动打开：http://localhost:8501

### 3. 页面导航

在Streamlit页面左侧可以看到菜单：

- **🏠 Home** - 平台介绍和快速开始
- **🔍 Prediction** - 预测页面（16个模型可选）
- **🧬 SMARTS Analysis** - SMARTS子结构分析
- **📊 Model Comparison** - 16个模型对比和筛选

---

## 📱 各页面功能

### 主页 (Home)
- 平台介绍
- 16个模型统计
- 最佳模型展示
- 快速开始指南

### 预测页面 (Prediction)
**功能**:
- 选择数据集（A, A+B, A+B+C, A+B+C+D）
- 选择模型（RF, XGB, LGBM, GAT+SMARTS）
- 单个分子预测
- 批量CSV预测
- **16个模型组合全部可用** ✅

### SMARTS分析页面 (SMARTS Analysis)
**功能**:
- 查看化学子结构重要性
- 正向/负向SMARTS展示
- 2D结构可视化
- 统计信息

### 模型对比页面 (Model Comparison)
**功能**:
- 查看全部16个模型性能
- 自定义筛选（数据集、模型、性能范围）
- 交互式可视化（柱状图、散点图、热力图）
- 智能推荐
- 数据导出

---

## 🎯 使用示例

### 示例1: 预测BBB渗透性

```
1. 点击侧边栏 "Prediction"
2. 选择数据集: A+B (Recommended)
3. 选择模型: XGBoost (AUC=0.9694)
4. 输入SMILES: CCO
5. 点击 "Predict"
6. 查看预测结果
```

### 示例2: 对比所有16个模型

```
1. 点击侧边栏 "Model Comparison"
2. 默认显示全部16个模型
3. 查看性能对比表格
4. 使用筛选器自定义选择
5. 查看可视化图表
6. 点击"查看全部16个模型"查看完整列表
```

### 示例3: 筛选特定模型

```
1. 进入 Model Comparison 页面
2. 侧边栏选择数据集: A+B
3. 侧边栏选择模型: XGB, LGBM
4. 调整AUC范围: (0.95, 1.0)
5. 查看筛选结果
```

---

## ⚠️ 常见问题

### Q1: 看不到页面导航？

**A**: 确保在`pages/`目录下启动Streamlit，页面文件命名正确：
- `0_prediction.py`
- `1_smarts_analysis.py`
- `2_model_comparison.py`

### Q2: 点击页面没反应？

**A**: 确保URL格式正确：
- Home: http://localhost:8501
- Prediction: http://localhost:8501/?page=prediction
- Model Comparison: http://localhost:8501/?page=model_comparison

### Q3: 端口被占用？

**A**: 使用其他端口启动：
```bash
streamlit run app_bbb_predict.py --server.port 8502
```

---

## 📊 16个模型快速参考

| 数据集 | RF | XGB | LGBM | GAT+SMARTS |
|--------|----|----|----|------------|
| **A** | ✅ | ✅ | ✅ | ✅ |
| **A+B** | ✅ | ✅ | ✅ | ✅ |
| **A+B+C** | ✅ | ✅ | ✅ | ✅ |
| **A+B+C+D** | ✅ | ✅ | ✅ | ✅ |

**最佳推荐**:
- 综合最佳: **XGB - A+B** (AUC=0.9694)
- 最高Precision: **LGBM - A** (Precision=0.9574, FP=4)
- 最高Recall: **GAT+SMARTS - A** (Recall=0.9892)

---

## ✅ 验证清单

- [x] 页面结构正确（3个页面文件）
- [x] 文件命名正确（0_, 1_, 2_）
- [x] 删除了冲突的旧页面
- [x] 所有16个模型可用
- [x] GAT模型加载问题已修复
- [x] 模型对比页面支持全部16个模型

---

**现在启动Streamlit，你应该能看到所有页面了！** 🎉

```bash
streamlit run app_bbb_predict.py
```
