# 🚀 Visualization Template - Quick Start Guide

## 选择您的工具

提供了两个版本的可视化模板：

1. **🌐 Web版本** (`visualization_template.html`) - 推荐！简单易用
2. **🐍 Python版本** (`visualization_template.py`) - 适合批量处理

---

## 🌐 方式1: Web版本 (推荐)

### 第1步: 打开模板

双击文件打开，或右键选择浏览器打开：
```
visualization_template.html
```

### 第2步: 选择数据来源

#### 选项A: 使用预设数据 (最快)

1. 点击 **"Preset 1: SMARTS Importance"** 标签页
2. 点击 **"✅ Load Preset 1 Data"** 按钮
3. 自动切换到可视化标签页

#### 选项B: 粘贴CSV数据

1. 点击 **"Data Entry"** 标签页
2. 在文本框中粘贴CSV数据，格式如下：
```csv
A,B,RF,0.984,0.938,0.980,0.958
A,B,RF+SMARTS,0.989,0.946,0.980,0.963
A,B,LGBM,0.964,0.944,0.941,0.943
```
3. 点击 **"📋 Load CSV Data"** 按钮

#### 选项C: 上传CSV文件

1. 点击 **"Data Entry"** 标签页
2. 点击 **"选择文件"** 按钮
3. 选择您的CSV文件

### 第3步: 生成图表

1. 切换到 **"Generate Charts"** 标签页
2. 点击 **"🎨 Generate All 4 Charts"** 生成所有图表
3. 或单独生成某个图表

### 第4步: 导出图表

点击 **"💾 Export All Charts (PNG)"** 一键导出所有图表

---

## 🐍 方式2: Python版本

### 使用预设数据1 (SMARTS重要性对比)

```bash
python visualization_template.py --preset 1
```

**输出:** 在 `outputs/visualization_template/` 生成4张图表

### 使用预设数据2 (数据集影响分析)

```bash
python visualization_template.py --preset 2
```

### 使用您自己的数据

```bash
python visualization_template.py --data path/to/your_data.csv
```

### 只生成某个图表

```bash
# 只生成热图
python visualization_template.py --preset 1 --chart heatmap

# 只生成AUC柱状图
python visualization_template.py --preset 1 --chart barplot
```

### 指定输出目录

```bash
python visualization_template.py --preset 1 --output-dir my_results/
```

---

## 📊 CSV数据格式

无论是Web版本还是Python版本，都使用相同的CSV格式：

```csv
Dataset,Model,AUC,Precision,Recall,F1
A,RF,0.916,0.957,0.968,0.963
A,RF+SMARTS,0.927,0.903,1.000,0.949
A,B,LGBM,0.964,0.944,0.941,0.943
```

**必需列:**
- `Dataset`: 数据集名称 (A, A,B, A,B,C, A,B,C,D)
- `Model`: 模型名称
- `AUC`: AUC值 (0.80-1.0)
- `Precision`: 精确率 (0.80-1.0)
- `Recall`: 召回率 (0.80-1.0)
- `F1`: F1分数 (0.80-1.0)

---

## 🎨 生成的4张图表

### 图1: 性能指标热图
- 4个子图展示AUC、Precision、Recall、F1
- 热图颜色从红到绿
- 所有模型和数据集的对比

### 图2: AUC柱状图
- 按数据集分组的柱状图
- 直接比较各模型的AUC
- 柱子上标注数值

### 图3: AUC vs F1散点图
- 横轴AUC，纵轴F1
- 不同形状和颜色区分模型
- 包含对角参考线

### 图4: 数据集复杂度影响
- 4条折线展示性能随数据集规模的变化
- 显示样本数量 (n=106, 496, 1060, 6296)
- 观察模型在不同规模数据集上的表现

---

## 🎯 两个预设示例的用途

### 预设1: SMARTS预训练重要性

**数据:** A,B数据集的8个模型

**比较:**
- RF vs RF+SMARTS
- LGBM vs LGBM+SMARTS
- XGB vs XGB+SMARTS
- GAT vs GAT+SMARTS

**目的:** 展示SMARTS特征增强的效果

**使用场景:**
- 说明预训练的重要性
- 量化性能提升 (ΔAUC)
- 支撑方法学部分

### 预设2: 数据集影响分析

**数据:** 4个数据集上的4个SMARTS模型

**模型:** RF+SMARTS, LGBM+SMARTS, XGB+SMARTS, GAT+SMARTS

**数据集:** A (106) → A,B (496) → A,B,C (1060) → A,B,C,D (6296)

**目的:** 观察模型性能随数据规模的变化

**使用场景:**
- 讨论数据规模对性能的影响
- 说明模型的可扩展性
- 选择最佳数据集规模

---

## 💡 使用建议

### 论文撰写

**主图:** 图1 (热图) 或 图2 (柱状图)
- 清晰展示所有模型的性能
- 适合作为Results部分的主图

**补充材料:** 图3 (散点图) 和 图4 (复杂度影响)
- 提供额外的分析视角
- 适合放在Supplementary Materials

### 演示汇报

**推荐顺序:**
1. 先展示图2 (柱状图) - 清晰直观
2. 再展示图3 (散点图) - 解释权衡
3. 最后展示图4 (趋势图) - 说明扩展性

**说明要点:**
- SMARTS预训练的效果 (预设1)
- 数据集规模的影响 (预设2)
- 不同模型的特点和适用场景

---

## 🔧 从您的项目数据生成图表

### 方式1: 使用已有CSV文件

```bash
# Web版本
1. 打开 visualization_template.html
2. Data Entry → 上传文件
3. 选择: outputs/model_comparison/complete_model_performance.csv
4. Generate Charts

# Python版本
python visualization_template.py \
    --data outputs/model_comparison/complete_model_performance.csv \
    --output-dir my_analysis/
```

### 方式2: 使用重新评估的数据

如果您运行了 `evaluate_missing_baselines.py`，可以使用生成的CSV：

```bash
python visualization_template.py \
    --data outputs/model_comparison/verified_model_performance.csv
```

---

## 🎨 颜色方案说明

模板使用统一的颜色方案：

| 模型类型 | 基础颜色 | SMARTS版本 |
|---------|---------|-----------|
| RF | 蓝色 #3498db | 深蓝 #1a5276 |
| LGBM | 绿色 #2ecc71 | 深绿 #1e8449 |
| XGB | 橙色 #f39c12 | 深橙 #a04000 |
| GAT | 紫色 #9b59b6 | 深紫 #6c3483 |

**规律:** 同一基础算法使用相同颜色，SMARTS版本使用更深色调

---

## ❓ 常见问题

### Q: Web版本在Chrome中打开但看不到图表？

A: 确保先加载数据：
1. 选择 Preset 1 或 Preset 2
2. 点击加载数据按钮
3. 再切换到 Generate Charts 标签页

### Q: Python版本报错 "ModuleNotFoundError"?

A: 安装所需依赖：
```bash
pip install pandas numpy matplotlib seaborn
```

### Q: 如何修改图表的颜色？

A:
- **Web版本:** 编辑HTML文件中的 `MODEL_COLORS` 对象
- **Python版本:** 编辑脚本中的 `MODEL_COLORS` 字典

### Q: 能否添加新的模型？

A: 可以！只需在CSV中添加新行：
```csv
A,B,MyNewModel,0.950,0.920,0.940,0.930
```

模板会自动识别并包含在图表中。

### Q: 导出的图片分辨率不够？

A:
- **Web版本:** 在代码中修改 `width` 和 `height` 参数
- **Python版本:** 已设置300 DPI，适合论文发表

---

## 📁 输出文件

使用模板后，您将得到：

```
outputs/visualization_template/
├── fig1_heatmap.png          # 热图
├── fig2_barplot.png          # AUC柱状图
├── fig3_scatter.png          # AUC vs F1散点图
├── fig4_complexity.png       # 数据集复杂度影响
└── performance_summary.csv   # 性能排名表
```

---

## 🎓 下一步

1. **熟悉模板:** 使用预设数据生成图表，了解各个功能
2. **使用您的数据:** 替换为您自己的实验结果
3. **定制化:** 调整颜色、图表大小等参数
4. **整合到论文:** 将生成的图表导入您的论文文档

---

## 📞 需要帮助？

1. 查看 `VISUALIZATION_TEMPLATE_README.md` 获取详细文档
2. 检查您的CSV格式是否正确
3. 确认数据值在合理范围内 (0.80-1.0)

---

**祝您使用愉快！** 🎉

快速生成高质量图表，支撑您的研究成果！
