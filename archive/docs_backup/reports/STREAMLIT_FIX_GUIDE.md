# Streamlit多页面问题修复指南

## ✅ 诊断结果

根据诊断脚本检查：
- ✅ pages目录存在
- ✅ 3个页面文件命名正确（0_, 1_, 2_）
- ✅ 所有页面语法正确
- ✅ 主入口文件存在

**结论**: 文件结构完全正确！

---

## 🔧 解决方案

### 方案1: 清除Streamlit缓存并重启

```bash
# 1. 停止当前运行的Streamlit (Ctrl+C)

# 2. 清除Streamlit��存
python -c "
from pathlib import Path
import shutil
import sys

cache = Path.home() / '.streamlit'
print(f'清理缓存目录: {cache}')
if cache.exists():
    shutil.rmtree(cache)
    print('缓存已清理')
else:
    print('缓存目录不存在')
"

# 3. 重新启动
streamlit run app_bbb_predict.py
```

### 方案2: 使用不同端口

```bash
# 使用端口8502启动
streamlit run app_bbb_predict.py --server.port 8502
```

### 方案3: 强制刷新浏览器

1. 停止Streamlit
2. 清除浏览器缓存
3. 重新启动Streamlit
4. 在浏览器中按 Ctrl+Shift+R 强制刷新

---

## 🎯 如何正确访问页面

### Streamlit页面导航方式

**方式1: 使用侧边栏菜单** ✅ 推荐
1. Streamlit启动后，页面左侧会有导航栏
2. 点击不同的页面名称即可切换

**方式2: 使用URL**
```
主页: http://localhost:8501
预测: http://localhost:8501/?page=prediction
对比: http://localhost:8501/?page=model_comparison
```

**注意**: 页面名称对应关系：
- 主页 → app_bbb_predict.py
- Prediction → pages/0_prediction.py
- SMARTS Analysis → pages/1_smarts_analysis.py
- Model Comparison → pages/2_model_comparison.py

---

## 🖼️ 截图检查清单

启动Streamlit后，你应该看到：

### 1. 主页
- 左侧导航栏显示4个选项
- 主页标题：🧪 BBB Permeability Prediction Platform
- 显示16个模型的统计信息
- 功能卡片和链接

### 2. 点击侧边栏 "Prediction"
- 页面标题：🔍 BBB Prediction Platform
- 侧边栏有数据集选择器
- 侧边栏有模型选择器
- 主区域有输入框和预测按钮

### 3. 点击侧边栏 "Model Comparison"
- 页面标题：📊 16 Model Performance Comparison
- 侧边栏有筛选器
- 主区域显示16个模型对比

---

## ⚠️ 常见错误

### 错误1: "点击没反应"

**检查**:
- [ ] 侧边栏是否展开？
- [ ] 点击的是正确的页面名称吗？
- [ ] 浏览器控制台有错误吗？

**解决**:
1. 确保侧边栏展开（点击左上角菜单图标）
2. 尝试点击不同的页面名称
3. 查看终端/控制台的错误信息

### 错误2: "页面空白"

**检查**:
- [ ] 终端有错误信息吗？
- [ ] 页面文件语法正确吗？

**解决**:
1. 查看终端错误
2. 运行: `python -m py_compile pages/0_prediction.py`
3. 检查缺失的依赖

### 错误3: "看不到侧边栏"

**原因**: 浏览器窗口太窄

**解决**:
1. 放大浏览器窗口
2. 或点击左上角的菜单图标（☰）展开侧边栏

---

## 🚀 完整重启流程

```bash
# 1. 停止当前Streamlit
# 按 Ctrl+C

# 2. 清除缓存
python -c "
from pathlib import Path
import shutil

# 清除Streamlit缓存
streamlit_cache = Path.home() / '.streamlit'
if streamlit_cache.exists():
    shutil.rmtree(streamlit_cache)
    print('Streamlit缓存已清理')

# 清除__pycache__
pycache = Path('.') / '__pycache__'
if pycache.exists():
    shutil.rmtree(pycache)
    print('Python缓存已清理')
"

# 3. 重新启动
streamlit run app_bbb_predict.py

# 4. 打开浏览器
# 访问: http://localhost:8501

# 5. 查看侧边栏
# 点击不同的页面进行测试
```

---

## 📸 需要的信息

如果上述方案都无法解决问题，请提供：

1. **Streamlit启动信息**（终端输出）
2. **浏览器控制台错误**（F12 → Console）
3. **页面截图**
4. **点击哪个元素没反应**

---

## 🎯 快速测试

### 测试1: 主页是否显示
```bash
streamlit run app_bbb_predict.py
```
应该看到：
- 标题：🧪 BBB Permeability Prediction Platform
- 统计卡片：16个模型，4个数据集
- 功能卡片

### 测试2: 点击Prediction
点击后应该看到：
- 侧边栏：数据集选择器、模型选择器
- 主区域：SMILES输入框、预测按钮

### 测试3: 点击Model Comparison
点击后应该看到：
- 侧边栏：筛选器
- 主区域：16个模型对比表格

---

**如果问题仍然存在，请运行诊断脚本并发送输出结果！**

```bash
python scripts/diagnose_streamlit.py
```
