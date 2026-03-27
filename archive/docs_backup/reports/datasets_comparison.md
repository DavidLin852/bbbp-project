# BBB预测数据集对比

## 数据集概览

项目提供4个不同规模的数据集，可根据需求选择：

| 数据集 | Groups | 总数 | 训练集 | 验证集 | 测试集 | 正例率 |
|--------|--------|------|--------|--------|--------|--------|
| **A** | A | 1058 | 846 | 106 | 106 | 87.9% |
| **A+B** | A+B | 4679 | 3743 | 468 | 468 | 73.0% |
| **A+B+C** | A+B+C | 7756 | 6204 | 776 | 776 | 63.8% |
| **A+B+C+D** | A+B+C+D | 7807 | 6245 | 781 | 781 | 63.5% |

## 数据来源说明

### Group A (1058条)
- **类型**: 高质量注释数据
- **特点**: 实验验证，最可靠
- **正例率**: 87.9%

### Group B (3621条)
- **类型**: 文献数据
- **特点**: 从文献中提取
- **正例率**: 68.6%

### Group C (3077条)
- **类型**: 计算预测数据
- **特点**: 基于计算方法预测
- **正例率**: 49.9% (接近平衡)

### Group D (51条)
- **类型**: 其他来源
- **特点**: 数量少，质量参差不齐
- **正例率**: 7.8%

## 数据集详细说明

### 1. A组数据集
**路径**: `data/splits/seed_0_A/`

**数据量**: 1058条
- 训练集: 846条
- 验证集: 106条
- 测试集: 106条

**特点**:
- ✅ 数据质量最高
- ✅ 完全实验验证
- ⚠️ 数据量较小
- ⚠️ 可能过拟合

**推荐用途**:
- 快速验证算法
- 原型开发
- 计算资源有限时

---

### 2. A+B组数据集 (默认)
**路径**: `data/splits/seed_0_A_B/`

**数据量**: 4679条
- 训练集: 3743条
- 验证集: 468条
- 测试集: 468条

**特点**:
- ✅ 质量和数量的最佳平衡
- ✅ 包含高质量和文献数据
- ✅ 默认配置，广泛测试
- ✅ 适合大多数应用

**推荐用途**:
- 标准模型训练
- 生产环境部署
- 论文发表
- **推荐大多数场景使用**

---

### 3. A+B+C组数据集
**路径**: `data/splits/seed_0_A_B_C/`

**数据量**: 7756条
- 训练集: 6204条
- 验证集: 776条
- 测试集: 776条

**特点**:
- ✅ 数据量大，覆盖面广
- ✅ 包含计算预测数据
- ⚠️ C组数据可能含有噪声
- ⚠️ 需要仔细验证模型性能

**推荐用途**:
- 提升模型覆盖面
- 探索性研究
- 数据增强实验

**注意事项**:
- 需要评估C组数据的影响
- 可能需要调整超参数
- 建议对比A+B和A+B+C的性能

---

### 4. A+B+C+D组数据集
**路径**: `data/splits/seed_0_A_B_C_D/`

**数据量**: 7807条
- 训练集: 6245条
- 验证集: 781条
- 测试集: 781条

**特点**:
- ✅ 使用所有可用数据
- ✅ 最大化训练数据量
- ⚠️ 数据质量参差不齐
- ⚠️ D组数据很少且质量低

**推荐用途**:
- 最大化训练数据
- 终极模型性能探索

**注意事项**:
- D组数据可能影响性能
- 建议与A+B+C对比
- 需要更多训练时间

## 如何选择数据集

### 决策流程图

```
开始
  │
  ├─ 追求最高可靠性？
  │   └─ 是 → A组数据集
  │
  ├─ 标准应用场景？
  │   └─ 是 → A+B组数据集 (推荐)
  │
  ├─ 需要更大覆盖面？
  │   └─ 是 → A+B+C组数据集
  │
  └─ 使用所有数据？
      └─ 是 → A+B+C+D组数据集
```

### 推荐策略

| 场景 | 推荐数据集 | 理由 |
|------|-----------|------|
| **快速验证** | A | 训练快，质量高 |
| **生产部署** | A+B | 平衡可靠，广泛验证 |
| **性能优化** | A+B+C | 更多数据，更广覆盖 |
| **极限探索** | A+B+C+D | 最大化数据利用 |
| **默认选择** | A+B | ✅ **推荐大多数情况** |

## 使用方法

### 在代码中使用

```python
from pathlib import Path
import pandas as pd

# 选择数据集
dataset_name = 'A+B'  # 或 'A', 'A+B+C', 'A+B+C+D'
folder_name = f'seed_0_{dataset_name.replace("+", "_")}'

# 加载数据
train = pd.read_csv(f'data/splits/{folder_name}/train.csv')
val = pd.read_csv(f'data/splits/{folder_name}/val.csv')
test = pd.read_csv(f'data/splits/{folder_name}/test.csv')

print(f'训练集: {len(train)}')
print(f'验证集: {len(val)}')
print(f'测试集: {len(test)}')
```

### 查看数据集信息

```python
import json
from pathlib import Path

dataset_name = 'A+B'
folder_name = f'seed_0_{dataset_name.replace("+", "_")}'

with open(f'data/splits/{folder_name}/dataset_info.json', 'r') as f:
    info = json.load(f)

print(f"数据集: {info['name']}")
print(f"Groups: {info['groups']}")
print(f"总样本: {info['total_size']}")
print(f"正例率: {info['positive_rate']:.1%}")
```

### 切换数据集进行训练

如果要使用不同数据集训练模型：

1. **方法1: 修改特征提取脚本**
   ```bash
   # 需要修改02_featurize_all.py支持不同数据集路径
   python scripts/02_featurize_all.py --dataset A+B+C
   ```

2. **方法2: 手动复制数据**
   ```bash
   # 备份默认数据
   cp -r data/splits/seed_0 data/splits/seed_0_backup

   # 复制新数据到默认位置
   cp -r data/splits/seed_0_A_B_C/* data/splits/seed_0/

   # 然后正常训练
   python scripts/02_featurize_all.py --seed 0
   ```

3. **方法3: 修改配置**
   ```python
   # 修改src/config.py
   @dataclass(frozen=True)
   class Paths:
       # ...
       data_splits: Path = root / "data" / "splits" / "seed_0_A_B_C"
       # ...
   ```

## 性能对比建议

建议进行以下对比实验：

1. **基准**: 在A+B组训练，评估性能
2. **增加数据**: 在A+B+C组训练，对比性能提升
3. **完整数据**: 在A+B+C+D组训练，评估D组影响
4. **小数据集**: 在A组训练，评估数据量影响

关键指标：
- AUC (整体性能)
- Precision (假阳性率)
- Recall (假阴性率)
- 推理时间

## 数据集文件结构

```
data/splits/
├── seed_0_A/              # A组数据集
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── dataset_info.json
├── seed_0_A_B/            # A+B组数据集 (默认)
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── dataset_info.json
├── seed_0_A_B_C/          # A+B+C组数据集
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── dataset_info.json
└── seed_0_A_B_C_D/        # A+B+C+D组数据集
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── dataset_info.json
```

## 常见问题

### Q: 默认使用哪个数据集？
A: A+B组数据集 (seed_0_A_B)，这是质量和数量的最佳平衡。

### Q: 应该选择哪个数据集？
A:
- **大多数情况**: A+B组（默认）
- **追求可靠性**: A组
- **提升覆盖面**: A+B+C组
- **最大化数据**: A+B+C+D组

### Q: C组数据可靠吗？
A: C组是计算预测数据，不是实验验证。可能包含噪声，建议通过实验验证其效果。

### Q: 如何评估不同数据集的效果？
A: 在相同模型和超参数下训练，对比测试集性能指标。

### Q: 可以混合使用多个数据集吗？
A: 可以。例如在A+B组训练，在C组测试泛化能力。

## 更新日志

- **2024-01-26**: 创建4个标准化数据集（A, A+B, A+B+C, A+B+C+D）
