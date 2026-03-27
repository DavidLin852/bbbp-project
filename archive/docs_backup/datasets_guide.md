# B3DB数据集说明

## 数据来源

B3DB (Blood-Brain Barrier Database) 是一个血脑屏障渗透性数据库，包含7807个小分子化合物及其BBB渗透性标签。

## 数据分组

原始数据按数据质量和来源分为4个组：

| Group | 样本数 | BBB+ | BBB- | 说明 |
|-------|--------|------|------|------|
| **A** | 1058 | 930 | 128 | 高质量注释数据 |
| **B** | 3621 | 2486 | 1135 | 文献数据 |
| **C** | 3077 | 1536 | 1541 | 计算预测数据 |
| **D** | 51 | 4 | 47 | 其他来源 |
| **总计** | **7807** | **4956** | **2851** | - |

## 项目中的数据集

### 1. 默认数据集 (seed_0) - 仅使用 A+B 组

**路径**: `data/splits/seed_0/`

**数据量**: 4679条
- 训练集: 3743条 (A: 847, B: 2896)
- 验证集: 468条 (A: 110, B: 358)
- 测试集: 468条 (A: 101, B: 367)

**特点**:
- 只使用高质量数据（A组）和文献数据（B组）
- 避免使用计算预测数据（C组）和质量较低的数据（D组）
- 正例率: 69.5%

**配置**:
```python
DatasetConfig.group_keep = ("A", "B")
```

### 2. 完整数据集 (seed_0_all_groups) - 使用 A+B+C+D 组

**路径**: `data/splits/seed_0_all_groups/`

**数据量**: 7807条
- 训练集: 6245条
- 验证集: 781条
- 测试集: 781条

**特点**:
- 使用所有可用数据
- 包含计算预测数据（C组）
- 数据量是默认的1.67倍
- 正例率: 63.5%

**创建方法**:
```bash
python scripts/create_full_dataset.py --groups A,B,C,D --seed 0 --output_name all_groups
```

### 3. C+D组数据集 (seed_0_groups_cd) - 仅使用 C+D 组

**路径**: `data/splits/seed_0_groups_cd/`

**数据量**: 3128条
- 训练集: 2502条
- 验证集: 313条
- 测试集: 313条

**特点**:
- 只包含之前未使用的数据
- Group C: 计算预测数据（3077条）
- Group D: 其他来源数据（51条）
- 正例率: 49.2%（接近平衡）

**创建方法**:
```bash
python scripts/create_full_dataset.py --create_cd_only --seed 0
```

**用途**:
- 测试模型在计算预测数据上的表现
- 作为额外的训练数据提升模型泛化能力
- 主动学习的候选数据池

## 数据集对比

| 数据集 | 样本数 | 正例率 | 数据质量 | 推荐用途 |
|--------|--------|--------|----------|----------|
| **seed_0 (A+B)** | 4679 | 69.5% | 高 | 默认训练，注重可靠性 |
| **seed_0_all_groups** | 7807 | 63.5% | 混合 | 最大化数据量，提升覆盖面 |
| **seed_0_groups_cd** | 3128 | 49.2% | 中低 | 测试泛化能力，数据增强 |

## 如何使用不同数据集

### 方法1: 使用脚本创建

```bash
# 创建包含所有组的数据集
python scripts/create_full_dataset.py --groups A,B,C,D --seed 0

# 创建只包含C+D组的数据集
python scripts/create_full_dataset.py --create_cd_only --seed 0

# 自定义组合（例如只用B+C）
python scripts/create_full_dataset.py --groups B,C --seed 0 --output_name groups_bc
```

### 方法2: 在代码中使用

```python
from pathlib import Path
import pandas as pd

# 使用默认数据集 (A+B)
train_ab = pd.read_csv('data/splits/seed_0/train.csv')
print(f"A+B组训练集: {len(train_ab)} 条")

# 使用完整数据集 (A+B+C+D)
train_all = pd.read_csv('data/splits/seed_0_all_groups/train.csv')
print(f"全组训练集: {len(train_all)} 条")

# 使用C+D组数据集
train_cd = pd.read_csv('data/splits/seed_0_groups_cd/train.csv')
print(f"C+D组训练集: {len(train_cd)} 条")
```

### 方法3: 修改配置

如果要永久更改默认数据集，修改 `src/config.py`:

```python
@dataclass(frozen=True)
class DatasetConfig:
    # ...
    group_keep: tuple[str, ...] = ("A", "B", "C", "D")  # 改为包含所有组
    # ...
```

然后重新运行数据划分脚本：
```bash
python scripts/01_prepare_splits.py --seed 0
```

## 训练建议

### 使用默认数据集 (A+B)

**优点**:
- 数据质量高，可靠性好
- 模型性能更有保障
- 适合发表和实际应用

**缺点**:
- 数据量相对较小
- 可能未覆盖某些化学空间

**推荐场景**:
- 追求高可靠性
- 计算资源有限
- 快速原型开发

### 使用完整数据集 (A+B+C+D)

**优点**:
- 数据量最大（7807条）
- 覆盖更广的化学空间
- 可能提升模型泛化能力

**缺点**:
- 包含计算预测数据（C组），可能引入噪声
- 训练时间更长
- 需要更多计算资源

**推荐场景**:
- 追求最大覆盖面
- 有充足计算资源
- 探索性研究

### 使用C+D组数据集（作为补充）

**用途**:
1. **数据增强**: 将C+D组数据添加到A+B组训练集中
2. **主动学习**: 从C+D组中选择不确定样本进行标注
3. **迁移学习**: 先在A+B组训练，再在C+D组微调
4. **领域适应**: 学习处理不同质量的数据

## 特征提取和训练

使用不同数据集时，需要重新进行特征提取和训练：

```bash
# 1. 创建数据集（如果还没有）
python scripts/create_full_dataset.py --groups A,B,C,D --seed 0

# 2. 特征提取（需要修改脚本支持数据集选择）
# 目前02_featurize_all.py使用固定的seed_0路径
# 需要手动将数据复制到正确位置或修改脚本

# 3. 训练模型
python scripts/03_run_baselines.py --seed 0
```

## 数据集统计

```python
import pandas as pd
from pathlib import Path

datasets = {
    'A+B组 (默认)': 'data/splits/seed_0',
    'A+B+C+D组 (完整)': 'data/splits/seed_0_all_groups',
    'C+D组': 'data/splits/seed_0_groups_cd'
}

for name, path in datasets.items():
    print(f"\n{name}:")
    train = pd.read_csv(Path(path) / 'train.csv')
    val = pd.read_csv(Path(path) / 'val.csv')
    test = pd.read_csv(Path(path) / 'test.csv')

    total = len(train) + len(val) + len(test)
    bbb_plus = (train['y_cls'] == 1).sum() + (val['y_cls'] == 1).sum() + (test['y_cls'] == 1).sum()

    print(f"  总样本数: {total}")
    print(f"  训练/验证/测试: {len(train)}/{len(val)}/{len(test)}")
    print(f"  正例数: {bbb_plus}")
    print(f"  正例率: {bbb_plus/total:.2%}")
```

## 常见问题

### Q: 为什么默认只使用A+B组？
A: A组是高质量注释数据，B组是文献数据，可靠性较高。C组是计算预测数据，D组样本太少且质量较低，可能影响模型性能。

### Q: 使用C组数据会提升性能吗？
A: 不一定。C组数据量很大（3077条），但都是计算预测的，可能包含噪声。建议通过实验对比。

### Q: 如何选择合适的数据集？
A:
- **注重可靠性**: 使用A+B组（默认）
- **追求覆盖面**: 使用A+B+C+D组
- **数据增强**: 用C+D组作为补充

### Q: 能混合使用多个数据集吗？
A: 可以。例如：
- 在A+B组上训练
- 在C+D组上测试泛化能力
- 将C+D组作为主动学习的候选池

## 更新日志

- **2024-01-26**: 创建完整数据集和C+D组数据集
- **初始版本**: 默认使用A+B组（4679条）
