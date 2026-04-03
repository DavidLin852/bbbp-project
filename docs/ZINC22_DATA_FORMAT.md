# ZINC22 数据格式说明

## 当前数据格式

您的 ZINC22 数据已经放在 `data/zinc22/` 目录下，格式如下：

```
data/zinc22/
├── H04/
│   ├── H04M000.smi.gz
│   ├── H04M100.smi.gz
│   ├── H04P000.smi.gz
│   └── ...
├── H05/
│   └── ...
└── H06/
    └── ...
```

每个 `.smi.gz` 文件包含：
- 压缩的 SMILES 数据（gzip 格式）
- 每行格式：`SMILES\tZINC_ID`
- 例如：`BN(C)C\tZINC450000002gDx`

## ✅ 好消息：不需要解压脚本

数据管道已经更新，**直接读取 .smi.gz 文件**，无需手动解压。

## 快速测试

### 1. 验证数据加载

```bash
# 测试数据能否正确加载
python scripts/pretrain/test_data_loading.py
```

**预期输出：**
```
============================================================
Testing ZINC22 Data Loading
============================================================

Test 1: Counting molecules in data/zinc22/
✅ Total molecules: XXX

Test 2: Loading graph dataset (1K samples)
✅ Loaded 1000 samples
   Sample 0: XX nodes, XX edges

Test 3: Loading SMILES dataset (1K samples)
✅ Loaded 1000 samples
   Sample 0: SMILES...

Test 4: Loading larger dataset (10K samples)
✅ Loaded 10000 samples
```

### 2. 运行预训练（smoke test）

```bash
# 图预训练（GIN，10K 样本，5 epochs）
python scripts/pretrain/pretrain_graph.py \
    --data_dir data/zinc22 \
    --num_samples 10000 \
    --epochs 5 \
    --batch_size 32 \
    --model_type gin

# SMILES 预训练（Transformer，10K 样本，5 epochs）
python scripts/pretrain/pretrain_smiles.py \
    --data_dir data/zinc22 \
    --num_samples 10000 \
    --epochs 5 \
    --batch_size 32
```

## 数据管道特性

**自动处理：**
- ✅ 直接读取 .smi.gz 文件（无需解压）
- ✅ 自动遍历所有子目录（H04, H05, H06, ...）
- ✅ SMILES 验证（过滤无效分子）
- ✅ 智能缓存（首次读取后缓存索引）
- ✅ 增量采样（可指定任意样本数）

**使用示例：**
```python
from src.pretrain.data import ZINC22Dataset

# 自动读取所有 .smi.gz 文件
dataset = ZINC22Dataset(
    data_dir="data/zinc22",      # 目录路径
    representation="graph",       # 或 "smiles"
    num_samples=100000,           # 任意数量
)

# 第一次会读取所有文件并缓存
# 后续加载会使用缓存，速度更快
```

## CFFF 上使用

### 1. 传输数据

```bash
# 传输整个 data/zinc22/ 目录
rsync -avz data/zinc22/ user@cfff:/path/to/bbbp-project/data/zinc22/
```

### 2. 在 CFFF 上测试

```bash
# 激活环境
conda activate bbb

# 测试数据加载
python scripts/pretrain/test_data_loading.py

# 运行预训练
python scripts/pretrain/pretrain_graph.py \
    --data_dir data/zinc22 \
    --num_samples 10000 \
    --epochs 5 \
    --device cuda
```

## 如果添加更多数据

**未来数据格式相同**（H07, H08, ... 子目录）：

只需将新目录放入 `data/zinc22/`：
```bash
# 新数据自动会被发现
data/zinc22/
├── H04/
├── H05/
├── H06/
└── H07/    # 新添加的目录
    └── *.smi.gz
```

**下次运行时自动包含所有新数据。**

## 故障排除

### 问题：找不到文件

```bash
# 检查目录结构
ls -R data/zinc22/ | head -20

# 应该看到：
# data/zinc22/H04/H04M000.smi.gz
# data/zinc22/H04/H04M100.smi.gz
# ...
```

### 问题：读取失败

```bash
# 测试单个文件
zcat data/zinc22/H04/H04M000.smi.gz | head -5

# 应该看到 SMILES 字符串
```

### 问题：缓存问题

```bash
# 清除缓存（可选）
rm -rf data/zinc22/cache/

# 下次运行会重新建立索引
```

## 总结

✅ **数据格式已支持**：无需修改数据或解压
✅ **自动读取**：直接读取 .smi.gz 文件
✅ **增量采样**：可指定任意样本数
✅ **智能缓存**：首次读取后加速
✅ **未来兼容**：相同格式的数据直接可用

**可以开始预训练了！**
