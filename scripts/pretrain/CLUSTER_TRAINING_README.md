# CFFF集群预训练指南

本指南说明如何在CFFF集群上运行B3P预训练任务。

## 📋 概述

项目包含两个预训练任务：

1. **图预训练** - GIN/GAT模型在ZINC22分子图上预训练
2. **SMILES Transformer预训练** - Transformer模型在SMILES序列上预训练（MLM）

## 🚀 快速开始

### 步骤1：准备代码和数据

```bash
# 在集群上
cd /path/to/your/workspace
git clone <your-repo-url> bbbp-project
cd bbbp-project

# 确保ZINC22数据在正确位置
# data/zinc22/
#   ├── H04/
#   ├── H05/
#   └── ...
```

### 步骤2：修改脚本中的路径

编辑每个 `.sh` 文件，修改 `PROJECT_DIR` 为你的实际路径：

```bash
PROJECT_DIR="/path/to/your/bbbp-project"  # 改成实际路径
```

### 步骤3：运行快速测试（推荐）

先运行小规模测试，确保代码正常工作：

```bash
# 提交测试任务（10K样本，5个epoch，约30分钟）
sbatch scripts/pretrain/run_pretrain_test_cluster.sh

# 查看日志
tail -f logs/pretrain_test_<jobid>.out
```

### 步骤4：运行完整预训练

测试通过后，运行完整训练：

**选项A：同时训练两个模型（48小时）**
```bash
sbatch scripts/pretrain/run_pretrain_cluster.sh
```

**选项B：分别训练（推荐，更灵活）**
```bash
# 图预训练（24小时）
sbatch scripts/pretrain/run_graph_pretrain_cluster.sh

# SMILES预训练（24小时）
sbatch scripts/pretrain/run_smiles_pretrain_cluster.sh
```

## 📊 训练配置说明

### 默认配置（完整训练）

| 参数 | 图预训练 | SMILES预训练 |
|------|---------|-------------|
| 样本数 | 1,000,000 | 1,000,000 |
| Epochs | 100 | 100 |
| Batch Size | 128 | 128 |
| 学习率 | 1e-3 | 1e-4 |
| GPU内存需求 | ~16GB | ~16GB |
| 训练时间 | ~24小时 | ~24小时 |

### 测试配置（快速验证）

| 参数 | 数值 |
|------|-----|
| 样本数 | 10,000 |
| Epochs | 5 |
| Batch Size | 32 |
| 训练时间 | ~30分钟 |

## ⚙️ 调整参数

根据你的集群资源调整参数：

### GPU内存不足？

```bash
# 减小batch size
BATCH_SIZE=64  # 或 32

# 减小模型大小
GRAPH_HIDDEN_DIM=128  # 原来256
TRANSFORMER_D_MODEL=256  # 原来512
```

### 有更多GPU？

```bash
# 增大batch size加速训练
BATCH_SIZE=256

# 增大模型容量
GRAPH_HIDDEN_DIM=512
TRANSFORMER_D_MODEL=1024
```

### 时间有限？

```bash
# 减少epochs
EPOCHS=50  # 原来100

# 或减少样本数
NUM_SAMPLES=500000  # 原来1M
```

## 📁 输出文件

训练完成后，模型保存在：

```
artifacts/models/pretrain/
├── graph/
│   ├── gin_pretrained_backbone.pt       # 预训练的GIN backbone
│   └── training_history.json
└── transformer/
    ├── transformer_pretrained_encoder.pt  # 预训练的Transformer encoder
    ├── tokenizer.pkl                      # 训练好的tokenizer
    └── training_history.json
```

## 🔍 监控训练

### 查看任务状态
```bash
# 查看所有任务
squeue -u $USER

# 查看特定任务
squeue -j <jobid>
```

### 查看日志
```bash
# 实时查看输出
tail -f logs/graph_pretrain_<jobid>.out

# 查看错误
tail -f logs/graph_pretrain_<jobid>.err
```

### 检查GPU使用
```bash
# 在计算节点上
nvidia-smi

# 持续监控
watch -n 1 nvidia-smi
```

## 📈 预期结果

### 图预训练
- 任务：预测分子属性（LogP, TPSA等）
- 损失函数：MSE
- 预期最终loss：< 0.1

### SMILES预训练
- 任务：Masked Language Modeling
- 损失函数：Cross Entropy
- 预期最终loss：< 1.0

## 🐛 常见问题

### 1. ImportError: No module named 'xxx'
```bash
# 确保安装了所有依赖
conda activate bbb
pip install -r requirements.txt
pip install -r requirements-research.txt
```

### 2. CUDA out of memory
```bash
# 减小batch size，在脚本中修改：
BATCH_SIZE=32  # 或更小
```

### 3. Data directory not found
```bash
# 检查ZINC22目录结构
ls data/zinc22/H04/
ls data/zinc22/H05/

# 确保路径正确
```

### 4. Job killed (timeout)
```bash
# 增加时间限制，在#SBATCH中修改：
#SBATCH --time=72:00:00  # 增加到72小时
```

## 📝 下一步

预训练完成后，可以进行微调：

```bash
# 图模型微调到B3DB
python scripts/pretrain/finetune_graph.py \
    --pretrained_path artifacts/models/pretrain/graph/gin_pretrained_backbone.pt \
    --task classification

# Transformer微调到B3DB
python scripts/transformer/run_transformer_benchmark.py \
    --pretrained_encoder artifacts/models/pretrain/transformer/transformer_pretrained_encoder.pt \
    --tasks classification
```

## 📧 获取帮助

遇到问题时：
1. 检查日志文件 `logs/*.err`
2. 查看SLURM状态 `sacct -j <jobid>`
3. 联系集群管理员

---

**祝训练顺利！** 🎉
