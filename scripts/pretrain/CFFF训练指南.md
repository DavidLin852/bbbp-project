# CFFF集群预训练 - 快速指南

## 📋 你现在拥有的文件

我已经为你创建了以下文件：

### 1. 训练脚本（.sh文件）
- `run_pretrain_cluster.sh` - 同时运行图和Transformer预训练（48小时）
- `run_graph_pretrain_cluster.sh` - 仅图预训练（24小时）✨ **推荐**
- `run_smiles_pretrain_cluster.sh` - 仅SMILES预训练（24小时）✨ **推荐**
- `run_pretrain_test_cluster.sh` - 快速测试（2小时）✨ **先运行这个**

### 2. 辅助文件
- `CLUSTER_TRAINING_README.md` - 详细使用说明
- `pretrain_config.py` - Python配置管理（可选）
- `check_pretrain_readiness.py` - 系统检查脚本

## 🚀 推荐流程

### 第1步：本地检查（5分钟）

```bash
# 在本地机器上
cd bbbp-project
python scripts/pretrain/check_pretrain_readiness.py
```

这会检查：
- ✓ Python依赖是否安装
- ✓ ZINC22数据是否正确
- ✓ 训练脚本是否存在
- ✓ GPU是否可用

### 第2步：修改路径（5分钟）

编辑这3个文件，修改路径：

```bash
# 1. run_graph_pretrain_cluster.sh
PROJECT_DIR="/path/to/your/bbbp-project"  # 改成你的集群路径

# 2. run_smiles_pretrain_cluster.sh
PROJECT_DIR="/path/to/your/bbbp-project"  # 改成你的集群路径

# 3. run_pretrain_test_cluster.sh
PROJECT_DIR="/path/to/your/bbbp-project"  # 改成你的集群路径
```

### 第3步：传输代码到集群（5分钟）

```bash
# 在本地机器上
cd bbbp-project

# 传输代码（不包括artifacts和cache）
rsync -avz \
    --exclude 'artifacts/' \
    --exclude 'data/zinc22/cache/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    ./ user@cluster:/path/to/cluster/bbbp-project/
```

### 第4步：在集群上运行测试（2小时）

```bash
# SSH到集群
ssh user@cluster

# 进入项目目录
cd /path/to/cluster/bbbp-project

# 提交测试任务
sbatch scripts/pretrain/run_pretrain_test_cluster.sh

# 查看状态
squeue -u $USER

# 查看日志
tail -f logs/pretrain_test_<jobid>.out
```

**测试任务做什么？**
- 图预训练：10K样本，5个epoch
- SMILES预训练：10K样本，5个epoch
- 预计时间：1-2小时

**为什么先测试？**
- ✓ 确保代码能正常运行
- ✓ 检查GPU内存是否够用
- ✓ 验证数据路径正确
- ✓ 测试集群环境配置

### 第5步：运行完整训练（24-48小时）

测试通过后，选择以下方案之一：

**方案A：分别训练（推荐，更灵活）**

```bash
# 先训练图模型（24小时）
sbatch scripts/pretrain/run_graph_pretrain_cluster.sh

# 等第一个完成，再训练Transformer（24小时）
sbatch scripts/pretrain/run_smiles_pretrain_cluster.sh
```

**方案B：同时训练（48小时）**

```bash
# 一次性提交两个任务
sbatch scripts/pretrain/run_pretrain_cluster.sh
```

## 📊 默认训练规模

| 参数 | 数值 | 说明 |
|------|------|------|
| 样本数 | 1,000,000 | 从ZINC22采样 |
| Epochs | 100 | 完整训练轮数 |
| Batch Size | 128 | GPU内存~16GB |
| 图模型 | GIN (256d, 5层) | 预测分子属性 |
| Transformer | (512d, 8头, 6层) | MLM预训练 |

## ⚙️ 根据资源调整

### GPU内存 < 16GB？

编辑脚本，修改这些参数：

```bash
# 减小batch size
BATCH_SIZE=64  # 或32

# 减小模型
GRAPH_HIDDEN_DIM=128  # 原来256
TRANSFORMER_D_MODEL=256  # 原来512
```

### 有更多GPU资源？

```bash
# 增大batch size（加速训练）
BATCH_SIZE=256

# 增大模型
GRAPH_HIDDEN_DIM=512
TRANSFORMER_D_MODEL=1024
```

### 时间有限？

```bash
# 减少epochs
EPOCHS=50  # 原来100

# 或减少样本
NUM_SAMPLES=500000  # 原来1M
```

## 📁 预期输出

训练完成后，你会得到：

```
artifacts/models/pretrain/
├── graph/
│   ├── gin_pretrained_backbone.pt       ← 图模型
│   └── training_history.json
└── transformer/
    ├── transformer_pretrained_encoder.pt ← Transformer
    ├── tokenizer.pkl
    └── training_history.json
```

## 🔍 监控训练

```bash
# 查看任务
squeue -u $USER

# 查看实时日志
tail -f logs/graph_pretrain_<jobid>.out

# 查看GPU使用
ssh <计算节点名>
nvidia-smi

# 查看任务详情
sacct -j <jobid>
```

## 🐛 常见问题

**Q: 任务被kill了？**
A: 可能是内存不足或超时，检查 `.err` 日志

**Q: CUDA out of memory？**
A: 减小 `BATCH_SIZE` 到 32 或 64

**Q: 找不到数据？**
A: 检查 `data/zinc22/` 下是否有 `H04/`, `H05/` 等目录

**Q: ImportError？**
A: 在集群上运行 `pip install -r requirements-research.txt`

## 📝 下一步

预训练完成后，模型可以直接用于微调：

```bash
# 图模型微调
python scripts/pretrain/finetune_graph.py \
    --pretrained_path artifacts/models/pretrain/graph/gin_pretrained_backbone.pt

# Transformer微调
python scripts/transformer/run_transformer_benchmark.py \
    --pretrained_encoder artifacts/models/pretrain/transformer/transformer_pretrained_encoder.pt
```

## 💡 关键点总结

1. **先运行测试** (`run_pretrain_test_cluster.sh`)
2. **修改路径** 在每个 `.sh` 文件中
3. **推荐分别训练** 图和Transformer分开跑
4. **监控日志** 使用 `tail -f logs/*.out`
5. **调整参数** 根据GPU内存和时间限制

---

**准备好了吗？开始吧！** 🎉

```bash
# 1. 检查系统
python scripts/pretrain/check_pretrain_readiness.py

# 2. 修改脚本中的路径
# 编辑 PROJECT_DIR=...

# 3. 传输到集群
rsync -avz --exclude='artifacts/' ./ user@cluster:/path/

# 4. 在集群上运行测试
sbatch scripts/pretrain/run_pretrain_test_cluster.sh

# 5. 测试通过后运行完整训练
sbatch scripts/pretrain/run_graph_pretrain_cluster.sh
```
