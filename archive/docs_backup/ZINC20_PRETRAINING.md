# ZINC20 大规模分子预训练

使用ZINC20数据库进行分子表示学习的大规模预训练。

## 背景

ZINC20是包含10亿+可商业化分子的数据库。通过在大量未标记分子数据上进行自监督预训练，模型可以学习到丰富的分子结构表示，然后在下游任务（如BBB渗透性预测）上微调。

## 预训练任务

### 1. Context Prediction (上下文预测)
- **目标**: 预测每个原子周围环境中的原子类型
- **方法**: 多标签分类，预测邻域中是否存在C, N, O, F, P, S, Cl, Br, I
- **优势**: 学习局部化学环境和官能团

### 2. Property Prediction (属性预测)
- **目标**: 预测分子的理化性质
- **属性**: logP, TPSA, MW, 旋转键数, HBD, HBA, 环数, sp3碳比例, 芳香比例
- **优势**: 学习与药物性质相关的全局特征

### 3. Masked Reconstruction (掩码重构)
- **目标**: 重构被掩码的节点特征
- **方法**: 类似BERT，随机掩码15%的节点，预测其原始特征
- **优势**: 学习鲁棒的原子表示

## 快速开始

### 1. 安装依赖

```bash
pip install requests tqdm pandas numpy torch torch-geometric rdkit
```

### 2. 下载数据

```bash
# 下载100万个分��
python pretrain_zinc20.py --step download --num-molecules 1000000

# 下载10万个分子（快速测试）
python pretrain_zinc20.py --step download --num-molecules 100000
```

输出位置: `data/zinc20/zinc20_{N}_seed{seed}.csv`

### 3. 预训练

```bash
# 使用所有预训练任务
python pretrain_zinc20.py --step pretrain --epochs 100 --batch-size 256

# 只使用属性预测（更快）
python pretrain_zinc20.py --step pretrain --no-context --use-property --epochs 50

# 只使用上下文预测
python pretrain_zinc20.py --step pretrain --use-context --no-property --epochs 50
```

### 4. 微调到BBB任务

```bash
# 使用最佳checkpoint
python pretrain_zinc20.py --step finetune --pretrain-ckpt artifacts/models/zinc20_pretrain/seed_42/best.pt

# 自动查找最新checkpoint
python pretrain_zinc20.py --step finetune
```

## 项目结构

```
src/pretrain/
├── zinc20_loader.py      # ZINC20数据加载和处理
├── zinc20_pretrain.py    # 预训练模型定义
└── ...

pretrain_zinc20.py        # 主训练脚本
```

## 关键类和函数

### 数据加载

```python
from src.pretrain.zinc20_loader import (
    ZINC20GraphDataset,
    ZINC20StreamingDataset,
    download_zinc20_tranches,
    compute_zinc_properties
)

# 下载数据
smiles_file = download_zinc20_tranches(
    output_dir="data/zinc20",
    num_molecules=1_000_000
)

# 创建数据集
dataset = ZINC20GraphDataset(
    root="artifacts/features/zinc20/train",
    smiles_file=smiles_file
)
```

### 预训练模型

```python
from src.pretrain.zinc20_pretrain import (
    ZINC20PretrainModel,
    PretrainConfig,
    load_pretrained_backbone
)

# 创建模型
cfg = PretrainConfig(
    in_dim=29,
    hidden=128,
    heads=4,
    num_layers=3,
    lambda_context=1.0,
    lambda_property=1.0
)
model = ZINC20PretrainModel(cfg)

# 加载预训练backbone用于微调
backbone = load_pretrained_backbone(
    checkpoint_path="path/to/checkpoint.pt",
    cfg=cfg,
    freeze=False
)
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-molecules` | 1000000 | 下载分子数量 |
| `--epochs` | 100 | 预训练轮数 |
| `--batch-size` | 256 | 批大小 |
| `--lr` | 0.002 | 学习率 |
| `--hidden` | 128 | 隐藏层维度 |
| `--heads` | 4 | GAT注意力头数 |
| `--num-layers` | 3 | GNN层数 |
| `--lambda-context` | 1.0 | Context loss权重 |
| `--lambda-property` | 1.0 | Property loss权重 |
| `--lambda-mask` | 0.5 | Mask loss权重 |

## 预期结果

### 预训练

- 100万分子, 100 epochs ~ 4-8小时 (单卡V100/A100)
- 预训练loss应下降到 < 0.5

### 微调后性能

| 模型 | AUC | Precision | Recall | F1 |
|------|-----|-----------|--------|-----|
| XGB (baseline) | 0.947 | 0.902 | 0.962 | 0.931 |
| GAT (from scratch) | ~0.92 | ~0.88 | ~0.94 | ~0.91 |
| GAT + ZINC20 | ~0.94+ | ~0.90+ | ~0.95+ | ~0.92+ |

*注: ZINC20预训练在BBB小数据集(7807样本)上的提升可能有限，但在更小的数据集上效果更明显*

## 进阶使用

### 自定义数据采样

```python
# 下载特定性质范围的分子
python -c "
from src.pretrain.zinc20_loader import download_zinc20_tranches

download_zinc20_tranches(
    output_dir='data/zinc20',
    num_molecules=500000,
    property_range={
        'logP': (0, 5),
        'MW': (200, 500)
    }
)
"
```

### 使用特定Tranche

```python
# 只下载特定分子量范围的tranches
download_zinc20_tranches(
    output_dir="data/zinc20",
    num_molecules=500000,
    tranches=["AA", "AB", "AC"]  # 小分子
)
```

### 冻结Backbone微调

```python
# 只训练分类头
backbone = load_pretrained_backbone(ckpt_path, cfg, freeze=True)
classifier = nn.Linear(cfg.hidden, 1)
```

## 故障排除

### 内存不足

- 减少批大小: `--batch-size 128`
- 使用流式数据集: 自动处理 (>500k分子)

### 下载失败

- 检查网络连接到zinc20.docking.org
- 尝试减少 `--num-molecules`

### CUDA OOM

```bash
# 使用CPU
python pretrain_zinc20.py --step pretrain --device cpu

# 减少模型大小
python pretrain_zinc20.py --step pretrain --hidden 64 --heads 2
```

## 参考文献

1. Sun, F. Y., et al. "InfoGraph: Unsupervised learning of graph representations." ICCV 2019.
2. Hu, W., et al. "Strategies for pre-training graph neural networks." ICLR 2020.
3. Liu, X., et al. "Masked Graph Modeling for Molecular Representation Learning." NeurIPS 2022.

## TODO

- [ ] 添加对比学习预训练 (GraphCL)
- [ ] 支持3D构象预训练
- [ ] 添加更多下游任务评估
- [ ] 分布式训练支持
