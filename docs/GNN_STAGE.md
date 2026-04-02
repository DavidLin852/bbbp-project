# GNN Stage — Supervised Graph Neural Network Baselines

## Overview

This stage adds supervised Graph Neural Network (GNN) baselines for BBB permeability prediction, operating directly on molecular graph structures rather than hand-crafted fingerprints or descriptors.

**Stage:** 2 (after classical ML baselines)
**Execution:** GPU (CUDA) on CFFF cluster — CPU fallback supported but not primary target
**Goal:** Compare graph-learned representations against classical fingerprint-based baselines using the same scaffold split protocol, task definitions, and evaluation metrics.

---

## Scope

### In Scope

- **Models:** GCN, GIN, GAT
- **Tasks:** Classification (BBB+/BBB-), Regression (logBB)
- **Split:** Scaffold split (seeds 0–4, same splits as classical baselines)
- **Input:** Molecular graphs from RDKit (atom + bond features)
- **Comparison:** Fair side-by-side comparison with classical baselines

### Out of Scope (do NOT implement in this stage)

- Transformer models (Graphormer, GPS, MolBERT)
- Pretrained / foundation model encoders
- ZINC22 pretraining
- Feature concatenation (GNN + fingerprints)
- Multi-task learning
- Knowledge distillation

---

## Graph Featurization

### Node Features (22 dimensions per atom)

| Feature | Type | Dimension |
|---------|------|----------|
| Atomic number | One-hot (H,C,N,O,F,P,S,Cl,Br,I + unknown) | 11 |
| Degree (bond count) | Scalar | 1 |
| Hydrogen count | Scalar | 1 |
| Formal charge | Scalar | 1 |
| Hybridization | One-hot (SP/SP2/SP3/SP3D/SP3D2 + unknown) | 6 |
| Aromaticity | Binary | 1 |
| Scaled mass | Scalar (mass / 100) | 1 |
| **Total** | | **22** |

### Edge Features (8 dimensions per bond)

| Feature | Type | Dimension |
|---------|------|----------|
| Bond type | One-hot (SINGLE/DOUBLE/TRIPLE/AROMATIC + unknown) | 5 |
| Conjugation | Binary | 1 |
| Ring membership | Binary | 1 |
| Stereo | Binary | 1 |
| **Total** | | **8** |

### Design Notes

- GCN and GIN backbones do **not** use edge features (they aggregate node messages only). This is standard practice for these architectures.
- GAT also does not use edge features by default for consistency.
- To use edge features in GAT, the architecture would need modification (future work).
- All molecules are processed from canonical SMILES (`SMILES_canon` column in splits).

---

## Model Architecture

### Backbones

| Model | Architecture | Key Property |
|-------|-------------|-------------|
| **GCN** | GCNConv layers + BatchNorm + ELU | Spectral-based, message passing |
| **GIN** | GINConv (MLP + sum agg) + BatchNorm + ELU | More expressive than GCN (WL test) |
| **GAT** | GATConv (multi-head attention) + BatchNorm + ELU | Attention-weighted neighbors |

### Shared Architecture

```
Message Passing Layers (3 × [Conv → BatchNorm → ELU → Dropout])
           ↓
Global Mean Pooling
           ↓
MLP Head: Linear(128) → ReLU → Dropout → Linear(1)
           ↓
Classification: Sigmoid output (BCE loss)
Regression: Linear output (MSE loss)
```

### Hyperparameters

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| Hidden dimension | 128 | Standard; sufficient for B3DB size |
| Number of layers | 3 | Balances expressiveness vs. overfitting |
| GAT attention heads | 4 | Standard practice; concat mode |
| Dropout | 0.3 | Moderate regularization for 700-sample training |
| Learning rate | 1e-3 | Standard Adam default |
| Weight decay | 1e-4 | Light regularization |
| Max epochs | 300 | With early stopping (patience=30) |
| Batch size | 64 | Default; reduce if GPU OOM |

---

## CUDA / GPU Design

### Device Management

- **Auto-detection:** `get_device()` checks `torch.cuda.is_available()` and defaults to CUDA
- **GPU selection:** `--gpu N` sets `CUDA_VISIBLE_DEVICES=N` for multi-GPU nodes
- **CPU fallback:** Fully supported; all code runs on CPU when CUDA unavailable
- **Memory cleanup:** `clear_gpu_memory()` called between experiments to free GPU memory

### Data Loading for GPU

- `pin_memory=True` when on CUDA — accelerates host-to-GPU transfer
- `num_workers=0` by default — safe on all platforms, including CFFF cluster nodes
- Increase `num_workers` on CFFF if the cluster scheduler supports multiprocess data loading

### Model Checkpointing

- Checkpoints saved as CPU tensors (`k: v.cpu()`) — portable across GPU/CPU
- Best model selected by validation metric (AUC or R²)
- Early stopping prevents overfitting and unnecessary GPU time

### Memory Considerations

| Batch Size | Expected GPU Memory | Notes |
|-----------|-------------------|-------|
| 64 | ~200-400 MB | Default; safe for most GPUs |
| 32 | ~100-200 MB | Reduce if OOM on smaller GPUs |
| 16 | ~50-100 MB | Minimum viable batch size |

If you encounter CUDA OOM errors, reduce batch size:
```bash
python scripts/gnn/run_gnn_benchmark.py --batch_size 32
```

---

## File Structure

```
src/gnn/
├── __init__.py              # Module exports
├── models.py                # GCN, GIN, GAT backbones + heads + GNNConfig
├── dataset.py               # B3DBGNNDataset, graph featurization, result containers
└── train.py                 # Training loop, evaluation, CUDA utilities, multi-seed runner

scripts/gnn/
└── run_gnn_benchmark.py    # Main experiment runner

artifacts/models/gnn/
└── seed_{N}/
    ├── classification/
    │   ├── gcn/model.pt + result.json
    │   ├── gin/model.pt + result.json
    │   └── gat/model.pt + result.json
    └── regression/
        └── (same structure)

artifacts/reports/gnn/
└── gnn_benchmark_scaffold_{timestamp}.csv
```

---

## CFFF Cluster Setup

### 1. Environment Setup

On a CFFF GPU node, install dependencies in this order:

```bash
# Activate your conda/pip environment
conda activate bbb-research
# or: source activate bbb-research

# Step 1: PyTorch with CUDA 11.8 (most common on CFFF)
# Check your CUDA version with: nvidia-smi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Step 2: Verify CUDA is working
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Step 3: PyTorch Geometric (core only, quick install)
pip install torch-geometric

# Step 4 (optional but recommended): PyG compiled extensions
# These optimize sparse operations for GNN message passing
# Get your PyTorch version first:
VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA=cu118
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${VERSION}+${CUDA}.html

# Step 5: Verify everything works
python -c "
import torch
from torch_geometric.nn import GCNConv, GINConv, GATConv
print('PyG imports OK')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### 2. Verify CUDA Works

```bash
# Basic CUDA test
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    # Quick matmul test
    x = torch.randn(1000, 1000).cuda()
    y = x @ x
    print('GPU matmul OK')
"

# PyG CUDA test
python -c "
import torch
from torch_geometric.nn import GCNConv
conv = GCNConv(22, 128).cuda()
x = torch.randn(32, 10, 22).cuda()  # 32 graphs, 10 nodes avg
edge_index = torch.randint(0, 10, (2, 50)).cuda()
out = conv(x, edge_index)
print(f'PyG CUDA OK: output shape={out.shape}')
"
```

### 3. Submit a Batch Job

Create a job script `run_gnn_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=gnn_bbb
#SBATCH --partition=gpu          # adjust to your cluster's partition names
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --time=48:00:00          # 48 hours; adjust as needed
#SBATCH --output=gnn_%j.out
#SBATCH --error=gnn_%j.err

# Load CUDA module if needed
# module load cuda/11.8

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bbb-research

# Check GPU
nvidia-smi

# Run GNN benchmarks
cd $HOME/bbbp_project/code

# Classification first (recommended)
python scripts/gnn/run_gnn_benchmark.py \
    --tasks classification \
    --seeds 0,1,2,3,4 \
    --models gcn,gin,gat \
    --device cuda

# Then regression
python scripts/gnn/run_gnn_benchmark.py \
    --tasks regression \
    --seeds 0,1,2,3,4 \
    --models gcn,gin,gat \
    --device cuda
```

Submit with:
```bash
sbatch run_gnn_job.sh
```

### 4. Interactive GPU Session (for debugging)

```bash
# Request interactive GPU node
salloc --partition=gpu --gres=gpu:1 --time=04:00:00

# Once on the node:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bbb-research

# Quick test
python scripts/gnn/run_gnn_benchmark.py --dry_run

# Run one seed, one model (fastest test)
python scripts/gnn/run_gnn_benchmark.py \
    --tasks classification \
    --seeds 0 \
    --models gcn \
    --device cuda
```

---

## Running the Experiments

### Quick Start

```bash
# 1. Dry run — verify commands
python scripts/gnn/run_gnn_benchmark.py --dry_run

# 2. Classification only (recommended first run on GPU)
python scripts/gnn/run_gnn_benchmark.py --tasks classification

# 3. Regression only
python scripts/gnn/run_gnn_benchmark.py --tasks regression

# 4. Both tasks
python scripts/gnn/run_gnn_benchmark.py

# 5. Skip existing (resume interrupted runs)
python scripts/gnn/run_gnn_benchmark.py --skip_existing
```

### Custom Configuration

```bash
# Fewer seeds (faster iteration)
python scripts/gnn/run_gnn_benchmark.py --seeds 0,1

# Single model
python scripts/gnn/run_gnn_benchmark.py --models gcn

# Use specific GPU (multi-GPU nodes)
python scripts/gnn/run_gnn_benchmark.py --gpu 0

# Reduce batch size if GPU OOM
python scripts/gnn/run_gnn_benchmark.py --batch_size 32

# More regularization
python scripts/gnn/run_gnn_benchmark.py --dropout 0.5 --epochs 500

# CPU only (fallback)
python scripts/gnn/run_gnn_benchmark.py --device cpu
```

### Expected Runtime

| Configuration | GPU | CPU | Notes |
|--------------|-----|-----|-------|
| 1 seed, 1 model, 1 task | ~2-5 min | ~10-20 min | Fastest test |
| 5 seeds, 3 models, 1 task | ~30-60 min | ~2-4 hours | Full classification sweep |
| 5 seeds, 3 models, 2 tasks | ~60-120 min | ~4-8 hours | Full GNN benchmark |

Early stopping typically converges within 50-150 epochs. Runtime varies by GPU model and batch size.

---

## Comparison with Classical Baselines

Results should be compared using the **same scaffold split seeds** (0–4).

### Expected Comparison Table Format

| Model Type | Representation | Model | Test AUC (mean ± std) |
|------------|---------------|-------|----------------------|
| Classical | MACCS (167d) | lgbm | 0.9535 |
| Classical | Morgan (2048d) | rf | 0.9504 |
| **GNN** | **Graph (22d)** | **GCN** | **?** |
| **GNN** | **Graph (22d)** | **GIN** | **?** |
| **GNN** | **Graph (22d)** | **GAT** | **?** |

Key comparison points:
1. Do GNNs outperform classical baselines on scaffold split?
2. Is the improvement significant given the dataset size (700 training samples)?
3. Which GNN architecture performs best on this task?

**Caveats:**
- GNNs have more learnable parameters than classical models, which may lead to overfitting on small datasets
- Graph featurization (22 dims) is much lower-dimensional than fingerprints (167–2048 dims), which may limit expressiveness
- Results should be interpreted conservatively; small AUC differences (e.g., 0.01) may not be meaningful

---

## Common Issues

| Issue | Solution |
|-------|---------|
| `CUDA out of memory` | Reduce `--batch_size` to 32 or 16 |
| `torch.cuda.is_available() == False` | Check GPU driver: `nvidia-smi`; check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"` |
| `ModuleNotFoundError: torch_geometric` | Run `pip install torch-geometric` |
| Slow training on CPU | Use GPU; or reduce `--batch_size 128` for CPU efficiency |
| Multi-GPU not used | Currently single-GPU only; use `--gpu 0` to select a specific GPU |
| `torch-scatter` not found | Optional; skip if not needed for basic GCN/GIN/GAT. Install with: `pip install torch-scatter torch-sparse` |

---

## Next Stages

### Stage 3: Transformer / Graph Transformer
- Graphormer, GPS, or similar architectures
- Longer-range modeling than message-passing GNNs
- Requires careful hyperparameter tuning on small datasets

### Stage 4: ZINC22 Pretraining
- Pretrain GNN/Transformer on large molecular database
- Transfer learning to B3DB downstream tasks
- Highest potential impact but also highest risk (compute, setup complexity)

---

*Last Updated: 2026-04-02*
*Stage: Supervised GNN Baselines (GPU-first)*
