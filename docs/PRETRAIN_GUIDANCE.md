# Pretraining Strategy Guide

## Overview

This guide covers pretraining strategies for both **Transformer** (SMILES-based) and **Graph** (GNN-based) models on ZINC22, and how to fine-tune on B3DB for BBB permeability prediction.

---

## Quick Start

### Step 1: Verify ZINC22 data

```bash
# Fast estimation (~2 min) - no RDKit parsing
python -c "
from src.pretrain.data import estimate_total_molecules
estimate_total_molecules('data/zinc22', sample_files=20)
"
```

### Step 2: Run Graph Pretraining (Recommended First)

Graph pretraining is **higher priority** than Transformer for BBB prediction:
- B3DB data is molecular graphs
- GNN captures structural patterns directly
- Property prediction (logP, TPSA, MW, rotatable bonds) is a strong proxy task

```bash
# Smoke test (~10 min)
python scripts/pretrain/pretrain_graph.py \
    --num_samples 100000 --epochs 5 --batch_size 256

# Full pretraining (~4-8 hours on 1 GPU, 2M molecules, 20 epochs)
python scripts/pretrain/pretrain_graph.py \
    --num_samples 2000000 --epochs 20 --batch_size 256 \
    --save_dir artifacts/models/pretrain/gin_full
```

### Step 3: Run Transformer Pretraining (Optional)

Transformer pretraining is useful for **SMILES generation** or **token-level tasks**:
- SMILES autoencoder pretraining
- Molecule generation with pretrained embeddings
- Less directly applicable to BBB classification

```bash
# Quick pretraining (~2 hours, 5M molecules, 10 epochs)
python scripts/pretrain/pretrain_smiles.py \
    --data_dir data/zinc22 \
    --num_samples 5000000 \
    --epochs 10 \
    --batch_size 512 \
    --n_layers 4 --d_model 256 --n_heads 8 \
    --save_dir artifacts/models/pretrain/transformer_v1
```

---

## Detailed Configuration Guide

### Graph Pretraining (GIN/GAT on ZINC22)

#### Data Loading (Fixed)
- Reads ALL 1414 `.smi.gz` files evenly (uniform coverage)
- Fast validation (no RDKit per-line parsing)
- Parallel reading with ProcessPoolExecutor
- Results cached to `data/zinc22/cache/smiles_index_*.pkl`

#### Property Prediction Targets
| Property | Description | Normalization |
|----------|-------------|---------------|
| logP | Crippen LogP | Raw value |
| TPSA | Topological polar surface area | Raw value |
| MW | Molecular weight | Divided by 1000 |
| Rotatable | Number of rotatable bonds | Raw value |

#### Recommended Configurations

| Scenario | Samples | Epochs | Batch | GPU | Time |
|----------|---------|--------|-------|-----|------|
| Smoke test | 100K | 5 | 256 | 1 | ~10 min |
| Medium | 500K | 10 | 256 | 1 | ~1 hour |
| Full | 2M | 20 | 256 | 1 | ~4-6 hours |
| Full + 2 GPUs | 2M | 20 | 256×2 | 2 | ~2-3 hours |
| Full + 2 GPUs | 5M | 20 | 512×2 | 2 | ~4-6 hours |

#### CLI Options

```bash
python scripts/pretrain/pretrain_graph.py \
    --data_dir data/zinc22 \
    --num_samples 2000000        # Number of molecules (default: 100K)
    --model gin                  # gnn type: gin, gat, gcn (default: gin)
    --hidden_dim 128             # Hidden dimension (default: 128)
    --num_layers 3              # GNN layers (default: 3)
    --batch_size 256             # Batch size (default: 256)
    --epochs 20                  # Epochs (default: 10)
    --lr 1e-3                    # Learning rate (default: 1e-3)
    --gradient_accumulation 1    # Accumulation steps (default: 1)
    --save_dir artifacts/models/pretrain/gin_full
    --num_workers 4              # Workers (default: 4)
    --log_interval 100           # Log every N steps (default: 100)
```

#### Multi-GPU

```bash
# DataParallel (simple, good for single-node multi-GPU)
python -m torch.distributed.run --nproc_per_node=2 \
    scripts/pretrain/pretrain_graph.py \
    --num_samples 2000000 --epochs 20 --batch_size 256
```

#### Graph Cache

Graphs are precomputed ONCE and cached:
```
data/zinc22/cache/graph_cache_2000000.pkl
```

This means:
- **First run**: Slow (needs RDKit parsing for all graphs)
- **Subsequent runs**: Fast (loads from pickle cache)
- To rebuild: delete the cache file

---

### Transformer Pretraining (SMILES MLM on ZINC22)

#### Data Loading (Fixed)
- Reads ALL 1414 `.smi.gz` files evenly
- Fast validation (no RDKit)
- Shuffled after loading
- Subsamples to `num_samples` after shuffle
- Results cached to `data/zinc22/cache/smiles_index_*.pkl`

#### MLM Configuration
| Parameter | Default | Recommended |
|-----------|---------|-------------|
| d_model | 256 | 256-512 |
| n_heads | 8 | 8-12 |
| n_layers | 4 | 4-8 |
| mask_ratio | 0.15 | 0.15-0.25 |
| max_length | 128 | 64-128 |
| batch_size | 32 | 256-1024 |

#### Recommended Configurations

| Scenario | Samples | Epochs | Batch | GPU | AMP | Time |
|----------|---------|--------|-------|-----|-----|------|
| Smoke test | 100K | 5 | 512 | 1 | Yes | ~5 min |
| Medium | 1M | 10 | 512 | 1 | Yes | ~30 min |
| Full | 5M | 10 | 512 | 1 | Yes | ~2-3 hours |
| Full + 2 GPUs | 5M | 10 | 512×2 | 2 | Yes | ~1-1.5 hours |

#### CLI Options

```bash
python scripts/pretrain/pretrain_smiles.py \
    --data_dir data/zinc22 \
    --num_samples 5000000        # Molecules (default: 10K)
    --batch_size 512             # Batch size (default: 32)
    --epochs 10                  # Epochs (default: 10)
    --n_layers 4                 # Transformer layers (default: 4)
    --n_heads 8                  # Attention heads (default: 8)
    --d_model 256                # Model dimension (default: 256)
    --mask_ratio 0.15             # Mask ratio (default: 0.15)
    --lr 1e-4                    # Learning rate (default: 1e-4)
    --save_dir artifacts/models/pretrain/transformer_v1
```

---

## Fine-tuning on B3DB

### Graph Pretrained Model

```bash
python scripts/gnn/run_gnn_benchmark.py \
    --pretrained_encoder artifacts/models/pretrain/gin_full/gin_pretrained_backbone.pt \
    --tasks classification
```

### Transformer Pretrained Model

```bash
python scripts/transformer/run_transformer_benchmark.py \
    --pretrained_encoder artifacts/models/pretrain/transformer_v1/transformer_pretrained_encoder.pt \
    --tasks classification
```

---

## Speed Optimization Summary

| Optimization | Effect | Effort |
|-------------|--------|--------|
| AMP (mixed precision) | 1.5-2x faster | Already in code |
| Larger batch size (256+) | Fewer steps, better GPU util | Change CLI flag |
| Read all files evenly | Proper data coverage | Already fixed |
| Cache graphs (once) | Eliminates RDKit overhead | Already in code |
| Multi-GPU (Graph) | ~1.8x with 2 GPUs | Change CLI flag |
| Multi-GPU (Transformer) | ~1.5x with 2 GPUs | Change CLI flag |
| Reduce max_length to 64 | ~2x fewer tokens | Change in code |
| Fewer epochs (10 vs 100) | 10x fewer epochs | Change CLI flag |

---

## Common Issues

### Q: `test_data_loading.py` is very slow (hours)
**A:** It's using RDKit to validate every SMILES line-by-line (50M × ~1ms = 50+ hours). Kill it and use `estimate_total_molecules()` instead.

### Q: Transformer training is slow (~2s/step)
**A:** Small batch size (32). Increase to 512+. Also check if AMP is enabled.

### Q: Graph pretraining is slow during first run
**A:** Normal — RDKit needs to parse all molecules. But it's only done ONCE. Subsequent runs load from cache.

### Q: CUDA out of memory
**A:** Reduce batch size. For GIN with hidden_dim=128: batch_size=256 is safe on 8GB GPU.

### Q: Early stopping in pretraining
**A:** No early stopping implemented. The model will train all specified epochs.

### Q: How many unique molecules per epoch?
**A:** After the fixes: `num_samples` molecules are loaded and shuffled. Each epoch iterates over all `num_samples` molecules. With `num_samples=5M` and batch_size=512, that's ~10K steps/epoch.

### Q: Transformer vs Graph — which to prioritize?
**A:** **Graph first**. B3DB is graph classification. GNN pretraining directly transfers. Transformer is useful for generation tasks or if you want token-level representations.

---

## Cluster Execution

### Single Node

```bash
sbatch scripts/pretrain/run_pretrain_graph.sh
```

### Multi-Node (if available)

```bash
srun python scripts/pretrain/pretrain_graph.py \
    --num_samples 5000000 --epochs 20 --batch_size 512
```

### Transferring Results

```bash
# From cluster to local
rsync -avz user@cluster:/path/to/artifacts/models/pretrain/ ./artifacts/models/pretrain/

# Exclude large caches
rsync -avz --exclude='*.pkl' --exclude='cache/' \
    user@cluster:/path/to/artifacts/ ./artifacts/
```
