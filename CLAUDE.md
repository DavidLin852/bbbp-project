# CLAUDE.md - BBB Permeability Prediction Project

Machine learning pipeline for predicting blood-brain barrier (BBB) permeability using the B3DB classification dataset.

---

## Current Mainline Pipeline

The **working baseline pipeline** is a 3-step modular process:

```
scripts/baseline/
├── 01_preprocess_b3db.py    # Load B3DB, scaffold split (80:10:10)
├── 02_compute_features.py   # Compute fingerprints/descriptors
└── 03_train_baselines.py    # Train RF/XGB/LGBM models
```

**Best performance:** MACCS + LightGBM (Classification AUC: 0.9535 ± 0.0388), GIN (Regression R²: 0.7062 ± 0.0473)

### Quick Start

```bash
# Step 1: Preprocess with scaffold split
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"

# Step 2: Compute features
python scripts/baseline/02_compute_features.py --seed 0 --feature morgan

# Step 3: Train models
python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models rf,xgb,lgbm
```

### Full Benchmark

```bash
python scripts/analysis/run_baseline_matrix.py      # Full classical benchmark
python scripts/analysis/aggregate_results.py        # Aggregate results
python scripts/analysis/generate_benchmark_summary.py  # Summary report
```

### GNN Baselines (Stage 2)

```bash
# Install research dependencies first
pip install -r requirements-research.txt

# Run GNN benchmarks
python scripts/gnn/run_gnn_benchmark.py --dry_run
python scripts/gnn/run_gnn_benchmark.py --tasks classification
```

See `docs/GNN_STAGE.md` for full documentation.

### Fine-tuning (Stage 4)

```bash
# List all fine-tuning experiments
python scripts/finetune/run_finetune_matrix.py --list

# Phase 1 (Groups A+B+D, 68 experiments):
#   Group A: GNN fine-tuning (pretrained GNN → end-to-end)
#   Group B: Embedding → LightGBM (pretrained GNN/Transformer → frozen emb → LGBM)
#   Group D: Transformer fine-tuning (pretrained encoder → end-to-end)
python scripts/finetune/run_finetune_matrix.py --phase 1

# Extract embeddings first (required for Groups B and C)
python scripts/finetune/run_finetune_matrix.py --extract_only

# Phase 2 (Group C, 136 experiments):
#   Group C: Embedding + Feature → LightGBM (embedding + fingerprints → LGBM)
python scripts/finetune/run_finetune_matrix.py --phase 2

# Aggregate results
python scripts/finetune/run_finetune_matrix.py --aggregate_only
```

See `docs/FINETUNE_PLAN.md` for full experiment design.

---

## Protected Areas

**Do NOT modify without explicit instruction:**

| Area | Path | Reason |
|------|------|--------|
| Baseline scripts | `scripts/baseline/*.py` | Stable, tested pipeline |
| Configuration | `src/config.py`, `src/config/` | Frozen dataclasses, immutable |
| Core data modules | `src/data/` | Used by all experiments |
| Core feature modules | `src/features/fingerprints.py`, `src/features/descriptors.py` | Baseline depends on these |
| Core models | `src/models/baseline_models.py` | Stable model wrappers |
| Split files | `data/splits/` | Reproducibility requires fixed splits |

**Modification rules:**
- Bug fixes: allowed with minimal scope
- New features: add new files, don't modify existing ones
- Refactoring: requires explicit approval

---

## Preserved Future Research Modules

These modules are preserved for future research but **not part of the current mainline**:

### In `src/` (Active Research)

| Module | Path | Status | Purpose |
|--------|------|--------|---------|
| VAE | `src/vae/` | Architecture ready | Molecule generation |
| GAN | `src/gan/` | Architecture ready | Molecule generation |
| Transformer | `src/transformer/` | Benchmarked | SMILES Transformer baseline |
| Pretrain | `src/pretrain/` | Pretraining complete (14/17) | ZINC22 pre-training |
| Finetune | `src/finetune/`, `scripts/finetune/` | Ready to run | Fine-tune pretrained models on BBB task |
| Path Prediction | `src/path_prediction/` | Experimental | Transport mechanism |
| Explainability | `src/explain/` | Future | Model interpretation |

### In `archive/` (Legacy/Reference)

| Directory | Contents |
|-----------|----------|
| `old_scripts/` | Original numbered scripts (01-12*.py) |
| `old_src/` | Previous implementations (vae, gan, generation) |
| `old_web/` | Streamlit web interface (archived) |

**Rule:** Research modules can be modified freely. Do not integrate into mainline without explicit instruction.

---

## Project Structure

```
code/
├── src/                       # Source modules
│   ├── config.py              # Central configuration (frozen dataclasses)
│   ├── config/                # Configuration modules
│   ├── data/                  # Data loading, preprocessing, splitting
│   ├── features/              # Feature extraction (fingerprints, descriptors)
│   ├── models/                # Model implementations
│   ├── train/                 # Training utilities
│   ├── evaluate/              # Evaluation and reporting
│   ├── utils/                 # General utilities
│   └── [research modules]/    # vae, gan, transformer, pretrain, etc.
│
├── scripts/                   # Executable scripts
│   ├── baseline/              # ✅ PROTECTED: Working 3-step pipeline
│   └── analysis/              # Experiment matrix, aggregation
│
├── data/                      # Data files
│   ├── raw/                   # Original B3DB datasets
│   ├── splits/                # Generated train/val/test splits
│   └── transport_mechanisms/  # Transport mechanism datasets
│
├── artifacts/                 # Generated outputs
│   ├── features/              # Computed features (.npz)
│   ├── models/                # Trained models (.joblib)
│   └── reports/               # Benchmark summaries
│
├── archive/                   # Legacy and preserved code
│   ├── old_scripts/
│   ├── old_src/
│   └── old_web/
│
├── configs/                   # Environment configurations
├── docs/                      # Documentation
└── assets/                    # Additional resources
```

---

## Collaboration Rules

### Code Changes

1. **Read this file first** before making structural changes
2. **Small scoped changes** over broad refactors
3. **Add new files** rather than modifying protected ones
4. **Keep documentation aligned** with code changes
5. **Do not mix code edits** with generated outputs in Git commits

### Git Workflow

```bash
# Check status before committing
git status

# Stage only source files, not generated outputs
git add src/ scripts/ *.md

# Commit with clear message
git commit -m "feat: add new feature X"

# Generated outputs are gitignored:
# - artifacts/features/
# - artifacts/models/
# - data/splits/
```

### Research vs Mainline

- **Mainline (protected):** Baseline pipeline for reproducible benchmarks
- **Research (flexible):** VAE, GAN, Transformer, pretrain modules
- **Integration:** Requires explicit instruction and validation

---

## Local Development + Git + CFFF Workflow

### Environment Setup

```bash
# Create conda environment
conda env create -f configs/environment.yml
conda activate bbb

# Or with pip
pip install -r requirements.txt
```

### Development Workflow

```bash
# 1. Pull latest changes
git pull

# 2. Make changes (respecting protected areas)

# 3. Test baseline pipeline still works
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"
python scripts/baseline/02_compute_features.py --seed 0 --feature morgan
python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models rf

# 4. Commit changes
git add <files>
git commit -m "description"
git push
```

### CFFF (Cluster) Execution

```bash
# 1. Transfer code to cluster
rsync -avz --exclude 'artifacts/' --exclude 'data/splits/' ./ user@cluster:/path/to/project/

# 2. Run baseline pipeline
sbatch run_baseline.sh  # or equivalent cluster command

# 3. Retrieve results
rsync -avz user@cluster:/path/to/project/artifacts/ ./artifacts/
```

### Output Locations

| Output | Path |
|--------|------|
| Data splits | `data/splits/seed_{N}/scaffold/{train,val,test}.csv` |
| Features | `artifacts/features/seed_{N}/scaffold/{feature}/X_{split}.npz` |
| Models | `artifacts/models/baselines/seed_{N}/scaffold/{feature}/{model}/` |
| Reports | `artifacts/reports/benchmark_summary.csv` |

---

## Feature Types

Each feature family is an **independent** baseline input. Do NOT concatenate feature families for the formal baseline suite.

### Benchmarked Features

| Feature | Dimension | CLI Flag | Status |
|---------|-----------|----------|--------|
| Morgan | 2048 | `morgan` | ✅ Benchmarked (classification + regression) |
| MACCS | 167 | `maccs` | ✅ Benchmarked (classification + regression) |
| FP2 | 2048 | `fp2` | ✅ Benchmarked (classification + regression) |
| Descriptors (basic) | 13 | `descriptors_basic` | ✅ Benchmarked (classification + regression) |

### Implemented but Not Yet Benchmarked

| Feature | Dimension | CLI Flag | Status |
|---------|-----------|----------|--------|
| AtomPairs | 1024 | `atom_pairs` | ⚙️ Implemented but not benchmarked |
| Descriptors (extended) | 30 | `descriptors_extended` | ⚙️ Implemented but not benchmarked |
| Descriptors (all) | 45 | `descriptors_all` | ⚙️ Implemented but not benchmarked |

### Optional Exploratory (Not Part of Formal Baseline)

- **Combined / concatenated features** — concatenation of multiple feature families is optional exploratory work, not a formal baseline. Do not include in default benchmark runs.

**Formal baseline feature dimensions (do not use approximate values):**
- `morgan`: 2048 bits
- `descriptors_basic`: 13 descriptors
- `maccs`: 167 bits
- `fp2`: 2048 bits
- `atom_pairs`: 1024 bits
- `descriptors_extended`: 30 descriptors
- `descriptors_all`: 45 descriptors

---

## Model Types

### Benchmarked Models

| Model | CLI Flag | Status |
|-------|----------|--------|
| Random Forest | `rf` | ✅ Benchmarked (classification + regression) |
| XGBoost | `xgb` | ✅ Benchmarked (classification + regression) |
| LightGBM | `lgbm` | ✅ Benchmarked (classification + regression) |
| SVM | `svm` | ✅ Benchmarked (classification + regression) |
| Logistic Regression | `lr` | ✅ Benchmarked (classification) |
| KNN | `knn` | ✅ Benchmarked (classification + regression) |
| Ridge | `ridge` | ✅ Benchmarked (regression) |

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Feature dimension mismatch | Use only supported feature types |
| LGBM type error | Convert features to float32 |
| Invalid SMILES | Use auto-fix mapping in preprocessing |
| Missing dependencies | Check `requirements.txt` or `configs/environment.yml` |
| PyG graph building OOM on cluster | Set `num_workers=1` in `CachedGraphDataset` to avoid shared memory exhaustion |
| Denoising collate NoneType error | Filter `None` graphs during dataset build and cache load (fixed in `denoising.py`, `edge_masking.py`, `contrastive.py`, `attr_masking.py`, `context_prediction.py`) |
| Pretrain history not saved | `training_history.json` now saved after training completes |

---

## Baseline Feature Policy

**Each feature family is an independent baseline input. Feature families must NOT be concatenated for the formal baseline suite.**

### Formal Baseline Features (evaluated independently)

Benchmarked:
- `morgan` — Morgan/ECFP fingerprints (2048 bits)
- `maccs` — MACCS structural keys (167 bits)
- `fp2` — Daylight fingerprints (2048 bits)
- `descriptors_basic` — Physicochemical descriptors (13 dimensions)

Not yet benchmarked:
- `atom_pairs` — Atom pair fingerprints (1024 bits)
- `descriptors_extended` — Extended descriptors (30 dimensions)
- `descriptors_all` — All descriptors (45 dimensions)

### Optional Exploratory Features

- **Combined / concatenated features** — concatenation of multiple feature families is optional exploratory work, not a formal baseline. Do not include in default benchmark runs.
- Extended descriptor sets — `descriptors_extended` (30), `descriptors_all` (45) are available for exploratory analysis.

### Why This Matters

Concatenating feature families makes it impossible to attribute performance gains to specific representations. The formal baseline suite evaluates each feature family independently so that results are interpretable and comparable across representations.

When adding new feature families, create a new CLI flag and evaluate it independently. Do not modify existing feature pipelines.

## Baseline Performance Reference

All results: B3DB Groups A+B, scaffold split.

### Classical Classification (4 features x 6 models, 5 seeds)

| Rank | Feature | Model | Test AUC | Test F1 |
|------|---------|-------|----------|---------|
| 1 | MACCS (167) | LightGBM | 0.9535 ± 0.0388 | 0.9403 ± 0.0258 |
| 2 | MACCS | RF | 0.9504 ± 0.0462 | 0.9417 ± 0.0206 |
| 3 | Morgan (2048) | RF | 0.9504 ± 0.0330 | 0.9407 ± 0.0185 |
| 4 | MACCS | XGBoost | 0.9496 ± 0.0448 | 0.9381 ± 0.0233 |
| 5 | FP2 (2048) | XGBoost | 0.9459 ± 0.0352 | 0.9443 ± 0.0148 |
| 6 | FP2 | LightGBM | 0.9439 ± 0.0393 | 0.9409 ± 0.0168 |
| 7 | FP2 | RF | 0.9436 ± 0.0392 | 0.9415 ± 0.0171 |
| 8 | FP2 | SVM | 0.9402 ± 0.0270 | 0.9353 ± 0.0185 |
| 9 | MACCS | SVM | 0.9387 ± 0.0419 | 0.9349 ± 0.0253 |
| 10 | Desc_basic (13) | RF | 0.9377 ± 0.0597 | 0.9459 ± 0.0219 |
| 11 | Morgan | LightGBM | 0.9359 ± 0.0456 | 0.9392 ± 0.0221 |
| 12 | Morgan | XGBoost | 0.9343 ± 0.0475 | 0.9346 ± 0.0204 |
| 13 | Desc_basic | XGBoost | 0.9336 ± 0.0590 | 0.9422 ± 0.0231 |
| 14 | Desc_basic | LightGBM | 0.9333 ± 0.0656 | 0.9421 ± 0.0215 |
| 15 | Morgan | SVM | 0.9266 ± 0.0424 | 0.9381 ± 0.0170 |
| 16 | MACCS | LR | 0.9216 ± 0.0469 | 0.9243 ± 0.0293 |
| 17 | MACCS | KNN | 0.9207 ± 0.0612 | 0.9353 ± 0.0193 |
| 18 | Morgan | LR | 0.9137 ± 0.0747 | 0.9314 ± 0.0243 |
| 19 | FP2 | LR | 0.9106 ± 0.0467 | 0.9324 ± 0.0174 |
| 20 | Desc_basic | KNN | 0.9052 ± 0.0675 | 0.9267 ± 0.0192 |
| 21 | FP2 | KNN | 0.9019 ± 0.0688 | 0.9267 ± 0.0222 |
| 22 | Desc_basic | LR | 0.8912 ± 0.0521 | 0.9217 ± 0.0153 |
| 23 | Morgan | KNN | 0.8874 ± 0.0639 | 0.9316 ± 0.0190 |
| 24 | Desc_basic | SVM | 0.8797 ± 0.1027 | 0.9298 ± 0.0185 |

### Classical Regression (4 features, 5 seeds)

| Rank | Feature | Model | Test R² | Test RMSE |
|------|---------|-------|---------|-----------|
| 1 | Desc_basic | RF | 0.4488 ± 0.0918 | 0.5798 ± 0.0457 |
| 2 | MACCS | RF | 0.4397 ± 0.0922 | 0.5842 ± 0.0362 |
| 3 | Desc_basic | XGBoost | 0.4347 ± 0.1163 | 0.5860 ± 0.0513 |
| 4 | MACCS | SVM | 0.4212 ± 0.0574 | 0.5955 ± 0.0268 |
| 5 | Desc_basic | Ridge | 0.4139 ± 0.0120 | 0.6007 ± 0.0356 |
| ... | ... | ... | ... | ... |

### GNN Baselines (5 seeds)

| Model | Classification AUC | Regression R² |
|-------|--------------------|---------------|
| GAT | 0.9356 ± 0.0314 | 0.6408 ± 0.0357 |
| GIN | 0.9271 ± 0.0349 | 0.7062 ± 0.0473 |
| GCN | 0.9255 ± 0.0384 | 0.2820 ± 0.0723 |

### Transformer Baseline (5 seeds)

| Task | Primary Metric | Secondary |
|------|---------------|-----------|
| Classification | AUC 0.8820 ± 0.0768 | F1 0.9013 ± 0.0209 |
| Regression | R² 0.1911 ± 0.2111 | RMSE 0.7015 ± 0.1007 |

### Unified Leaderboard

**Classification (AUC):**

| Rank | Category | Model | Representation | Test AUC |
|------|----------|-------|----------------|----------|
| 1 | Classical | LightGBM | MACCS (167) | 0.9535 ± 0.039 |
| 2 | Classical | RF | MACCS (167) | 0.9504 ± 0.046 |
| 3 | Classical | RF | Morgan (2048) | 0.9504 ± 0.033 |
| 4 | Classical | XGBoost | MACCS (167) | 0.9496 ± 0.045 |
| 5 | Classical | XGBoost | FP2 (2048) | 0.9459 ± 0.035 |
| 6 | GNN | GAT | Molecular Graph | 0.9356 ± 0.031 |
| 7 | GNN | GIN | Molecular Graph | 0.9271 ± 0.035 |
| 8 | GNN | GCN | Molecular Graph | 0.9255 ± 0.038 |
| 9 | Transformer | Transformer | SMILES | 0.8820 ± 0.077 |

**Regression (R²):**

| Rank | Category | Model | Representation | Test R² |
|------|----------|-------|----------------|---------|
| 1 | GNN | GIN | Molecular Graph | 0.7062 ± 0.047 |
| 2 | GNN | GAT | Molecular Graph | 0.6408 ± 0.036 |
| 3 | GNN | GCN | Molecular Graph | 0.2820 ± 0.072 |
| 4 | Classical | RF | Desc_basic (13) | 0.4488 ± 0.092 |
| 5 | Transformer | Transformer | SMILES | 0.1911 ± 0.211 |

### Key Findings

- **Classification:** Classical methods (tree models + fingerprints) outperform GNN and Transformer. MACCS (167 bits) matches or exceeds Morgan (2048 bits), suggesting structural keys capture BBB-relevant features efficiently.
- **Regression:** GNN dominates classical methods, with GIN achieving R² 0.71 vs classical best 0.45. Graph structure is more informative for predicting continuous permeability values.
- **Transformer:** Weakest across both tasks, likely due to limited SMILES sequence-level signal for BBB permeability.

---

## Pretraining Results (ZINC22, Stage 3)

Pretraining on ZINC22 molecular graphs with three strategies: Property Prediction, Denoising, Transformer MLM. See `scripts/pretrain/run_pretrain_matrix.py` for the full experiment matrix.

### Pretraining Strategies

| Strategy | Task | Target | Backbone |
|----------|------|--------|----------|
| Property Prediction | Predict 11 molecular properties (logP, TPSA, MW, etc.) | Regression | GIN, GAT |
| Denoising | Reconstruct clean node features from noisy input | Node feature reconstruction | GIN, GAT |
| Transformer MLM | Masked language modeling on SMILES | Sequence modeling | Transformer |

### Pretraining Experiment Matrix

**14/17 experiments completed.** 3 still pending (D_E10_GIN_5M, D_E20_GIN_256_5M, T_E20_TRANS_512_5M).

#### Property Prediction Results (10 epochs, all converged)

| Experiment | Samples | Model | Initial Loss | Final Loss | Reduction |
|------------|---------|-------|-------------|------------|-----------|
| P_E10_GIN_100K | 100K | GIN-128d | 0.314 | 0.156 | -50% |
| P_E10_GIN_500K | 500K | GIN-128d | 0.224 | 0.111 | -50% |
| P_E10_GIN_1M | 1M | GIN-128d | 0.197 | 0.098 | -50% |
| P_E10_GIN_2M | 2M | GIN-128d | 0.211 | 0.102 | -52% |
| P_E10_GIN_5M | 5M | GIN-128d | 0.176 | 0.086 | -51% |
| P_E10_GAT_1M | 1M | GAT-128d | 0.221 | 0.127 | -42% |
| P_E20_GIN_256_5M | 5M | GIN-256d | 0.139 | 0.041 | **-71%** |

#### Denoising Results (10 epochs)

| Experiment | Model | Initial Loss | Final Loss | Reduction |
|------------|-------|-------------|------------|-----------|
| D_E10_GIN_100K | GIN-128d | 0.046 | 0.0007 | -98% |
| D_E10_GIN_1M | GIN-128d | 0.013 | 0.0002 | -99% |
| D_E10_GAT_1M | GAT-128d | 0.017 | 0.0003 | -98% |
| D_E10_GIN_5M | GIN-128d | 0.005 | 0.005 | ⏳ Re-running |

#### Transformer MLM Results (10 epochs)

| Experiment | Samples | Layers | Initial Loss | Final Loss | Reduction |
|------------|---------|--------|-------------|------------|-----------|
| T_E10_TRANS_100K | 100K | 4L-256d | 2.217 | 0.590 | -73% |
| T_E10_TRANS_1M | 1M | 4L-256d | 0.884 | 0.242 | -73% |
| T_E10_TRANS_5M | 5M | 4L-256d | 0.445 | 0.148 | -67% |
| T_E10_TRANS_1M_L6 | 1M | 6L-256d | 0.838 | 0.207 | -75% |
| T_E20_TRANS_512_5M | 5M | 8L-512d | - | - | ⏳ Re-running |

### Key Observations

- **Property Prediction** converges steadily across all sample sizes. Larger models (256d, 20ep) achieve significantly better loss (0.041).
- **Denoising** converges extremely fast (-99%), suggesting the task may be too easy — consider increasing noise std.
- **Transformer** shows clear benefits from larger datasets and deeper architectures.
- **3 experiments pending**: D_E10_GIN_5M, D_E20_GIN_256_5M, T_E20_TRANS_512_5M (flagship scale, may need reduced batch size).

### Pending Experiments

Run the remaining 3 after current jobs complete:

```bash
python scripts/pretrain/run_pretrain_matrix.py --run 11   # D_E10_GIN_5M
python scripts/pretrain/run_pretrain_matrix.py --run 13   # D_E20_GIN_256_5M
python scripts/pretrain/run_pretrain_matrix.py --run 17   # T_E20_TRANS_512_5M
```

### Pretrained Model Artifacts

All pretrained backbones are saved in `artifacts/models/pretrain/exp_matrix/{exp_id}/`:

```
{exp_id}/
├── *_pretrained_backbone.pt      # Final backbone weights (for fine-tuning)
├── *_pretrain_epoch_*.pt         # Per-epoch checkpoints (with history)
├── training_history.json          # Training loss curves
└── graph_cache_{N}.pkl            # Cached molecular graphs
```

---

## Citation

If you use this code or data, cite the B3DB database.

---

**Last Updated:** 2026-04-10
**Project Status:** Stage 4 (Fine-tuning) — code ready, pending pretrain experiments to complete
**Best Classification:** MACCS + LightGBM (AUC = 0.9535)
**Best Regression:** GIN (R² = 0.7062)
**Next Stage:** Run fine-tuning matrix (204 experiments × 5 seeds), aggregate results, compare against baselines
