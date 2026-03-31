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

**Best performance:** Random Forest + Morgan fingerprints (AUC: 0.9401 ± 0.0454)

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
python scripts/analysis/run_baseline_matrix.py      # 18 experiments
python scripts/analysis/aggregate_results.py        # Aggregate results
python scripts/analysis/generate_benchmark_summary.py  # Summary report
```

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
| Transformer | `src/transformer/` | Future | MolBERT, Graphormer |
| Pretrain | `src/pretrain/` | Future | ZINC22 pre-training |
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

### Implemented in Current Pipeline

| Feature | Dimension | CLI Flag | Status |
|---------|-----------|----------|--------|
| Morgan | 2048 | `morgan` | ✅ Working — benchmarked |
| Descriptors (basic) | 13 | `descriptors_basic` | ✅ Working — benchmarked |

### Available in Code (Extended Baseline — Not Yet Run)

| Feature | Dimension | CLI Flag | Status |
|---------|-----------|----------|--------|
| MACCS | 167 | `maccs` | ⚙️ Implemented but not benchmarked |
| FP2 | 2048 | `fp2` | ⚙️ Implemented but not benchmarked |
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

### Implemented in Current Pipeline

| Model | CLI Flag | Status |
|-------|----------|--------|
| Random Forest | `rf` | ✅ Working — benchmarked |
| XGBoost | `xgb` | ✅ Working — benchmarked |
| LightGBM | `lgbm` | ✅ Working — benchmarked |

### Available in Code (Extended Baseline — Not Yet Run)

| Model | CLI Flag | Status |
|-------|----------|--------|
| SVM | `svm` | ⚙️ Implemented but not benchmarked |
| Logistic Regression | `lr` | ⚙️ Implemented but not benchmarked |
| KNN | `knn` | ⚙️ Implemented but not benchmarked |

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Feature dimension mismatch | Use only supported feature types |
| LGBM type error | Convert features to float32 |
| Invalid SMILES | Use auto-fix mapping in preprocessing |
| Missing dependencies | Check `requirements.txt` or `configs/environment.yml` |

---

## Baseline Feature Policy

**Each feature family is an independent baseline input. Feature families must NOT be concatenated for the formal baseline suite.**

### Formal Baseline Features (evaluated independently)

Currently benchmarked:
- `morgan` — Morgan/ECFP fingerprints (2048 bits)
- `descriptors_basic` — Physicochemical descriptors (13 dimensions)

Planned for formal baseline (already implemented, pending benchmark runs):
- `maccs` — MACCS structural keys (167 bits)
- `fp2` — Daylight fingerprints (2048 bits)

### Optional Exploratory Features

- **Combined / concatenated features** — concatenation of multiple feature families is optional exploratory work, not a formal baseline. Do not include in default benchmark runs.
- Extended descriptor sets — `descriptors_extended` (30), `descriptors_all` (45) are available for exploratory analysis.

### Why This Matters

Concatenating feature families makes it impossible to attribute performance gains to specific representations. The formal baseline suite evaluates each feature family independently so that results are interpretable and comparable across representations.

When adding new feature families, create a new CLI flag and evaluate it independently. Do not modify existing feature pipelines.

## Baseline Performance Reference

**Established benchmark (B3DB Groups A+B, scaffold split, 3 seeds):**

| Rank | Feature | Model | Test AUC | Test F1 |
|------|---------|-------|----------|---------|
| 1 | Morgan | RF | 0.9401 ± 0.0454 | 0.9391 ± 0.0270 |
| 2 | Morgan | XGBoost | 0.9198 ± 0.0674 | 0.9335 ± 0.0310 |
| 3 | Morgan | LightGBM | 0.9195 ± 0.0619 | 0.9363 ± 0.0331 |
| 4 | descriptors_basic | RF | 0.9159 ± 0.0730 | 0.9397 ± 0.0270 |
| 5 | descriptors_basic | XGBoost | 0.9115 ± 0.0712 | 0.9370 ± 0.0308 |
| 6 | descriptors_basic | LightGBM | 0.9114 ± 0.0815 | 0.9395 ± 0.0298 |

Results for maccs, fp2, atom_pairs, and additional models (SVM, LR, KNN) are not yet in the formal benchmark. Run the extended baseline suite to populate these.

---

## Citation

If you use this code or data, cite the B3DB database.

---

**Last Updated:** 2026-03-31
**Project Status:** Baseline pipeline stable, research modules in development
**Best Model:** Random Forest + Morgan (AUC = 0.9401)
