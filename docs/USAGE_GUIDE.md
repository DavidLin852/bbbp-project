# BBB Permeability Prediction - Usage Guide

**Last Updated:** 2026-04-03
**Project Stage:** Baseline Phase Complete
**Target Users:** Researchers working on BBB prediction and molecular property prediction

---

## Table of Contents

1. [Project Structure Overview](#1-project-structure-overview)
2. [Local Development Workflow](#2-local-development-workflow)
3. [Git Workflow (Local Push / DSW Pull)](#3-git-workflow-local-push--dsw-pull)
4. [CFFF/DSW Workflow](#4-cfffdsw-workflow)
5. [Required Data Placement](#5-required-data-placement)
6. [Preprocessing Commands](#6-preprocessing-commands)
7. [Feature Computation Commands](#7-feature-computation-commands)
8. [Classical Classification Commands](#8-classical-classification-commands)
9. [Classical Regression Commands](#9-classical-regression-commands)
10. [Graph Baseline Commands](#10-graph-baseline-commands)
11. [Sequence/Transformer Baseline Commands](#11-sequencetransformer-baseline-commands)
12. [Result Aggregation Commands](#12-result-aggregation-commands)
13. [Archiving Outputs to CFFF](#13-archiving-outputs-to-cfff)
14. [Updating Benchmark Summaries](#14-updating-benchmark-summaries)

---

## 1. Project Structure Overview

### Top-Level Directory Structure

```
bbbp-project/
├── configs/                   # Environment and dependency configurations
│   ├── environment.yml        # Conda environment specification
│   └── *.example             # Example configuration files
│
├── data/                      # Data files (mixed gitignore)
│   ├── raw/                  # ✅ Tracked in Git: Original B3DB datasets
│   │   ├── B3DB_classification.tsv
│   │   └── B3DB_regression.tsv
│   ├── splits/               # ❌ Gitignored: Generated train/val/test splits
│   │   └── seed_{N}/
│   │       ├── classification_scaffold/
│   │       └── regression_scaffold/
│   └── transport_mechanisms/ # Transport mechanism datasets
│
├── artifacts/                 # ❌ Gitignored: All generated outputs
│   ├── features/             # Computed features (.npz files)
│   │   └── seed_{N}/scaffold/{feature}/
│   ├── models/               # Trained models and checkpoints
│   │   ├── baselines/        # Classical models (.joblib)
│   │   ├── gnn/              # GNN models (.pt)
│   │   └── transformer/      # Transformer models (.pt)
│   └── reports/              # Benchmark results and summaries
│       ├── benchmark_summary.csv
│       ├── gnn/
│       └── transformer/
│
├── scripts/                  # Executable scripts (✅ Tracked in Git)
│   ├── baseline/             # Classical baseline pipeline
│   │   ├── 01_preprocess_b3db.py
│   │   ├── 02_compute_features.py
│   │   └── 03_train_baselines.py
│   ├── gnn/                  # GNN baseline pipeline
│   │   └── run_gnn_benchmark.py
│   ├── transformer/          # Transformer baseline pipeline
│   │   └── run_transformer_benchmark.py
│   └── analysis/             # Result aggregation and analysis
│       ├── run_baseline_matrix.py
│       ├── aggregate_results.py
│       └── generate_benchmark_summary.py
│
├── src/                      # Source modules (✅ Tracked in Git)
│   ├── config.py             # Central configuration
│   ├── data/                 # Data loading and preprocessing
│   ├── features/             # Feature extraction
│   ├── models/               # Model implementations
│   ├── gnn/                  # GNN models
│   ├── transformer/          # Transformer models
│   ├── train/                # Training utilities
│   ├── evaluate/             # Evaluation metrics
│   └── utils/                # General utilities
│
├── docs/                     # Documentation (✅ Tracked in Git)
│   ├── CURRENT_BASELINE_SUMMARY.md
│   ├── BASELINE_LANDSCAPE.md
│   ├── SEQUENCE_BASELINE_COMPARISON.md
│   └── USAGE_GUIDE.md        # This file
│
├── archive/                  # Legacy code (✅ Tracked in Git)
│   ├── old_scripts/
│   ├── old_src/
│   └── old_web/
│
├── requirements.txt          # Python dependencies
├── requirements-research.txt # Research dependencies (PyTorch, etc.)
├── CLAUDE.md                 # AI assistant documentation
├── PROJECT_CONTEXT.md        # Project context and roadmap
└── README.md                 # Project overview
```

### Key Git Tracking Rules

**Tracked in Git:**
- ✅ All source code (`src/`, `scripts/`)
- ✅ Documentation (`docs/`, `*.md`)
- ✅ Configuration files (`configs/`, `requirements*.txt`)
- ✅ Raw data (`data/raw/`)
- ✅ Legacy code (`archive/`)

**NOT Tracked in Git (gitignored):**
- ❌ Generated data splits (`data/splits/`)
- ❌ Computed features (`artifacts/features/`)
- ❌ Trained models (`artifacts/models/`)
- ❌ Result reports (`artifacts/reports/`)
- ❌ Temporary files

---

## 2. Local Development Workflow

### 2.1 Initial Setup

**Step 1: Clone and Navigate**

```bash
# Clone repository (if not already done)
cd /path/to/your/workspace
git clone <repository-url> bbbp-project
cd bbbp-project
```

**Step 2: Create Conda Environment**

```bash
# Create environment from YAML
conda env create -f configs/environment.yml
conda activate bbb

# OR create manually
conda create -n bbb python=3.10
conda activate bbb
pip install -r requirements.txt
pip install -r requirements-research.txt  # For GNN/Transformer
```

**Step 3: Verify Installation**

```bash
# Check Python version
python --version  # Should be 3.10+

# Check key packages
python -c "import rdkit; import sklearn; import torch; print('OK')"
```

### 2.2 Daily Development Workflow

**Workflow:**

```bash
# 1. Pull latest changes
git pull origin master

# 2. Activate environment
conda activate bbb

# 3. Run your experiments (see sections 6-12)

# 4. Check results
ls artifacts/reports/

# 5. Commit changes (if any)
git status
git add <files>
git commit -m "description"
git push
```

**Testing Code Changes:**

```bash
# After modifying code, test the pipeline:
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"
python scripts/baseline/02_compute_features.py --seed 0 --feature morgan
python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models rf
```

---

## 3. Git Workflow (Local Push / DSW Pull)

### 3.1 Local Development → Push to Remote

**When working locally:**

```bash
# 1. Make changes and test
# ... (run experiments, verify results)

# 2. Stage and commit
git status                                          # Check what changed
git add src/ scripts/ docs/ *.md                    # Stage source files
git commit -m "feat: add new baseline model"        # Commit with message

# 3. Push to remote
git push origin master                              # Push to GitHub/GitLab
```

**Important:**
- ❌ DO NOT commit `artifacts/`, `data/splits/`, or other generated files
- ✅ Only commit source code, documentation, and configuration
- ✅ Use descriptive commit messages (see below)

**Commit Message Format:**

```bash
# Features
feat: add new molecular fingerprint feature

# Bug fixes
fix: correct scaffold split logic for edge cases

# Documentation
docs: update usage guide with new commands

# Refactoring
refactor: simplify feature extraction pipeline

# Tests
test: add unit tests for data preprocessing
```

### 3.2 DSW/Server → Pull Locally

**When pulling changes from DSW/server:**

```bash
# 1. Pull latest changes
git pull origin master

# 2. Resolve merge conflicts if any
# ... (edit files, resolve conflicts)

# 3. Re-run experiments if needed
# Generated outputs (splits, features, models) are not in Git
# You may need to regenerate them:
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"
```

---

## 4. CFFF/DSW Workflow

### 4.1 Transferring Code to CFFF

**From Local to CFFF:**

```bash
# 1. Transfer code (exclude generated outputs)
rsync -avz \
    --exclude 'artifacts/' \
    --exclude 'data/splits/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    bbbp-project/ user@cfff:/path/to/directory/

# 2. OR use git (recommended)
# On CFFF:
cd /path/to/directory/
git clone <repository-url> bbbp-project
cd bbbp-project
```

### 4.2 Running on CFFF

**Step 1: Set Up Environment**

```bash
# On CFFF cluster
module load python/3.10  # Or appropriate module
conda env create -f configs/environment.yml
conda activate bbb
```

**Step 2: Verify Data**

```bash
# Check raw data exists
ls -lh data/raw/B3DB_classification.tsv  # Should be ~2.6 MB
ls -lh data/raw/B3DB_regression.tsv      # Should be ~413 KB

# If missing, transfer from local:
rsync -avz user@local:/path/to/bbbp-project/data/raw/ data/raw/
```

**Step 3: Run Experiments**

```bash
# Quick test
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"

# Full benchmark
python scripts/analysis/run_baseline_matrix.py
```

**Step 4: Retrieve Results**

```bash
# From CFFF to local
rsync -avz user@cfff:/path/to/bbbp-project/artifacts/reports/ artifacts/reports/
```

### 4.3 Batch Job Submission (Optional)

If using SLURM or similar:

```bash
# Create submission script
cat > run_baseline.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=bbb_baseline
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

conda activate bbb
python scripts/analysis/run_baseline_matrix.py
EOF

# Submit job
sbatch run_baseline.sh
```

---

## 5. Required Data Placement

### 5.1 Data File Locations

**Raw Data (tracked in Git):**

```
data/raw/
├── B3DB_classification.tsv    # MUST EXIST: Classification dataset
└── B3DB_regression.tsv         # MUST EXIST: Regression dataset
```

**Generated Splits (gitignored, will be created):**

```
data/splits/
└── seed_{N}/                  # N = 0, 1, 2, 3, 4
    ├── classification_scaffold/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    └── regression_scaffold/
        ├── train.csv
        ├── val.csv
        └── test.csv
```

### 5.2 Transferring Data

**If raw data is missing:**

```bash
# From local to CFFF
rsync -avz data/raw/ user@cfff:/path/to/bbbp-project/data/raw/

# From CFFF to local
rsync -avz user@cfff:/path/to/bbbp-project/data/raw/ data/raw/
```

**Data Sources:**
- B3DB datasets should be obtained from the official B3DB repository
- Place in `data/raw/` before running preprocessing

---

## 6. Preprocessing Commands

### 6.1 Single Seed Preprocessing

**Classification Task:**

```bash
# Scaffold split for classification
python scripts/baseline/01_preprocess_b3db.py \
    --seed 0 \
    --groups "A,B" \
    --task classification

# Output:
# data/splits/seed_0/classification_scaffold/train.csv
# data/splits/seed_0/classification_scaffold/val.csv
# data/splits/seed_0/classification_scaffold/test.csv
```

**Regression Task:**

```bash
# Scaffold split for regression
python scripts/baseline/01_preprocess_b3db.py \
    --seed 0 \
    --groups "A,B" \
    --task regression

# Output:
# data/splits/seed_0/regression_scaffold/train.csv
# data/splits/seed_0/regression_scaffold/val.csv
# data/splits/seed_0/regression_scaffold/test.csv
```

### 6.2 Multiple Seeds (Loop)

**Bash Loop:**

```bash
# Preprocess all 5 seeds for classification
for seed in 0 1 2 3 4; do
    python scripts/baseline/01_preprocess_b3db.py \
        --seed $seed \
        --groups "A,B" \
        --task classification
done

# Preprocess all 5 seeds for regression
for seed in 0 1 2 3 4; do
    python scripts/baseline/01_preprocess_b3db.py \
        --seed $seed \
        --groups "A,B" \
        --task regression
done
```

### 6.3 Command-Line Options

```bash
python scripts/baseline/01_preprocess_b3db.py --help

# Options:
# --seed: Random seed (default: 0)
# --groups: B3DB groups to use (default: "A,B")
# --task: "classification" or "regression" (default: classification)
# --split_type: "scaffold" or "random" (default: scaffold)
# --train_ratio: Training set ratio (default: 0.8)
# --val_ratio: Validation set ratio (default: 0.1)
# --test_ratio: Test set ratio (default: 0.1)
```

---

## 7. Feature Computation Commands

### 7.1 Single Feature, Single Seed

```bash
# Compute Morgan fingerprints for seed 0
python scripts/baseline/02_compute_features.py \
    --seed 0 \
    --feature morgan

# Output:
# artifacts/features/seed_0/scaffold/morgan/X_train.npz
# artifacts/features/seed_0/scaffold/morgan/X_val.npz
# artifacts/features/seed_0/scaffold/morgan/X_test.npz
```

### 7.2 Multiple Features

```bash
# Compute all available features for seed 0
for feature in morgan descriptors_basic maccs fp2 atom_pairs; do
    python scripts/baseline/02_compute_features.py \
        --seed 0 \
        --feature $feature
done
```

### 7.3 Multiple Seeds and Features

```bash
# Nested loop: 5 seeds × 6 features
for seed in 0 1 2 3 4; do
    for feature in morgan descriptors_basic maccs fp2 atom_pairs; do
        python scripts/baseline/02_compute_features.py \
            --seed $seed \
            --feature $feature
    done
done
```

### 7.4 Available Feature Types

| Feature | CLI Flag | Dimension | Description |
|---------|----------|-----------|-------------|
| Morgan | `morgan` | 2048 | ECFP4 circular fingerprints |
| MACCS | `maccs` | 167 | MACCS structural keys |
| FP2 | `fp2` | 2048 | Daylight fingerprints |
| Atom Pairs | `atom_pairs` | 1024 | Atom pair fingerprints |
| Descriptors (basic) | `descriptors_basic` | 13 | Basic physicochemical |
| Descriptors (extended) | `descriptors_extended` | 30 | Extended physicochemical |
| Descriptors (all) | `descriptors_all` | 45 | All physicochemical |

---

## 8. Classical Classification Commands

### 8.1 Single Model, Single Feature

```bash
# Train Random Forest on Morgan fingerprints
python scripts/baseline/03_train_baselines.py \
    --seed 0 \
    --feature morgan \
    --models rf

# Output:
# artifacts/models/baselines/seed_0/scaffold/morgan/rf/model.joblib
# artifacts/models/baselines/seed_0/scaffold/morgan/rf/comparison.json
```

### 8.2 Multiple Models

```bash
# Train all classical models on Morgan
python scripts/baseline/03_train_baselines.py \
    --seed 0 \
    --feature morgan \
    --models rf,xgb,lgbm,svm,lr,knn
```

### 8.3 Full Benchmark Matrix

**Automated Script:**

```bash
# Run 18 experiments: 3 seeds × 2 features × 3 models
python scripts/analysis/run_baseline_matrix.py

# This runs:
# - Seeds: 0, 1, 2
# - Features: morgan, descriptors_basic
# - Models: rf, xgb, lgbm
```

**Manual Loop:**

```bash
# Manual full benchmark
for seed in 0 1 2; do
    for feature in morgan descriptors_basic; do
        for model in rf xgb lgbm; do
            python scripts/baseline/03_train_baselines.py \
                --seed $seed \
                --feature $feature \
                --models $model
        done
    done
done
```

### 8.4 Available Models

| Model | CLI Flag | Description |
|-------|----------|-------------|
| Random Forest | `rf` | Random Forest Classifier |
| XGBoost | `xgb` | XGBoost Classifier |
| LightGBM | `lgbm` | LightGBM Classifier |
| SVM | `svm` | Support Vector Machine |
| Logistic Regression | `lr` | Logistic Regression |
| KNN | `knn` | K-Nearest Neighbors |

---

## 9. Classical Regression Commands

### 9.1 Single Model

```bash
# Train Random Forest regressor on Morgan
python scripts/baseline/03_train_baselines.py \
    --seed 0 \
    --feature morgan \
    --models rf \
    --task regression
```

### 9.2 Full Regression Benchmark

```bash
# Loop over seeds and models
for seed in 0 1 2 3 4; do
    for model in rf xgb lgbm; do
        python scripts/baseline/03_train_baselines.py \
            --seed $seed \
            --feature morgan \
            --models $model \
            --task regression
    done
done
```

---

## 10. Graph Baseline Commands

### 10.1 Quick Test (Dry Run)

```bash
# Test with 1 seed, reduced epochs
python scripts/gnn/run_gnn_benchmark.py --dry_run

# Runs:
# - 1 seed (seed 0)
# - 5 epochs max (instead of 300)
# - Tests both classification and regression
```

### 10.2 Classification Benchmark

```bash
# Run GNN classification benchmark (5 seeds)
python scripts/gnn/run_gnn_benchmark.py \
    --tasks classification \
    --seeds 0,1,2,3,4

# Runs GCN, GIN, GAT on classification task
# Output: artifacts/reports/gnn/gnn_benchmark_scaffold_*.csv
```

### 10.3 Regression Benchmark

```bash
# Run GNN regression benchmark (5 seeds)
python scripts/gnn/run_gnn_benchmark.py \
    --tasks regression \
    --seeds 0,1,2,3,4

# Runs GCN, GIN, GAT on regression task
```

### 10.4 Both Tasks

```bash
# Run full GNN benchmark (classification + regression)
python scripts/gnn/run_gnn_benchmark.py \
    --tasks classification,regression \
    --seeds 0,1,2,3,4

# Estimated time: 1-2 hours (GPU), 4-8 hours (CPU)
```

### 10.5 GNN Configuration

```bash
# Custom configuration
python scripts/gnn/run_gnn_benchmark.py \
    --tasks classification \
    --seeds 0,1,2,3,4 \
    --batch_size 64 \
    --epochs 300 \
    --early_stopping_patience 30 \
    --learning_rate 0.001
```

---

## 11. Sequence/Transformer Baseline Commands

### 11.1 Quick Test (Dry Run)

```bash
# Test with 1 seed, reduced epochs
python scripts/transformer/run_transformer_benchmark.py --dry_run

# Runs:
# - 1 seed (seed 0)
# - 5 epochs max (instead of 100)
# - Tests both classification and regression
```

### 11.2 Classification Benchmark

```bash
# Run Transformer classification benchmark (5 seeds)
python scripts/transformer/run_transformer_benchmark.py \
    --tasks classification \
    --seeds 0,1,2,3,4

# Output: artifacts/reports/transformer/transformer_benchmark_scaffold_*.csv
# Estimated time: 1-2 hours (GPU), 4-8 hours (CPU)
```

### 11.3 Regression Benchmark

```bash
# Run Transformer regression benchmark (5 seeds)
python scripts/transformer/run_transformer_benchmark.py \
    --tasks regression \
    --seeds 0,1,2,3,4
```

### 11.4 Both Tasks

```bash
# Run full Transformer benchmark
python scripts/transformer/run_transformer_benchmark.py \
    --tasks classification,regression \
    --seeds 0,1,2,3,4
```

### 11.5 Custom Configuration

```bash
# Custom hyperparameters
python scripts/transformer/run_transformer_benchmark.py \
    --tasks classification \
    --seeds 0,1,2,3,4 \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --max_smiles_length 128
```

---

## 12. Result Aggregation Commands

### 12.1 Aggregate Classical Results

```bash
# Aggregate all classical baseline results
python scripts/analysis/aggregate_results.py

# Output:
# artifacts/reports/baseline_results_master.csv
# artifacts/reports/baseline_summary_by_feature.csv
# artifacts/reports/baseline_summary_by_model.csv
```

### 12.2 Generate Benchmark Summary

```bash
# Generate comprehensive benchmark summary
python scripts/analysis/generate_benchmark_summary.py

# Output:
# artifacts/reports/benchmark_summary.csv
```

### 12.3 View Results

```bash
# View benchmark summary
cat artifacts/reports/benchmark_summary.csv

# Or with pandas (if installed)
python -c "import pandas as pd; print(pd.read_csv('artifacts/reports/benchmark_summary.csv').to_string())"
```

---

## 13. Archiving Outputs to CFFF

### 13.1 Directory Structure for Archiving

**On CFFF Shared Storage:**

```
/cfff/shared/bbbp-project/
├── results/
│   ├── 2026-04-03_classical_baselines/
│   │   ├── benchmark_summary.csv
│   │   ├── models/
│   │   └── logs/
│   ├── 2026-04-03_gnn_baselines/
│   │   ├── gnn_benchmark_scaffold_*.csv
│   │   ├── checkpoints/
│   │   └── logs/
│   └── 2026-04-03_transformer_baselines/
│       ├── transformer_benchmark_scaffold_*.csv
│       ├── checkpoints/
│       └── logs/
```

### 13.2 Archiving Commands

**Archive Results:**

```bash
# Create timestamped archive
TIMESTAMP=$(date +%Y%m%d)
ARCHIVE_DIR="/cfff/shared/bbbp-project/results/${TIMESTAMP}_baselines"
mkdir -p $ARCHIVE_DIR

# Copy result reports
cp -r artifacts/reports/* $ARCHIVE_DIR/

# Copy model checkpoints (optional, if space permits)
cp -r artifacts/models/* $ARCHIVE_DIR/

# Verify
ls -lh $ARCHIVE_DIR
```

**Archive with Metadata:**

```bash
# Create archive with README
cat > $ARCHIVE_DIR/README.md << 'EOF'
# BBB Baseline Results - 2026-04-03

## Contents
- benchmark_summary.csv: Classical baseline results
- gnn/: GNN baseline results
- transformer/: Transformer baseline results

## Configuration
- Dataset: B3DB Groups A+B
- Split: Scaffold stratified (80:10:10)
- Seeds: 5 seeds (0-4)
- Hardware: [GPU/CPU specification]

## Key Results
- Best classification: RF + Morgan (AUC: 0.9401)
- Best regression: GIN (R²: 0.7062)

EOF
```

---

## 14. Updating Benchmark Summaries

### 14.1 After Running New Experiments

**Step 1: Aggregate Results**

```bash
# If running classical baselines
python scripts/analysis/aggregate_results.py
python scripts/analysis/generate_benchmark_summary.py
```

**Step 2: Update Summary Documents**

```bash
# Manually update docs/CURRENT_BASELINE_SUMMARY.md
# with new results from artifacts/reports/
```

**Step 3: Commit Documentation**

```bash
git add docs/ artifacts/reports/*.csv
git commit -m "docs: update baseline summary with new results"
git push
```

### 14.2 Result File Locations

**Classical Baselines:**
```
artifacts/reports/
├── benchmark_summary.csv              # Main summary
├── baseline_results_master.csv        # All results
├── baseline_summary_by_feature.csv    # By feature
└── baseline_summary_by_model.csv      # By model
```

**GNN Baselines:**
```
artifacts/reports/gnn/
└── gnn_benchmark_scaffold_YYYYMMDD_HHMMSS.csv
```

**Transformer Baselines:**
```
artifacts/reports/transformer/
└── transformer_benchmark_scaffold_YYYYMMDD_HHMMSS.csv
```

### 14.3 Creating Unified Leaderboard

**Manual Integration:**

```bash
# Combine all results into one CSV
cat > artifacts/reports/unified_leaderboard.csv << 'EOF'
category,model,representation,task,test_auc_mean,test_auc_std,test_r2_mean,test_r2_std
Classical,RF,Morgan,classification,0.9401,0.0454,,
Graph,GAT,Molecular Graph,classification,0.9356,0.0314,,
Sequence,Transformer,SMILES,classification,0.8822,0.0756,,
Graph,GIN,Molecular Graph,regression,,,0.7062,0.0473
Sequence,Transformer,SMILES,regression,,,0.1911,0.2014
EOF
```

---

## 15. Troubleshooting

### 15.1 Common Issues

**Issue: Missing splits**

```bash
# Error: data/splits/seed_0/scaffold/test.csv not found
# Solution: Run preprocessing first
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"
```

**Issue: Missing features**

```bash
# Error: artifacts/features/seed_0/scaffold/morgan/X_train.npz not found
# Solution: Run feature computation first
python scripts/baseline/02_compute_features.py --seed 0 --feature morgan
```

**Issue: GPU out of memory**

```bash
# Solution: Reduce batch size
python scripts/gnn/run_gnn_benchmark.py --tasks classification --batch_size 32

# Or run on CPU
python scripts/gnn/run_gnn_benchmark.py --tasks classification --device cpu
```

**Issue: Import errors**

```bash
# Error: ModuleNotFoundError: No module named 'src'
# Solution: Ensure you're running from project root
cd /path/to/bbbp-project
python scripts/baseline/01_preprocess_b3db.py --seed 0
```

---

## 16. Quick Reference Commands

### Complete Pipeline (Classical)

```bash
# 1. Preprocess
for seed in 0 1 2; do
    python scripts/baseline/01_preprocess_b3db.py --seed $seed --groups "A,B"
done

# 2. Compute features
for seed in 0 1 2; do
    python scripts/baseline/02_compute_features.py --seed $seed --feature morgan
done

# 3. Train models
for seed in 0 1 2; do
    python scripts/baseline/03_train_baselines.py --seed $seed --feature morgan --models rf,xgb,lgbm
done

# 4. Aggregate results
python scripts/analysis/aggregate_results.py
python scripts/analysis/generate_benchmark_summary.py
```

### Complete Pipeline (GNN)

```bash
# Run full GNN benchmark
python scripts/gnn/run_gnn_benchmark.py --tasks classification,regression --seeds 0,1,2,3,4
```

### Complete Pipeline (Transformer)

```bash
# Run full Transformer benchmark
python scripts/transformer/run_transformer_benchmark.py --tasks classification,regression --seeds 0,1,2,3,4
```

---

**End of Usage Guide**

For questions or issues, refer to:
- `CLAUDE.md` - Comprehensive project documentation
- `PROJECT_CONTEXT.md` - Project context and roadmap
- `docs/CURRENT_BASELINE_SUMMARY.md` - Current baseline results
