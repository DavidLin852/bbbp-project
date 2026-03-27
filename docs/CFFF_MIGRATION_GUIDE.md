# CFFF Migration Guide - BBB Baseline Pipeline

**Date:** 2025-03-27
**Target:** Baseline pipeline (RF/XGB/LGBM models with B3DB dataset)
**Status:** Ready for migration ✅

---

## Overview

This guide provides step-by-step instructions for migrating the BBB baseline pipeline to CFFF platform.

**Scope:**
- ✅ Baseline pipeline only (scripts/baseline/, scripts/analysis/)
- ✅ Training classical ML models (RF, XGB, LGBM)
- ✅ Generating benchmark results
- ❌ Deep learning models (GNN, VAE, GAN)
- ❌ Streamlit web interface

---

## Prerequisites

### Local Machine

1. ✅ Git repository initialized and cleaned
2. ✅ README.md updated with current workflow
3. ✅ Configuration refactored (src/config/)
4. ✅ Dependencies split (requirements-baseline.txt)
5. ✅ Scripts reorganized (scripts/analysis/exploratory/)

### CFFF Platform

- DSW (Dev Service Writer) workspace
- Access to conda/pip
- Python 3.10+ available

---

## Minimal Directory Layout on CFFF

```
/cfff/work/bbb_project/
├── scripts/
│   ├── baseline/
│   │   ├── 01_preprocess_b3db.py
│   │   ├── 02_compute_features.py
│   │   └── 03_train_baselines.py
│   └── analysis/
│       ├── aggregate_results.py
│       ├── generate_benchmark_summary.py
│       └── run_baseline_matrix.py
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── paths.py
│   │   ├── baseline.py
│   │   └── research.py
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── train/
│   ├── evaluate/
│   └── utils/
├── data/
│   └── raw/
│       ├── B3DB_classification.tsv
│       └── B3DB_regression.tsv
├── docs/
│   └── (reference documentation)
├── requirements-baseline.txt
├── configs/
│   └── environment-baseline.yml
├── README.md
└── .gitignore
```

**Excluded from Git (will be generated):**
- `data/splits/` - Generated scaffold splits
- `artifacts/features/` - Computed features
- `artifacts/models/` - Trained models
- `artifacts/reports/` - Benchmark reports

---

## Migration Steps

### Phase 1: DSW Setup (Interactive)

**Goal:** Set up environment and verify baseline pipeline

#### Step 1: Clone Repository

```bash
# In DSW terminal
cd /cfff/work/
git clone <repository-url> bbb_project
cd bbb_project

# Verify structure
ls -la scripts/baseline/
ls -la src/
ls -la data/raw/
```

**Expected:**
- ✅ 3 scripts in scripts/baseline/
- ✅ 15 modules in src/
- ✅ 2 data files in data/raw/

---

#### Step 2: Create Conda Environment

```bash
# Create baseline environment
conda env create -f configs/environment-baseline.yml

# Activate environment
conda activate bbb-baseline

# Verify installation
python -c "import rdkit; import sklearn; import xgboost; import lightgbm; print('OK: All dependencies installed')"
```

**Expected output:**
```
OK: All dependencies installed
```

**If error occurs:**
```bash
# Manual installation
conda create -n bbb-baseline python=3.10
conda activate bbb-baseline
conda install -c conda-forge rdkit scikit-learn xgboost lightgbm pandas numpy
```

---

#### Step 3: Quick Verification Test

```bash
# Test Step 1: Preprocessing (30 seconds)
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"

# Test Step 2: Feature computation (30 seconds)
python scripts/baseline/02_compute_features.py --seed 0 --feature morgan

# Test Step 3: Training (2 minutes)
python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models rf
```

**Expected outputs:**
```
# Step 1
Created data/splits/seed_0/scaffold/train.csv (2994 samples)
Created data/splits/seed_0/scaffold/val.csv (374 samples)
Created data/splits/seed_0/scaffold/test.csv (375 samples)

# Step 2
Computed features: artifacts/features/seed_0/scaffold/morgan/X_train.npz

# Step 3
Trained Random Forest
Test AUC: 0.9641
Model saved to: artifacts/models/baselines/seed_0/scaffold/morgan/rf/
```

**If errors occur:**
- Check `data/splits/seed_0/scaffold/` exists
- Check `artifacts/features/seed_0/scaffold/morgan/` exists
- Check conda environment is activated

---

#### Step 4: Verify Benchmark Results

```bash
# Generate benchmark summary
python scripts/analysis/generate_benchmark_summary.py

# View results
cat artifacts/reports/benchmark_summary.csv
```

**Expected output:**
```
rank,feature,model_name,test_auc_mean,test_auc_std,...
1,morgan,rf,0.9401,0.0454,...
```

---

### Phase 2: Batch/AI Tasks (Non-Interactive)

**Goal:** Run full benchmark matrix

#### Step 1: Prepare Batch Job

Create batch script `run_benchmark.sh`:

```bash
#!/bin/bash
# BBB Baseline Benchmark - Batch Job

echo "Starting BBB baseline benchmark..."
date

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bbb-baseline

# Run full benchmark matrix (18 experiments)
python scripts/analysis/run_baseline_matrix.py \
    --seeds 0,1,2 \
    --splits scaffold \
    --features morgan,descriptors_basic \
    --models rf,xgb,lgbm

# Aggregate results
python scripts/analysis/aggregate_results.py

# Generate summary
python scripts/analysis/generate_benchmark_summary.py

echo "Benchmark complete!"
date

# Copy results to output directory
mkdir -p /cfff/output/bbb_benchmark/
cp artifacts/reports/benchmark_summary.csv /cfff/output/bbb_benchmark/
cp artifacts/reports/benchmark_report.txt /cfff/output/bbb_benchmark/
```

---

#### Step 2: Submit Batch Job

```bash
# Make script executable
chmod +x run_benchmark.sh

# Submit to batch system
# (Adjust command based on CFFF batch system)
bsub -n 4 -W 4:00 -o benchmark.log ./run_benchmark.sh

# Or using CFFF-specific command
submit_job ./run_benchmark.sh
```

**Expected runtime:** 30-60 minutes (18 experiments)

---

#### Step 3: Monitor Results

```bash
# Monitor progress
tail -f benchmark.log

# Check intermediate results
ls -la artifacts/models/baselines/seed_*/scaffold/*/rf/comparison.json

# View final results
cat /cfff/output/bbb_benchmark/benchmark_summary.csv
```

---

## Minimal Verification Commands (In Order)

### Level 1: Environment Setup (5 minutes)

```bash
# 1. Clone repository
git clone <repo-url> && cd bbb_project

# 2. Create environment
conda env create -f configs/environment-baseline.yml
conda activate bbb-baseline

# 3. Verify imports
python -c "import rdkit, sklearn, xgboost, lightgbm; print('OK')"
```

**Success criteria:**
- ✅ Environment created
- ✅ All imports work

---

### Level 2: Single Experiment (5 minutes)

```bash
# 1. Preprocess
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"
# Check: data/splits/seed_0/scaffold/train.csv exists

# 2. Compute features
python scripts/baseline/02_compute_features.py --seed 0 --feature morgan
# Check: artifacts/features/seed_0/scaffold/morgan/X_train.npz exists

# 3. Train model
python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models rf
# Check: artifacts/models/baselines/seed_0/scaffold/morgan/rf/model.joblib exists
```

**Success criteria:**
- ✅ All 3 steps complete
- ✅ Output files exist
- ✅ Test AUC > 0.90

---

### Level 3: Full Benchmark (60 minutes)

```bash
# Run full benchmark matrix
python scripts/analysis/run_baseline_matrix.py

# Generate summary
python scripts/analysis/generate_benchmark_summary.py

# Check results
cat artifacts/reports/benchmark_summary.csv
```

**Success criteria:**
- ✅ 18 experiments complete
- ✅ Benchmark summary generated
- ✅ Best baseline: RF + Morgan, AUC > 0.94

---

## What to Clone vs Upload

### Clone from Git (Source Code)

**Via Git:**
```bash
git clone <repository-url> bbb_project
cd bbb_project
```

**Includes:**
- ✅ scripts/ (baseline and analysis)
- ✅ src/ (all modules)
- ✅ docs/ (documentation)
- ✅ requirements-baseline.txt
- ✅ configs/environment-baseline.yml
- ✅ README.md
- ✅ .gitignore

**Excluded by .gitignore:**
- ❌ data/splits/ (will be generated)
- ❌ artifacts/features/ (will be generated)
- ❌ artifacts/models/ (will be generated)

---

### Upload as Data (Large Files)

**Method 1: Direct upload (if small)**
```bash
# On local machine
scp data/raw/B3DB_classification.tsv user@cfff:/cfff/work/bbb_project/data/raw/

# Or use CFFF web interface to upload
# Upload to: /cfff/work/bbb_project/data/raw/
```

**Method 2: Via CFFF data service**
```bash
# Upload B3DB datasets to CFFF data storage
# Then symlink to project directory
ln -s /cfff/data/B3DB_classification.tsv /cfff/work/bbb_project/data/raw/
```

**Files to upload:**
- ✅ `data/raw/B3DB_classification.tsv` (2.6 MB)
- ✅ `data/raw/B3DB_regression.tsv` (413 KB)

**Total:** ~3 MB

---

## Directory Cleanup

### Remove Unnecessary Files (Optional)

```bash
# Remove research scripts (not needed for baseline)
rm -rf scripts/analysis/exploratory/

# Remove archived code (not needed for baseline)
rm -rf archive/

# Remove research documentation (not needed for baseline)
rm -rf docs/OLD_*.md

# Remove legacy requirements (not needed for baseline)
rm -f requirements-legacy.txt
rm -f configs/environment.yml

# Keep only essential files
ls scripts/baseline/
ls scripts/analysis/
ls src/
ls data/raw/
```

**Result:** Project size ~50 MB (down from ~350 MB)

---

## Common Issues

### Issue 1: Module Import Error

**Error:**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Ensure you're in project root
cd /cfff/work/bbb_project

# Verify scripts are there
ls scripts/baseline/

# Run with absolute path
python /cfff/work/bbb_project/scripts/baseline/01_preprocess_b3db.py --seed 0
```

---

### Issue 2: Data File Not Found

**Error:**
```
FileNotFoundError: data/raw/B3DB_classification.tsv
```

**Solution:**
```bash
# Check if data exists
ls -la data/raw/

# If missing, upload from local machine
scp local/path/to/B3DB_classification.tsv user@cfff:/cfff/work/bbb_project/data/raw/
```

---

### Issue 3: Conda Environment Not Found

**Error:**
```
conda: environment 'bbb-baseline' does not exist
```

**Solution:**
```bash
# Create environment
conda env create -f configs/environment-baseline.yml

# Or manually
conda create -n bbb-baseline python=3.10
conda activate bbb-baseline
conda install -c conda-forge rdkit scikit-learn xgboost lightgbm pandas numpy
```

---

### Issue 4: OutOfMemory Error

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```bash
# Reduce batch size or use smaller feature set
python scripts/baseline/02_compute_features.py --seed 0 --feature descriptors_basic
python scripts/baseline/03_train_baselines.py --seed 0 --feature descriptors_basic --models rf
```

---

## Performance Expectations

### Single Experiment (Seed 0, Morgan, RF)

| Step | Runtime | Memory |
|------|---------|--------|
| Preprocessing | 30 sec | 1 GB |
| Feature computation | 30 sec | 2 GB |
| Training | 2 min | 2 GB |
| **Total** | **3 min** | **2 GB** |

---

### Full Benchmark (18 Experiments)

| Configuration | Runtime | Memory |
|--------------|---------|--------|
| 3 seeds × 1 split × 2 features × 3 models | 30-60 min | 4 GB |

---

## Success Criteria

### Environment Setup

- ✅ Conda environment created
- ✅ All dependencies installed
- ✅ Data files present

### Single Experiment

- ✅ Preprocessing completes (3 CSV files created)
- ✅ Feature computation completes (3 NPZ files created)
- ✅ Training completes (model.joblib created)
- ✅ Test AUC > 0.90

### Full Benchmark

- ✅ 18 experiments complete
- ✅ Benchmark summary generated
- ✅ Best baseline: RF + Morgan, AUC > 0.94

---

## Quick Reference

### DSW Setup Commands

```bash
# Clone
git clone <repo-url> && cd bbb_project

# Environment
conda env create -f configs/environment-baseline.yml
conda activate bbb-baseline

# Verify
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"
```

### Batch Job Commands

```bash
# Run benchmark
python scripts/analysis/run_baseline_matrix.py

# Aggregate
python scripts/analysis/aggregate_results.py

# Summary
python scripts/analysis/generate_benchmark_summary.py
```

### Verify Results

```bash
# Check outputs
ls -la data/splits/seed_0/scaffold/
ls -la artifacts/models/baselines/seed_0/scaffold/morgan/rf/
cat artifacts/reports/benchmark_summary.csv
```

---

## Summary

✅ **DSW Phase (Interactive)**
1. Clone repository
2. Create baseline environment
3. Verify single experiment
4. Verify benchmark summary

⏳ **Batch Phase (Non-Interactive)**
1. Create batch script
2. Submit full benchmark job
3. Monitor results
4. Collect outputs

**Estimated Time:**
- DSW setup: 15 minutes
- Single experiment: 3 minutes
- Full benchmark: 30-60 minutes

**Estimated Space:**
- Git repository: ~50 MB
- Data files: ~3 MB
- Generated outputs: ~1 GB

---

**Last Updated:** 2025-03-27
**Status:** Ready for CFFF migration ✅
**Baseline Version:** RF + Morgan (0.9401 ± 0.0454 AUC)
