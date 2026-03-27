# Dependency Split Summary

**Date:** 2025-03-27
**Action:** Split dependencies into baseline and research sets

---

## Overview

Split the monolithic `requirements.txt` into separate dependency files for cleaner environment setup and easier CFFF migration.

---

## New Dependency Structure

### Dependency Files

```
requirements.txt                    # Guide (references split files)
requirements-baseline.txt           # Core baseline dependencies ⭐
requirements-research.txt           # Research module dependencies ⚠️
requirements-legacy.txt             # All dependencies (backward compatibility)
```

### Conda Environment Files

```
configs/
├── environment.yml                 # Original (all dependencies)
├── environment-baseline.yml        # Baseline environment ⭐
└── environment-research.yml        # Research environment ⚠️
```

---

## Baseline Dependencies (Core)

**Files:**
- `requirements-baseline.txt`
- `configs/environment-baseline.yml`

**Purpose:** Dependencies for the working baseline pipeline only

**Includes:**
- ✅ Core Python: pandas, numpy, scipy, scikit-learn
- ✅ ML models: xgboost, lightgbm
- ✅ Molecular computation: rdkit
- ✅ Utilities: joblib, tqdm
- ✅ Visualization: matplotlib, seaborn

**Excludes:**
- ❌ Deep learning: torch, torch-geometric, torch-scatter, torch-sparse
- ❌ Web framework: streamlit, plotly

**Size:** ~500 MB (conda install)

**Use for:**
- ✅ Local baseline development
- ✅ CFFF baseline execution
- ✅ Reproducing benchmark results (RF + Morgan: 0.9401 AUC)

**Installation:**
```bash
# With pip
pip install -r requirements-baseline.txt

# With conda (RECOMMENDED)
conda env create -f configs/environment-baseline.yml
conda activate bbb-baseline
```

---

## Research Dependencies (Optional)

**Files:**
- `requirements-research.txt`
- `configs/environment-research.yml`

**Purpose:** Dependencies for experimental and research modules

**Includes:**
- ✅ All baseline dependencies (above)
- ✅ Deep learning: torch, torchvision, torch-geometric, torch-scatter, torch-sparse
- ✅ Web framework: streamlit, plotly

**Size:** ~2-3 GB (with PyTorch CPU) or ~5 GB (with CUDA)

**Use for:**
- ⚠️ VAE/GAN molecule generation
- ⚠️ GNN models (GAT, GCN)
- ⚠️ Transformer models (MolBERT, Graphormer)
- ⚠️ Streamlit web interface
- ⚠️ Advanced research features

**Installation:**
```bash
# With pip
pip install -r requirements-baseline.txt
pip install -r requirements-research.txt

# With conda (RECOMMENDED)
conda env create -f configs/environment-research.yml
conda activate bbb-research
```

**Note:** Research environment includes all baseline dependencies.

---

## Legacy Dependencies (Backward Compatibility)

**Files:**
- `requirements-legacy.txt`
- `configs/environment.yml` (original)

**Purpose:** All dependencies in one file (backward compatibility)

**Includes:**
- ✅ Everything (baseline + research)

**Size:** ~2-3 GB (with PyTorch CPU)

**Use for:**
- ⚠️ Existing setups that don't want to change
- ⚠️ Environments that need all features

**Installation:**
```bash
# With pip
pip install -r requirements-legacy.txt

# With conda
conda env create -f configs/environment.yml
conda activate bbb
```

---

## Dependency Comparison

| Dependency | Baseline | Research | Legacy | Purpose |
|------------|----------|----------|--------|---------|
| **pandas** | ✅ | ✅ | ✅ | Data manipulation |
| **numpy** | ✅ | ✅ | ✅ | Numerical computing |
| **scipy** | ✅ | ✅ | ✅ | Scientific computing |
| **scikit-learn** | ✅ | ✅ | ✅ | ML algorithms |
| **xgboost** | ✅ | ✅ | ✅ | XGBoost model |
| **lightgbm** | ✅ | ✅ | ✅ | LightGBM model |
| **rdkit** | ✅ | ✅ | ✅ | Molecular computation |
| **joblib** | ✅ | ✅ | ✅ | Model persistence |
| **tqdm** | ✅ | ✅ | ✅ | Progress bars |
| **matplotlib** | ✅ | ✅ | ✅ | Plotting |
| **seaborn** | ✅ | ✅ | ✅ | Visualization |
| **torch** | ❌ | ✅ | ✅ | Deep learning |
| **torchvision** | ❌ | ✅ | ✅ | Vision models |
| **torch-geometric** | ❌ | ✅ | ✅ | Graph neural networks |
| **torch-scatter** | ❌ | ✅ | ✅ | GNN operations |
| **torch-sparse** | ❌ | ✅ | ✅ | Sparse operations |
| **streamlit** | ❌ | ✅ | ✅ | Web interface |
| **plotly** | ❌ | ✅ | ✅ | Interactive plots |

---

## Recommended Installation Order

### For Local Baseline Development

**Step 1: Create baseline environment**
```bash
conda env create -f configs/environment-baseline.yml
conda activate bbb-baseline
```

**Step 2: Verify installation**
```bash
python -c "import rdkit; import sklearn; import xgboost; import lightgbm; print('OK')"
```

**Step 3: Run baseline test**
```bash
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"
```

---

### For CFFF Baseline Execution

**Step 1: Transfer repository to CFFF**
```bash
rsync -av --exclude='data/splits/' \
          --exclude='artifacts/features/' \
          --exclude='artifacts/models/' \
          bbb_project/ user@cfff:/path/to/project/
```

**Step 2: On CFFF, create baseline environment**
```bash
conda env create -f configs/environment-baseline.yml
conda activate bbb-baseline
```

**Step 3: Verify baseline pipeline**
```bash
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"
python scripts/baseline/02_compute_features.py --seed 0 --feature morgan
python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models rf
```

**Step 4: Run full benchmark (optional)**
```bash
python scripts/analysis/run_baseline_matrix.py
```

---

### For Future Research Modules

**Step 1: Create research environment**
```bash
conda env create -f configs/environment-research.yml
conda activate bbb-research
```

**Step 2: Verify installation**
```bash
python -c "import torch; import torch_geometric; import streamlit; print('OK')"
```

**Step 3: Run research script**
```bash
# Example: VAE training
python src/vae/train_vae.py

# Example: Streamlit app
streamlit run archive/old_web/app_bbb_predict.py
```

---

## Benefits of Split

### 1. Smaller Baseline Environment

**Before:**
- Single environment with all dependencies
- Size: ~2-3 GB (with PyTorch)
- Installation time: 10-30 minutes

**After:**
- Separate baseline environment
- Size: ~500 MB (no PyTorch)
- Installation time: 2-5 minutes

**Benefits:**
- ✅ Faster installation
- ✅ Smaller disk footprint
- ✅ Fewer dependencies to manage
- ✅ Easier CFFF deployment

---

### 2. Clear Separation of Concerns

**Baseline dependencies:**
- Only what's needed for RF/XGB/LGBM models
- Well-tested, stable versions
- Minimal maintenance burden

**Research dependencies:**
- Deep learning frameworks (PyTorch, PyG)
- Web framework (Streamlit)
- Experimental features
- May require additional setup (CUDA)

**Benefits:**
- ✅ Clear which deps are production-ready
- ✅ Easy to exclude experimental features
- ✅ Better dependency hygiene

---

### 3. CFFF-Friendly

**Baseline deployment:**
- Only baseline dependencies needed
- Smaller environment transfer
- Faster setup on CFFF

**Research work:**
- Can use research environment locally
- Don't need to deploy to CFFF
- Separated from production code

**Benefits:**
- ✅ Cleaner CFFF deployment
- ✅ Faster environment setup
- ✅ Reduced dependency conflicts

---

## File Sizes (Estimated)

| Environment | Size | Installation Time |
|-------------|------|-------------------|
| **Baseline only** | ~500 MB | 2-5 min |
| **Research (CPU)** | ~2-3 GB | 10-20 min |
| **Research (CUDA)** | ~5 GB | 15-30 min |

---

## Dependency Details

### Baseline Dependencies

```
# Core scientific computing (150 MB)
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0

# ML models (50 MB)
xgboost>=2.0.0
lightgbm>=4.0.0

# Molecular computation (200 MB)
rdkit>=2023.3.0

# Utilities (10 MB)
joblib>=1.3.0
tqdm>=4.66.0

# Visualization (100 MB)
matplotlib>=3.8.0
seaborn>=0.13.0

Total: ~500 MB
```

### Research Dependencies (Additional)

```
# Deep learning frameworks (1.5-3 GB)
torch>=2.0.0
torchvision>=0.15.0
torch-geometric>=2.4.0
torch-scatter>=2.1.0
torch-sparse>=0.6.0

# Web framework (100 MB)
streamlit>=1.29.0
plotly>=5.18.0

Additional: ~2-3 GB (CPU) or ~4-5 GB (CUDA)
```

---

## Backward Compatibility

### All Existing Installations Still Work

**Option 1: Use split files (RECOMMENDED)**
```bash
# Before (old way)
pip install -r requirements.txt

# After (new way)
pip install -r requirements-baseline.txt
```

**Option 2: Use legacy file (NO CHANGE)**
```bash
# Still works exactly as before
pip install -r requirements-legacy.txt
```

**Option 3: Use conda (NO CHANGE)**
```bash
# Still works exactly as before
conda env create -f configs/environment.yml
```

---

## Migration Guide

### For Existing Setups

**If you have existing environment with all dependencies:**

**Option 1: Keep using it (no changes needed)**
```bash
conda activate bbb  # Your existing environment
# Everything still works
```

**Option 2: Create new baseline environment (recommended)**
```bash
# Create new lightweight environment
conda env create -f configs/environment-baseline.yml
conda activate bbb-baseline

# Test baseline pipeline
python scripts/baseline/01_preprocess_b3db.py --seed 0
```

**Option 3: Migrate to split environments**
```bash
# Export existing environment
conda env export > environment-full.yml

# Create baseline environment
conda env create -f configs/environment-baseline.yml

# Keep old environment for research work
conda env rename bbb bbb-research
```

---

## Summary

✅ **Dependencies successfully split**

**Key improvements:**
1. ✅ Baseline environment: ~500 MB (down from ~2-3 GB)
2. ✅ Clear separation: core vs experimental
3. ✅ Faster installation: 2-5 min (down from 10-30 min)
4. ✅ CFFF-friendly: smaller transfer, faster setup
5. ✅ Backward compatible: legacy files preserved

**Baseline dependencies (production-ready):**
- `requirements-baseline.txt`
- `configs/environment-baseline.yml`

**Research dependencies (experimental):**
- `requirements-research.txt`
- `configs/environment-research.yml`

**Legacy (backward compatibility):**
- `requirements-legacy.txt`
- `configs/environment.yml` (original)

---

**Last Updated:** 2025-03-27
**Status:** Complete ✅
**Backward Compatibility:** Maintained ✅
**CFFF-Ready:** Yes ✅
