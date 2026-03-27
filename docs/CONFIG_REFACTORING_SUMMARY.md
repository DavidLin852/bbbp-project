# Configuration Refactoring Summary

**Date:** 2025-03-27
**Action:** Refactored monolithic `src/config.py` into modular structure

---

## New Configuration Layout

### Before (Monolithic)

```
src/config.py (257 lines)
├── Paths                      # File system paths
├── DatasetConfig              # B3DB dataset configuration
├── SplitConfig                # Train/val/test split ratios
├── FeaturizeConfig            # (Deprecated) Feature extraction
├── FingerprintConfig          # Fingerprint parameters
├── DescriptorConfig           # Descriptor parameters
├── FeatureConfig              # Combined feature config
├── TransformerConfig         # ❌ Research (not baseline)
├── StackingConfig            # ❌ Research (not baseline)
├── SHAPConfig                # ❌ Research (not baseline)
├── VAEConfig                 # ❌ Research (not baseline)
├── VAETrainConfig            # ❌ Research (not baseline)
├── GANConfig                 # ❌ Research (not baseline)
├── GANTrainConfig            # ❌ Research (not baseline)
└── GenerationConfig          # ❌ Research (not baseline)
```

### After (Modular)

```
src/config/
├── __init__.py                # Unified interface
├── paths.py                   # File system paths (shared)
├── baseline.py                # Baseline pipeline configs
│   ├── DatasetConfig
│   ├── SplitConfig
│   ├── FingerprintConfig
│   ├── DescriptorConfig
│   ���── FeatureConfig
└── research.py                # Research module configs
    ├── TransformerConfig
    ├── VAEConfig
    ├── VAETrainConfig
    ├── GANConfig
    ├── GANTrainConfig
    ├── GenerationConfig
    ├── StackingConfig
    └── SHAPConfig

src/config.py                  # Backward compatibility (re-exports)
```

---

## Configuration Modules

### 1. `src/config/paths.py` - Shared Paths

**Purpose:** File system paths used by both baseline and research code

**Classes:**
- `Paths` - All project paths (data, artifacts, models, reports, etc.)

**Usage:**
```python
from src.config.paths import Paths

paths = Paths()
print(paths.data_raw)        # E:\PythonProjects\bbb_project\data\raw
print(paths.data_splits)     # E:\PythonProjects\bbb_project\data\splits
print(paths.artifacts)       # E:\PythonProjects\bbb_project\artifacts
print(paths.models)          # E:\PythonProjects\bbb_project\artifacts\models
print(paths.reports)         # E:\PythonProjects\bbb_project\artifacts\reports
```

**Who should use this:**
- ✅ All code (baseline and research)
- ✅ CFFF deployment
- ✅ New code (recommended)

---

### 2. `src/config/baseline.py` - Baseline Pipeline

**Purpose:** Configuration for the working baseline pipeline

**Classes:**
- `DatasetConfig` - B3DB dataset loading and filtering
- `SplitConfig` - Train/val/test split ratios
- `FingerprintConfig` - Molecular fingerprint parameters
- `DescriptorConfig` - Physicochemical descriptor parameters
- `FeatureConfig` - Combined feature configuration

**Usage:**
```python
from src.config.baseline import DatasetConfig, SplitConfig, FeatureConfig

# Use defaults
dataset = DatasetConfig()
split = SplitConfig()
features = FeatureConfig()

# Or customize
dataset = DatasetConfig(group_keep=("A", "B", "C"))
split = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
features = FeatureConfig(use_morgan=True, use_descriptors=True)
```

**Who should use this:**
- ✅ Baseline pipeline scripts (`scripts/baseline/*.py`)
- ✅ Baseline analysis scripts (`scripts/analysis/*.py`)
- ✅ New baseline experiments
- ✅ CFFF deployment

**Default values:**
- `DatasetConfig.group_keep`: `("A", "B")` (best balance)
- `SplitConfig.train_ratio`: `0.8` (80% training)
- `FeatureConfig.use_morgan`: `True` (best performance)

---

### 3. `src/config/research.py` - Research Modules

**Purpose:** Configuration for experimental features (future work)

**Classes:**
- `TransformerConfig` - Transformer models (MolBERT, Graphormer)
- `VAEConfig`, `VAETrainConfig` - VAE molecule generation
- `GANConfig`, `GANTrainConfig` - GAN molecule generation
- `GenerationConfig` - Generation pipeline
- `StackingConfig` - Ensemble methods
- `SHAPConfig` - Interpretability analysis

**Usage:**
```python
from src.config.research import VAEConfig, GANConfig

# VAE generation
vae_config = VAEConfig(
    latent_dim=256,
    beta=0.001,
    min_bbb_prob=0.7
)

# GAN generation
gan_config = GANConfig(
    latent_dim=256,
    n_critic=5,
    reward_bbb=1.0
)
```

**Who should use this:**
- ❌ NOT baseline pipeline
- ⚠️ Research modules only (future work)
- ⚠️ Experimental features

**WARNING:** These configs are NOT used in the baseline pipeline. They are preserved for future research.

---

### 4. `src/config/__init__.py` - Unified Interface

**Purpose:** Provides a unified import interface for the entire project

**Usage:**
```python
# Import from unified interface
from src.config import Paths, DatasetConfig, SplitConfig, FeatureConfig

# This is equivalent to:
# from src.config.paths import Paths
# from src.config.baseline import DatasetConfig, SplitConfig, FeatureConfig
```

**Who should use this:**
- ✅ Existing code (backward compatibility)
- ✅ New code (convenience)
- ✅ CFFF deployment

---

### 5. `src/config.py` - Backward Compatibility

**Purpose:** Maintains backward compatibility with existing imports

**What it does:**
- Re-exports all classes from new modular structure
- Provides deprecated alias (`FeaturizeConfig` → `FeatureConfig`)

**Usage:**
```python
# Old imports still work
from src.config import Paths, DatasetConfig, SplitConfig, FeatureConfig

# Deprecated alias also works
from src.config import FeaturizeConfig  # Alias for FeatureConfig
```

**Who should use this:**
- ✅ Existing code (no changes needed)
- ⚠️ New code should use specific modules instead

---

## How Existing Scripts Still Work

### Backward Compatibility

All existing imports continue to work without changes:

```python
# scripts/baseline/01_preprocess_b3db.py
from src.config import Paths  # ✅ Still works

# src/data/preprocessing.py
from src.config import Paths  # ✅ Still works

# Any script
from src.config import DatasetConfig, SplitConfig  # ✅ Still works
```

### Why It Works

The `src/config.py` file now acts as a **compatibility shim**:

1. **Re-exports** all classes from new modules
2. **Preserves** all existing import paths
3. **Adds** deprecated aliases for renamed classes

**No code changes needed** for existing scripts!

---

## Which Config Modules to Use

### For Baseline Work (Current Pipeline)

**Recommended imports:**
```python
# Option 1: Import from unified interface (convenient)
from src.config import Paths, DatasetConfig, SplitConfig, FeatureConfig

# Option 2: Import from specific modules (explicit)
from src.config.paths import Paths
from src.config.baseline import DatasetConfig, SplitConfig, FeatureConfig
```

**Use cases:**
- ✅ Running baseline experiments
- ✅ Training baseline models (RF, XGB, LGBM)
- ✅ Generating benchmark results
- ✅ CFFF deployment

**Scripts that use these:**
- `scripts/baseline/01_preprocess_b3db.py`
- `scripts/baseline/02_compute_features.py`
- `scripts/baseline/03_train_baselines.py`
- `scripts/analysis/run_baseline_matrix.py`

---

### For Research Work (Future Modules)

**Recommended imports:**
```python
# Import from research module
from src.config.research import VAEConfig, GANConfig, GenerationConfig
from src.config.paths import Paths  # Paths are shared
```

**Use cases:**
- ⚠️ VAE/GAN molecule generation (future)
- ⚠️ Transformer models (future)
- ⚠️ Ensemble methods (future)
- ⚠️ Interpretability analysis (future)

**Scripts that might use these:**
- `src/vae/train_vae.py` (preserved, not baseline)
- `src/gan/train_molgan.py` (preserved, not baseline)
- `src/pretrain/zinc20_pretrain.py` (preserved, not baseline)

---

### For New Code

**Recommended approach:**

1. **For baseline pipeline:**
```python
from src.config.paths import Paths
from src.config.baseline import DatasetConfig, SplitConfig, FeatureConfig
```

2. **For research modules:**
```python
from src.config.paths import Paths
from src.config.research import VAEConfig, GANConfig, etc.
```

3. **For convenience (both work):**
```python
from src.config import Paths, DatasetConfig, SplitConfig
```

---

## Benefits of Refactoring

### 1. Separation of Concerns
- ✅ Baseline config separated from research config
- ✅ Clear distinction between working and experimental features
- ✅ Easier to find relevant configuration

### 2. Maintainability
- ✅ Smaller, focused modules (paths.py, baseline.py, research.py)
- ✅ Easier to update baseline config without affecting research
- ✅ Easier to add new research configs

### 3. CFFF-Friendly
- ✅ Baseline config is self-contained
- ✅ Research config can be ignored for deployment
- ✅ Clear which configs are needed for production

### 4. Backward Compatibility
- ✅ All existing imports still work
- ✅ No code changes needed
- ✅ Gradual migration path

---

## Migration Guide for New Code

### Recommended Pattern

```python
# ===== FOR BASELINE WORK =====
from src.config.paths import Paths
from src.config.baseline import DatasetConfig, SplitConfig, FeatureConfig

# Use defaults
paths = Paths()
dataset = DatasetConfig()  # Groups A,B by default
split = SplitConfig()  # 80:10:10 by default
features = FeatureConfig()  # Morgan + descriptors

# ===== FOR RESEARCH WORK =====
from src.config.paths import Paths
from src.config.research import VAEConfig, GANConfig

paths = Paths()
vae = VAEConfig(latent_dim=256, beta=0.001)
gan = GANConfig(latent_dim=256, n_critic=5)
```

### Legacy Pattern (Still Works)

```python
# Old way (still works for backward compatibility)
from src.config import Paths, DatasetConfig, SplitConfig

paths = Paths()
dataset = DatasetConfig()
split = SplitConfig()
```

---

## File Structure

```
src/config/
├── __init__.py          # 42 lines - Unified interface
├── paths.py             # 43 lines - File system paths
├── baseline.py          # 145 lines - Baseline pipeline config
├── research.py          # 256 lines - Research module config
└── (legacy) config.py   # 65 lines - Backward compatibility

src/config.py            # 65 lines - Re-exports from modular structure
```

**Total lines:** 257 (before) → 576 (after, with extensive documentation)

**Code complexity:** Reduced (monolithic → modular)

---

## Testing

### Verification Tests

1. ✅ **Backward compatible imports work**
```bash
python -c "from src.config import Paths, DatasetConfig, SplitConfig; print('OK')"
```

2. ✅ **Direct imports from new modules work**
```bash
python -c "from src.config.paths import Paths; print('OK')"
python -c "from src.config.baseline import DatasetConfig; print('OK')"
python -c "from src.config.research import VAEConfig; print('OK')"
```

3. ✅ **Baseline scripts still work**
```bash
python scripts/baseline/01_preprocess_b3db.py --help
```

---

## Summary

✅ **Configuration successfully refactored**

**Key improvements:**
1. ✅ Separated baseline config from research config
2. ✅ Modular structure (paths, baseline, research)
3. ✅ Backward compatibility maintained
4. ✅ CFFF-friendly (clear separation)
5. ✅ Extensive documentation in each module

**No breaking changes:**
- ✅ All existing imports still work
- ✅ No code changes needed
- ✅ Baseline pipeline unaffected

**Recommended for future:**
- ✅ Use `from src.config.baseline import ...` for baseline work
- ✅ Use `from src.config.research import ...` for research work
- ✅ Use `from src.config.paths import Paths` for paths (shared)

---

**Last Updated:** 2025-03-27
**Status:** Complete ✅
**Backward Compatibility:** Maintained ✅
**CFFF-Ready:** Yes ✅
