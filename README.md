# BBB Permeability Prediction

**Machine learning pipeline for predicting blood-brain barrier (BBB) permeability** using the B3DB classification dataset.

Modular baseline pipeline with classical ML models (Random Forest, XGBoost, LightGBM) for binary classification (BBB+ vs BBB-).

---

## Quick Start

### 1. Install Dependencies

```bash
# Create conda environment
conda env create -f configs/environment.yml
conda activate bbb

# Or install with pip
pip install -r requirements.txt
```

### 2. Run Baseline Pipeline

```bash
# Step 1: Preprocess data (scaffold split)
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"

# Step 2: Compute features
python scripts/baseline/02_compute_features.py --seed 0 --feature morgan

# Step 3: Train models
python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models rf,xgb,lgbm
```

### 3. Run Full Benchmark Matrix

```bash
# Run 18 experiments: 3 seeds × 1 split × 2 features × 3 models
python scripts/analysis/run_baseline_matrix.py

# Aggregate results
python scripts/analysis/aggregate_results.py

# Generate benchmark summary
python scripts/analysis/generate_benchmark_summary.py
```

**Output:** `artifacts/reports/benchmark_summary.csv`

---

## Baseline Performance

**Established baseline (B3DB Groups A+B, scaffold split, 3 seeds):**

| Rank | Feature | Model | Test AUC | Test F1 | Seeds |
|------|---------|-------|----------|---------|-------|
| 🥇 | Morgan | **Random Forest** | **0.9401 ± 0.0454** | 0.9391 ± 0.0270 | 3 |
| 2 | Morgan | XGBoost | 0.9198 ± 0.0674 | 0.9335 ± 0.0310 | 3 |
| 3 | Morgan | LightGBM | 0.9195 ± 0.0619 | 0.9363 ± 0.0331 | 3 |
| 4 | Descriptors (basic) | Random Forest | 0.9159 ± 0.0730 | 0.9397 ± 0.0270 | 3 |
| 5 | Descriptors (basic) | XGBoost | 0.9115 ± 0.0712 | 0.9370 ± 0.0308 | 3 |
| 6 | Descriptors (basic) | LightGBM | 0.9114 ± 0.0815 | 0.9395 ± 0.0298 | 3 |

**Best baseline:** Random Forest + Morgan fingerprints (ECFP4, 2048 bits)

**Key findings:**
- Morgan fingerprints outperform basic descriptors by ~1.4% AUC
- Random Forest is most stable (lowest std across seeds)
- All models show moderate overfitting (train AUC ~0.999, test AUC ~0.91-0.94)

---

## Pipeline Overview

### Working Baseline Pipeline

**3-step modular pipeline:**

```
scripts/baseline/
├── 01_preprocess_b3db.py    # Load B3DB, scaffold split
├── 02_compute_features.py   # Compute features (fingerprints/descriptors)
└── 03_train_baselines.py    # Train RF/XGB/LGBM models

scripts/analysis/
├── aggregate_results.py         # Aggregate all experiment results
├── generate_benchmark_summary.py # Generate benchmark summary
└── run_baseline_matrix.py       # Run full experiment matrix
```

**Source modules:**

```
src/
├── config.py              # Centralized configuration (frozen dataclasses)
├── data/                  # Data loading and preprocessing
│   ├── preprocessing.py   # B3DBPreprocessor class
│   ├── scaffold_split.py  # Scaffold-based splitting
│   └── dataset.py         # Dataset classes
├── features/              # Feature extraction
│   ├── fingerprints.py    # Morgan, MACCS, AtomPairs, FP2
│   ├── descriptors.py     # Physicochemical descriptors
│   └── graph.py          # PyTorch Geometric graphs (future GNN)
├── models/                # Model wrappers
│   ├── baseline_models.py # RF, XGB, LGBM, SVM, KNN, etc.
│   └── model_factory.py   # Model factory
├── train/                 # Training utilities
│   └── trainer.py        # Trainer class
├── evaluate/              # Evaluation utilities
│   ├── comparison.py     # Model comparison
│   └── report.py         # Report generation
└── utils/                 # Utilities
    ├── io.py             # File I/O
    ├── metrics.py        # Evaluation metrics
    ├── plotting.py       # Plotting utilities
    ├── seed.py           # Reproducibility
    └── split.py          # Data splitting
```

### Preserved Research Modules

**Archived legacy code (not baseline):**

```
archive/
├── old_scripts/          # Old numbered scripts (01-12*.py)
│   ├── numbered_scripts/ # Superseded by scripts/baseline/
│   ├── mechanism/        # Mechanism prediction scripts
│   ├── analysis/         # Old visualization scripts
│   └── visualization/    # Visualization scripts
├── old_src/              # Old source modules
│   ├── baseline/         # Old baseline implementation
│   ├── featurize/        # Old feature extraction
│   ├── vae/             # VAE generation models
│   ├── gan/             # GAN generation models
│   └── generation/       # Generation pipeline
└── old_web/              # Streamlit web interface
    └── pages/            # Streamlit pages
```

**Future research modules (preserved in src/):**

```
src/
├── pretrain/             # ZINC22 pre-training (future)
├── transformer/          # Transformer models (future)
├── explain/              # Interpretability research
├── vae/                  # VAE generation models
├── gan/                  # GAN generation models
└── path_prediction/      # Transport mechanism prediction
```

---

## Usage Examples

### Run Single Experiment

```bash
# Preprocess data with scaffold split (80:10:10)
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"

# Compute Morgan fingerprints
python scripts/baseline/02_compute_features.py --seed 0 --feature morgan

# Train Random Forest
python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models rf
```

**Output:**
- `data/splits/seed_0/scaffold/train.csv`, `val.csv`, `test.csv`
- `artifacts/features/seed_0/scaffold/morgan/X_train.npz`, etc.
- `artifacts/models/baselines/seed_0/scaffold/morgan/rf/model.joblib`
- `artifacts/models/baselines/seed_0/scaffold/morgan/rf/comparison.json`

### Run Full Benchmark

```bash
# Run experiment matrix (3 seeds × 2 features × 3 models = 18 experiments)
python scripts/analysis/run_baseline_matrix.py

# Check results
cat artifacts/reports/benchmark_summary.csv
```

### Custom Experiments

```bash
# Different feature types
python scripts/baseline/02_compute_features.py --seed 0 --feature descriptors_basic

# Different models
python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models xgb,lgbm

# Multiple seeds
for seed in 0 1 2 3 4; do
    python scripts/baseline/01_preprocess_b3db.py --seed $seed --groups "A,B"
    python scripts/baseline/02_compute_features.py --seed $seed --feature morgan
    python scripts/baseline/03_train_baselines.py --seed $seed --feature morgan --models rf
done
```

---

## Dataset Information

**B3DB (Blood-Brain Barrier Database):**

| Groups | Samples | BBB+ Rate | Use Case |
|--------|---------|-----------|----------|
| A | 846 | 87.7% | High precision, low FP |
| **A,B** | **3,743** | **76.5%** | **Best balance** ⭐ |
| A,B,C | 6,203 | 66.7% | Large scale |
| A,B,C,D | 6,244 | 63.5% | Maximum coverage |

**Default:** Groups A,B (balance between quality and quantity)

**Scaffold split:** 80:10:10 (train:val:test)
- More realistic than random split
- Tests generalization to new scaffolds
- Recommended for BBB prediction

---

## Local Development + CFFF Execution

### Local Development

```bash
# 1. Clone repository
git clone <repository-url>
cd bbb_project

# 2. Create environment
conda env create -f configs/environment.yml
conda activate bbb

# 3. Run baseline test
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"
python scripts/baseline/02_compute_features.py --seed 0 --feature morgan
python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models rf

# 4. Verify results
cat artifacts/models/baselines/seed_0/scaffold/morgan/rf/comparison.json
```

### CFFF Deployment

```bash
# 1. Transfer repository to CFFF
rsync -av --exclude='data/splits/' \
          --exclude='artifacts/features/' \
          --exclude='artifacts/models/' \
          bbb_project/ user@cfff:/path/to/project/

# 2. On CFFF, install dependencies
conda env create -f configs/environment.yml
conda activate bbb

# 3. Verify data files
ls -lh data/raw/B3DB_classification.tsv  # Should be 2.6 MB
ls -lh data/raw/B3DB_regression.tsv      # Should be 413 KB

# 4. Run baseline test
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"
python scripts/baseline/02_compute_features.py --seed 0 --feature morgan
python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models rf

# 5. Run full benchmark (optional)
python scripts/analysis/run_baseline_matrix.py
```

**Notes:**
- Generated outputs (splits, features, models) are in `.gitignore`
- Only source code and raw data are tracked in Git
- CFFF will regenerate all outputs on first run

---

## Configuration

**Centralized configuration in `src/config.py`:**

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path = Path.cwd()
    data: Path = root / "data"
    artifacts: Path = root / "artifacts"
    # ...

@dataclass(frozen=True)
class DatasetConfig:
    filename: str = "B3DB_classification.tsv"
    groups: str = "A,B"  # Default: Groups A,B
    # ...

@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    # ...
```

**Benefits:** Immutable, type-safe, centralized configuration.

---

## Feature Types

| Feature | Dimension | Description |
|---------|-----------|-------------|
| **morgan** | 2048 | ECFP4-like fingerprints (radius=2) ⭐ |
| **maccs** | 167 | MACCS keys |
| **atom_pairs** | 1024 | Atom pair fingerprints |
| **fp2** | 2048 | Daylight fingerprints |
| **descriptors_basic** | 13 | Basic physicochemical properties |
| **descriptors_extended** | ~30 | Extended physicochemical properties |
| **descriptors_all** | ~45 | All physicochemical properties |
| **graph** | - | PyTorch Geometric graphs (for GNN) |

**Default:** Morgan fingerprints (best performance)

---

## Model Types

Supported baseline models:

| Model | Class | Description |
|-------|-------|-------------|
| **rf** | RandomForestClassifier | Random Forest ⭐ |
| **xgb** | XGBClassifier | XGBoost |
| **lgbm** | LGBMClassifier | LightGBM |
| svm | SVC | Support Vector Machine |
| knn | KNeighborsClassifier | K-Nearest Neighbors |
| lr | LogisticRegression | Logistic Regression |
| nb | GaussianNB | Naive Bayes |
| gb | GradientBoostingClassifier | Gradient Boosting |
| ada | AdaBoostClassifier | AdaBoost |

**Default:** Random Forest (best performance)

---

## Documentation

- **`docs/BASELINE_BENCHMARK.md`** - Comprehensive benchmark documentation
- **`docs/QUICK_START_EXPERIMENTS.md`** - Experiment quick reference
- **`docs/RESULTS_TRACKING.md`** - Results tracking guide
- **`docs/PROJECT_CONTEXT.md`** - Project context and roadmap
- **`docs/NEW_STRUCTURE.md`** - New modular structure explanation
- **`CLAUDE.md`** - Comprehensive documentation for AI assistants

---

## Common Issues

### Module import error

```bash
# Error: ModuleNotFoundError: No module named 'src'
# Solution: Scripts automatically add project root to sys.path
# If issues persist, run from project root:
cd /path/to/bbb_project
python scripts/baseline/01_preprocess_b3db.py --seed 0
```

### Missing splits

```bash
# Error: data/splits/seed_0/scaffold/test.csv not found
# Solution: Run preprocessing first
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups "A,B"
```

### Feature dimension mismatch

```bash
# Error: X has 5326 features, but expecting 5287
# Solution: Use consistent feature type
python scripts/baseline/02_compute_features.py --seed 0 --feature morgan
python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models rf
```

### LightGBM type error

```bash
# Error: Expected np.float32
# Solution: Features automatically converted to float32
# If issues persist, check feature generator output
```

---

## Citation

If you use this code or data, please cite the B3DB database:

```bibtex
@article{b3db2021,
  title={B3DB: the blood-brain barrier database for computational brain drug delivery},
  author={...},
  journal={...},
  year={2021}
}
```

---

## License

This project is for research and educational purposes.

---

## Acknowledgments

- **B3DB Database**: Blood-Brain Barrier Database for computational brain drug delivery
- **RDKit**: Open-source cheminformatics
- **scikit-learn**: Machine learning library
- **XGBoost / LightGBM**: Gradient boosting frameworks

---

**Last Updated:** 2025-03-27
**Project Status:** Baseline established ✅
**Python Version:** 3.10+
**Best Baseline:** Random Forest + Morgan (AUC: 0.9401 ± 0.0454)
