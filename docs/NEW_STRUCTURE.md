# BBB Permeability Prediction Project - Restructured Codebase

## Overview

This document describes the reorganized modular structure of the BBB permeability prediction project. The new structure is designed for long-term research and execution on the CFFF platform.

## Project Structure

```
bbb_project/
├── src/                          # Source code modules
│   ├── data/                     # Data preprocessing
│   │   ���── preprocessing.py      # B3DB data loading and cleaning
│   │   ├── scaffold_split.py     # Scaffold-based splitting
│   │   └── dataset.py            # Dataset container classes
│   │
│   ├── features/                 # Feature extraction
│   │   ├── fingerprints.py       # Molecular fingerprints (ECFP, MACCS, etc.)
│   │   ├── descriptors.py        # Physicochemical descriptors
│   │   └── graph.py              # Graph representations for GNNs
│   │
│   ├── models/                   # Model definitions
│   │   ├── baseline_models.py    # Classical ML models (RF, XGB, etc.)
│   │   └── model_factory.py      # Factory for creating models
│   │
│   ├── train/                    # Training logic
│   │   └── trainer.py            # Model training and evaluation
│   │
│   ├── evaluate/                 # Evaluation utilities
│   │   ├── comparison.py         # Model comparison
│   │   └── report.py             # Report generation
│   │
│   └── utils/                    # Utility functions
│       ├── io.py                 # I/O utilities
│       ├── metrics.py            # Evaluation metrics
│       ├── plotting.py           # Plotting utilities
│       ├── seed.py               # Random seed handling
│       └── split.py              # Data splitting utilities
│
├── scripts/                      # Entry point scripts
│   └── baseline/                 # Baseline experiment scripts
│       ├── 01_preprocess_b3db.py # Data preprocessing
│       ├── 02_compute_features.py # Feature computation
│       └── 03_train_baselines.py  # Model training
│
├── data/                         # Data directory
│   ├── raw/                      # Raw datasets (B3DB)
│   ├── splits/                   # Preprocessed splits
│   └── external/                 # External datasets
│
├── artifacts/                    # Generated artifacts
│   ├── features/                 # Computed features
│   ├── models/                   # Trained models
│   ├── metrics/                  # Evaluation metrics
│   └── reports/                  # Analysis reports
│
├── configs/                      # Configuration files
├── docs/                         # Documentation
├── src/config.py                 # Global configuration
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview
```

## Module Responsibilities

### `src/data/` - Data Preprocessing

**Classes:**
- `B3DBPreprocessor`: Load, clean, and canonicalize B3DB datasets
- `B3DBDataset`: Container for train/val/test splits
- `ProcessedData`: Data container with metadata

**Functions:**
- `scaffold_split()`: Perform scaffold-based splitting (recommended)
- `random_split()`: Perform random stratified splitting (baseline)

**Key Features:**
- SMILES canonicalization and validation
- Duplicate removal
- Scaffold computation for structural splitting
- Support for both classification and regression tasks

### `src/features/` - Feature Extraction

**Classes:**
- `FingerprintGenerator`: Generate molecular fingerprints
  - Morgan/ECFP (default: 2048 bits, radius 2)
  - MACCS keys (167 bits)
  - Atom pairs (1024 bits)
  - FP2/RDKit (2048 bits)
  - Combined fingerprints

- `DescriptorGenerator`: Compute physicochemical descriptors
  - Basic: 13 core descriptors
  - Extended: ~100 descriptors
  - All: 200+ descriptors

- `GraphGenerator`: Generate graph representations for GNNs
  - Node features: atom type, degree, hybridization, etc.
  - Edge features: bond type, conjugation, ring status

### `src/models/` - Model Definitions

**Classes:**
- `BaselineModel`: Wrapper for classical ML models
  - Supported: RF, XGB, LightGBM, SVM, KNN, LR, NB, GB, ADA, ETC

- `ModelConfig`: Configuration for model hyperparameters

- `ModelFactory`: Factory for creating model instances

**Key Features:**
- Unified interface for all model types
- Automatic handling of sparse/dense features
- Model saving/loading

### `src/train/` - Training Logic

**Classes:**
- `Trainer`: Train and evaluate models
- `TrainingConfig`: Training configuration
- `TrainingResult`: Container for training results

**Functions:**
- `train_multiple_models()`: Train multiple models in parallel

**Key Features:**
- Automatic metric computation (AUC, accuracy, F1, etc.)
- Model checkpointing
- Prediction saving

### `src/evaluate/` - Evaluation

**Classes:**
- `ModelComparison`: Compare multiple models
- Methods for sorting, summarizing, and saving results

**Functions:**
- `compare_models()`: Create comparison from results
- `generate_report()`: Generate CSV and text reports

## Quick Start

### 1. Preprocess B3DB Data

```bash
# Classification with scaffold split (recommended)
python scripts/baseline/01_preprocess_b3db.py \
    --seed 0 \
    --groups A,B \
    --split_type scaffold \
    --task classification

# Regression with random split
python scripts/baseline/01_preprocess_b3db.py \
    --seed 0 \
    --groups A,B \
    --split_type random \
    --task regression
```

**Output:**
- `data/splits/seed_0/classification_scaffold/`
  - `train.csv`, `val.csv`, `test.csv`
  - `statistics.json`

### 2. Compute Features

```bash
# Morgan fingerprints (ECFP4)
python scripts/baseline/02_compute_features.py \
    --seed 0 \
    --split scaffold \
    --feature morgan

# Combined fingerprints
python scripts/baseline/02_compute_features.py \
    --seed 0 \
    --split scaffold \
    --feature combined

# All descriptors
python scripts/baseline/02_compute_features.py \
    --seed 0 \
    --split scaffold \
    --feature descriptors_all
```

**Output:**
- `artifacts/features/seed_0/scaffold/morgan/`
  - `X_train.npy`, `X_val.npy`, `X_test.npy`
  - `y_train.npy`, `y_val.npy`, `y_test.npy`
  - `metadata.json`

### 3. Train Baseline Models

```bash
# Train RF, XGB, LightGBM
python scripts/baseline/03_train_baselines.py \
    --seed 0 \
    --split scaffold \
    --feature morgan \
    --models rf,xgb,lgbm

# Train all models
python scripts/baseline/03_train_baselines.py \
    --seed 0 \
    --split scaffold \
    --feature morgan \
    --models rf,xgb,lgbm,svm,knn,lr,nb,gb,ada,etc
```

**Output:**
- `artifacts/models/baselines/seed_0/scaffold/morgan/`
  - `rf/`: Trained model and predictions
  - `xgb/`: Trained model and predictions
  - `comparison.json`: All results
  - `reports/`: Summary tables and best model info

## Recommended Workflow

### Phase 1: Baseline Experiments (Current Priority)

1. **Data Preprocessing**
   ```bash
   python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups A,B --split_type scaffold
   ```

2. **Feature Computation**
   ```bash
   # Test different feature types
   for feat in morgan maccs atom_pairs fp2 combined; do
       python scripts/baseline/02_compute_features.py --seed 0 --split scaffold --feature $feat
   done
   ```

3. **Model Training**
   ```bash
   # Train core models on each feature type
   for feat in morgan maccs combined; do
       python scripts/baseline/03_train_baselines.py --seed 0 --split scaffold --feature $feat --models rf,xgb,lgbm
   done
   ```

4. **Analysis**
   - Check `artifacts/models/baselines/seed_0/scaffold/*/reports/results_summary.csv`
   - Identify best model-feature combinations

### Phase 2: GNN Models (Future)

- Use `src/features/graph.py` to generate graph data
- Implement GNN models (GAT, GCN)
- Train and evaluate

### Phase 3: Advanced Features (Future)

- ZINC22 pretraining
- Ensemble methods
- Interpretability analysis

## Configuration

Global configuration is in `src/config.py` using frozen dataclasses:

- `Paths`: Directory paths
- `DatasetConfig`: B3DB column names and filters
- `SplitConfig`: Train/val/test ratios (default: 80:10:10)
- `FingerprintConfig`: Fingerprint parameters
- `DescriptorConfig`: Descriptor set selection

## Running on CFFF

To run on CFFF platform:

1. **Ensure data is available:**
   - Copy `data/raw/B3DB_classification.tsv` and `B3DB_regression.tsv` to CFFF
   - Or download from source if available

2. **Run scripts sequentially:**
   ```bash
   # Step 1: Preprocess
   python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups A,B --split_type scaffold

   # Step 2: Features
   python scripts/baseline/02_compute_features.py --seed 0 --split scaffold --feature morgan

   # Step 3: Train
   python scripts/baseline/03_train_baselines.py --seed 0 --split scaffold --feature morgan --models rf,xgb,lgbm
   ```

3. **Collect results:**
   - Models and reports are in `artifacts/`
   - Download `artifacts/models/baselines/seed_0/scaffold/morgan/reports/`

## Key Design Decisions

### 1. Scaffold Split as Default
- Ensures structural diversity between splits
- More realistic evaluation of generalization
- Recommended by recent BBBP literature

### 2. Groups A,B as Default
- Balances dataset size (3,743 samples) with data quality
- 76.5% BBB+ rate
- Best trade-off for initial experiments

### 3. Separate Classification and Regression
- Different loss functions and metrics
- Can use different splits if needed
- Clear separation of concerns

### 4. Modular Design
- Easy to add new models, features, or metrics
- Clear separation between data, features, models, training
- Suitable for long-term research

## Assumptions and Uncertainties

### Assumptions:
1. B3DB datasets are available in `data/raw/`
2. Scaffold split is preferred over random split for evaluation
3. Groups A,B provide sufficient data for initial experiments
4. Morgan fingerprints will be the primary feature type

### Uncertainties:
1. **ZINC22 integration**: Not implemented yet - size and format need to be determined
2. **GNN implementation**: Graph data pipeline needs testing with actual GNN models
3. **CFFF environment**: Exact paths and resource constraints may require adjustments
4. **Regression task**: Less priority than classification - may need different features

## Next Steps

1. ✅ Complete modular structure
2. ✅ Create baseline experiment scripts
3. ⏳ Test scripts on local machine
4. ⏳ Run initial experiments on B3DB
5. ⏳ Compare scaffold vs random splits
6. ⏳ Add GNN models when baseline is stable
7. ⏳ Plan ZINC22 pretraining integration

## Legacy Code

The following directories contain legacy code that is preserved but not actively used:

- `src/baseline/`: Old baseline training code
- `src/featurize/`: Old feature extraction code
- `src/finetune/`: GNN fine-tuning code
- `src/pretrain/`: GNN pre-training code
- `src/transformer/`: Transformer models
- `src/vae/`, `src/gan/`: Generation models
- `src/explain/`: Interpretability code
- `scripts/*_backup/`: Backup scripts

These can be referenced but are not part of the new modular workflow.
