# Codebase Reorganization Summary

## Overview

The BBB permeability prediction project has been reorganized into a cleaner, modular structure suitable for long-term research and execution on the CFFF platform. The reorganization preserves existing useful code while creating a clear separation of concerns.

## Files Created

### New Modules (src/)

#### `src/data/` - Data Preprocessing
- **`__init__.py`**: Module exports
- **`preprocessing.py`**: `B3DBPreprocessor` class for loading, cleaning, and canonicalizing B3DB datasets
- **`scaffold_split.py`**: Scaffold-based and random splitting functions
- **`dataset.py`**: `B3DBDataset` container class for train/val/test splits

#### `src/features/` - Feature Extraction
- **`__init__.py`**: Module exports
- **`fingerprints.py`**: `FingerprintGenerator` class for molecular fingerprints (Morgan, MACCS, AtomPairs, FP2, Combined)
- **`descriptors.py`**: `DescriptorGenerator` class for physicochemical descriptors (basic, extended, all)
- **`graph.py`**: `GraphGenerator` class for PyTorch Geometric graph representations

#### `src/models/` - Model Definitions
- **`__init__.py`**: Module exports
- **`baseline_models.py`**: `BaselineModel` and `ModelConfig` for classical ML models (RF, XGB, LightGBM, SVM, KNN, LR, NB, GB, ADA, ETC)
- **`model_factory.py`**: `ModelFactory` for convenient model creation

#### `src/train/` - Training Logic
- **`__init__.py`**: Module exports
- **`trainer.py`**: `Trainer` class for model training and evaluation, `TrainingResult` container, `train_multiple_models()` function

#### `src/evaluate/` - Evaluation
- **`__init__.py`**: Module exports
- **`comparison.py`**: `ModelComparison` class for comparing multiple models
- **`report.py`**: `generate_report()` function for generating evaluation reports

### New Entry Point Scripts (scripts/)

#### `scripts/baseline/`
- **`01_preprocess_b3db.py`**: Load B3DB data, perform splitting, save train/val/test splits
- **`02_compute_features.py`**: Compute fingerprints or descriptors for preprocessed splits
- **`03_train_baselines.py`**: Train baseline ML models on computed features

### Documentation (docs/)
- **`NEW_STRUCTURE.md`**: Comprehensive documentation of new structure
- **`QUICK_REFERENCE.md`**: Quick reference guide with common commands and examples

## Files Preserved

The following existing files were preserved and not modified:

### Configuration
- `src/config.py`: All configuration dataclasses (Paths, DatasetConfig, etc.)

### Utilities
- `src/utils/io.py`: I/O utilities
- `src/utils/metrics.py`: Evaluation metrics (ClsMetrics, classification_metrics)
- `src/utils/plotting.py`: Plotting utilities
- `src/utils/seed.py`: Random seed handling
- `src/utils/split.py`: Data splitting utilities

### Legacy Code (preserved but not actively used)
- `src/baseline/`: Old baseline training code
- `src/featurize/`: Old feature extraction code (now refactored into src/features/)
- `src/finetune/`, `src/pretrain/`, `src/transformer/`: Deep learning models
- `src/vae/`, `src/gan/`: Generation models
- `src/explain/`: Interpretability code
- `src/path_prediction/`: Transport mechanism prediction
- `scripts/*_backup/`: Backup scripts

## Module Responsibilities

### `src/data/` - Data Preprocessing
- **Responsibility**: Load, clean, and split B3DB datasets
- **Key Classes**: `B3DBPreprocessor`, `B3DBDataset`
- **Key Functions**: `scaffold_split()`, `random_split()`
- **Input**: Raw B3DB TSV files
- **Output**: Cleaned train/val/test CSV files

### `src/features/` - Feature Extraction
- **Responsibility**: Compute molecular representations
- **Key Classes**: `FingerprintGenerator`, `DescriptorGenerator`, `GraphGenerator`
- **Input**: SMILES strings
- **Output**: Feature matrices (numpy arrays or PyG Data objects)

### `src/models/` - Model Definitions
- **Responsibility**: Define and instantiate ML models
- **Key Classes**: `BaselineModel`, `ModelFactory`
- **Input**: Configuration
- **Output**: Configured model instances

### `src/train/` - Training Logic
- **Responsibility**: Train models and compute metrics
- **Key Classes**: `Trainer`, `TrainingResult`
- **Input**: Models, features, labels
- **Output**: Trained models, predictions, metrics

### `src/evaluate/` - Evaluation
- **Responsibility**: Compare models and generate reports
- **Key Classes**: `ModelComparison`
- **Input**: Training results
- **Output**: Comparison tables, reports

## Key Design Decisions

1. **Scaffold Split as Default**: Ensures structural diversity between splits, more realistic evaluation

2. **Groups A,B as Default**: Balances dataset size (3,743 samples) with data quality (76.5% BBB+ rate)

3. **Separate Classification and Regression**: Different tasks with different metrics and requirements

4. **Modular Design**: Easy to add new models, features, or metrics without modifying existing code

5. **Preserved Legacy Code**: Old implementations are preserved for reference but not used in new workflow

## Assumptions

1. B3DB datasets are available in `data/raw/`
2. Scaffold split is preferred over random split for evaluation
3. Groups A,B provide sufficient data for initial experiments
4. Morgan fingerprints will be the primary feature type
5. Classical ML models (RF, XGB, LightGBM) are the baseline

## Uncertainties

1. **ZINC22 Integration**: Not implemented yet - data size and format need to be determined

2. **GNN Implementation**: Graph data pipeline exists in `src/features/graph.py` but hasn't been tested with actual GNN training

3. **CFFF Environment**: Exact paths and resource constraints may require adjustments when deployed

4. **Regression Task**: Lower priority than classification - may need different features and evaluation metrics

## Next Steps

1. **Test New Scripts**: Run the three baseline scripts to verify the workflow
2. **Run Initial Experiments**: Train RF, XGB, LightGBM on Morgan features with scaffold split
3. **Compare Splits**: Evaluate scaffold vs random split performance difference
4. **Add GNN Models**: Implement GNN training when baseline is stable
5. **Plan ZINC22**: Design pretraining integration (not implemented yet)

## Migration Guide

For users familiar with the old structure:

### Old → New Mapping

| Old | New |
|-----|-----|
| `scripts/01_prepare_splits.py` | `scripts/baseline/01_preprocess_b3db.py` |
| `scripts/02_featurize_all.py` | `scripts/baseline/02_compute_features.py` |
| `scripts/03_run_baselines.py` | `scripts/baseline/03_train_baselines.py` |
| `src/featurize/fingerprints.py` | `src/features/fingerprints.py` |
| `src/featurize/rdkit_descriptors.py` | `src/features/descriptors.py` |
| `src/featurize/graph_pyg.py` | `src/features/graph.py` |
| `src/baseline/train_baselines.py` | `src/train/trainer.py` + `src/models/` |

### Key Changes

1. **Unified Interface**: All models now use the same `BaselineModel` interface
2. **Clear Separation**: Data, features, models, training, and evaluation are in separate modules
3. **Factory Pattern**: `ModelFactory` for convenient model creation
4. **Result Containers**: `TrainingResult` and `ModelComparison` for structured results
5. **Report Generation**: Automatic CSV and text reports

## Usage Example

```bash
# Complete workflow
python scripts/baseline/01_preprocess_b3db.py --seed 0
python scripts/baseline/02_compute_features.py --seed 0 --feature morgan
python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models rf,xgb,lgbm

# Check results
cat artifacts/models/baselines/seed_0/scaffold/morgan/reports/results_summary.csv
```

## Conclusion

The reorganization achieves:
- ✅ Cleaner, more maintainable code structure
- ✅ Clear separation of concerns
- ✅ Preserved useful existing code
- ✅ Ready for B3DB preprocessing and baseline experiments
- ✅ Suitable for long-term research and CFFF platform
- ✅ Not implementing ZINC22 pretraining yet (as requested)
- ✅ Focus on B3DB preprocessing and baselines first (as requested)
