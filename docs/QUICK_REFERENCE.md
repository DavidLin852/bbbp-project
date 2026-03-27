# Quick Reference - BBB Prediction Project

## Common Commands

### Data Preprocessing

```bash
# Classification (default: groups A,B, scaffold split)
python scripts/baseline/01_preprocess_b3db.py --seed 0

# Regression
python scripts/baseline/01_preprocess_b3db.py --seed 0 --task regression

# Custom groups
python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups A,B,C

# Random split (for comparison)
python scripts/baseline/01_preprocess_b3db.py --seed 0 --split_type random
```

### Feature Computation

```bash
# Morgan (ECFP4) - most common
python scripts/baseline/02_compute_features.py --seed 0 --feature morgan

# Combined fingerprints (Morgan + MACCS + AtomPairs + FP2)
python scripts/baseline/02_compute_features.py --seed 0 --feature combined

# All descriptors
python scripts/baseline/02_compute_features.py --seed 0 --feature descriptors_all
```

### Model Training

```bash
# Core models (RF, XGB, LightGBM)
python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models rf,xgb,lgbm

# All models
python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models rf,xgb,lgbm,svm,knn,lr,nb,gb,ada,etc
```

## Module Usage Examples

### Data Module

```python
from src.data import B3DBPreprocessor, scaffold_split, B3DBDataset

# Load and preprocess
preprocessor = B3DBPreprocessor()
data = preprocessor.load_classification(
    filepath="data/raw/B3DB_classification.tsv",
    groups=("A", "B"),
)

# Perform scaffold split
split = scaffold_split(
    df=data.df,
    label_col="y_cls",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=0,
)

# Create dataset
dataset = B3DBDataset(
    train_df=split.train,
    val_df=split.val,
    test_df=split.test,
    task="classification",
)

# Save splits
dataset.save_splits("output_dir")
```

### Features Module

```python
from src.features import FingerprintGenerator, DescriptorGenerator

# Fingerprints
fp_gen = FingerprintGenerator()
X_morgan = fp_gen.compute(smiles_list, fingerprint_type="morgan")
X_combined = fp_gen.compute(smiles_list, fingerprint_type="combined")

# Descriptors
desc_gen = DescriptorGenerator(descriptor_set="all")
X_desc = desc_gen.compute(smiles_list)
X_desc_norm = desc_gen.fit_normalize(X_desc)
```

### Models Module

```python
from src.models import ModelFactory

factory = ModelFactory()

# Create specific models
rf_model = factory.create_rf()
xgb_model = factory.create_xgb()
lgbm_model = factory.create_lgbm()

# Train
rf_model.fit(X_train, y_train)

# Predict
probs = rf_model.predict_proba(X_test)[:, 1]
preds = rf_model.predict(X_test)
```

### Training Module

```python
from src.train import Trainer, TrainingConfig

config = TrainingConfig(
    output_dir="artifacts/models",
    save_model=True,
    save_predictions=True,
)

trainer = Trainer(model, config)
result = trainer.train(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
)

print(f"Test AUC: {result.test_metrics.auc:.4f}")
```

### Evaluation Module

```python
from src.evaluate import ModelComparison, generate_report

# Compare models
comparison = ModelComparison(results)

# Get sorted results
df_sorted = comparison.sort_by_test_auc()

# Get best model
best = comparison.get_best_model()

# Generate report
generate_report(comparison, output_dir="reports")
```

## Data Splits

### Recommended Splits

| Split | Groups | Samples | BBB+ Rate | Use Case |
|-------|--------|---------|-----------|----------|
| A,B | A,B | 3,743 | 76.5% | Best balance ⭐ |
| A,B,C | A,B,C | 6,203 | 66.7% | Larger scale |
| A,B,C,D | A,B,C,D | 6,244 | 63.5% | Maximum coverage |

### Split Types

| Type | Description | Recommendation |
|------|-------------|----------------|
| Scaffold | Structural diversity | ✅ Recommended |
| Random | Stratified random | For comparison only |

## Feature Types

### Fingerprints

| Type | Dimension | Description |
|------|-----------|-------------|
| morgan | 2048 | ECFP4 (radius 2) |
| maccs | 167 | MACCS keys |
| atom_pairs | 1024 | Hashed atom pairs |
| fp2 | 2048 | RDKit fingerprint |
| combined | 5287 | All 4 fingerprints |

### Descriptors

| Set | Count | Description |
|-----|-------|-------------|
| basic | 13 | Core descriptors (MW, LogP, TPSA, etc.) |
| extended | ~100 | Extended set with topological indices |
| all | 200+ | Comprehensive descriptor set |

## Model Types

| Code | Model | Description |
|------|-------|-------------|
| rf | Random Forest | Robust baseline |
| xgb | XGBoost | High performance |
| lgbm | LightGBM | Fast training |
| svm | SVM RBF | Non-linear |
| knn | KNN | Instance-based |
| lr | Logistic Regression | Linear baseline |
| nb | Naive Bayes | Probabilistic |
| gb | Gradient Boosting | sklearn GB |
| ada | AdaBoost | Boosting |
| etc | Extra Trees | Random forest variant |

## File Locations

### Input Data
- Raw B3DB: `data/raw/B3DB_*.tsv`
- Processed splits: `data/splits/seed_{seed}/{task}_{split}/`

### Features
- Computed features: `artifacts/features/seed_{seed}/{split}/{feature}/`

### Models
- Trained models: `artifacts/models/baselines/seed_{seed}/{split}/{feature}/`
- Reports: `artifacts/models/.../reports/`

## Troubleshooting

### Import Errors
```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Missing Data
```bash
# Check data locations
ls data/raw/
ls data/splits/
```

### Memory Issues
- Use `morgan` instead of `combined` for fewer features
- Reduce `n_estimators` in model config
- Use smaller groups (e.g., just group A)

### Slow Training
- Reduce model complexity (n_estimators, max_depth)
- Use fewer models
- Use `lgbm` for faster training than `xgb`
