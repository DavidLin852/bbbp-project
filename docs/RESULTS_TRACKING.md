# Results Tracking and Reporting

This document explains how to aggregate and summarize baseline experiment results.

## Result Files Structure

### Primary Output Files

All aggregated results are stored in `artifacts/reports/`:

1. **`baseline_results_master.csv`** - Complete table of all experimental results
2. **`baseline_summary_by_model.csv`** - Results grouped by model type
3. **`baseline_summary_by_feature.csv`** - Results grouped by feature type

### Master Result Table Format

The master table (`baseline_results_master.csv`) is the **canonical baseline result table** with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `seed` | Random seed for reproducibility | 0, 1, 2, ... |
| `split_type` | Data split method | `scaffold`, `random` |
| `feature` | Feature type used | `morgan`, `combined`, `descriptors_basic`, etc. |
| `model_name` | Model identifier | `rf`, `xgb`, `lgbm`, `svm`, etc. |
| `train_auc` | Training ROC AUC | 0.998 |
| `train_accuracy` | Training accuracy | 0.985 |
| `train_f1` | Training F1 score | 0.990 |
| `val_auc` | Validation ROC AUC | 0.945 |
| `val_accuracy` | Validation accuracy | 0.915 |
| `val_f1` | Validation F1 score | 0.935 |
| `test_auc` | **Test ROC AUC (primary metric)** | 0.963 |
| `test_accuracy` | Test accuracy | 0.940 |
| `test_f1` | Test F1 score | 0.956 |

**Primary metric for model comparison:** `test_auc`

## Updating Results After New Experiments

### Step 1: Run New Experiments

```bash
# Example: Run experiments with different features
python scripts/baseline/02_compute_features.py --seed 0 --split scaffold --feature maccs
python scripts/baseline/03_train_baselines.py --seed 0 --split scaffold --feature maccs --models rf,xgb,lgbm

# Example: Run with different seeds
python scripts/baseline/01_preprocess_b3db.py --seed 1
python scripts/baseline/02_compute_features.py --seed 1 --split scaffold --feature morgan
python scripts/baseline/03_train_baselines.py --seed 1 --split scaffold --feature morgan --models rf,xgb,lgbm
```

### Step 2: Aggregate Results

```bash
# Aggregate all results
python scripts/analysis/aggregate_results.py

# Or filter by specific conditions
python scripts/analysis/aggregate_results.py --seed 0
python scripts/analysis/aggregate_results.py --feature morgan
python scripts/analysis/aggregate_results.py --split scaffold
```

### Step 3: View Updated Summary

```bash
# View master table
cat artifacts/reports/baseline_results_master.csv

# View top results
head -11 artifacts/reports/baseline_results_master.csv | column -t -s,
```

## Experiment Tracking Guidelines

### Recommended Experiment Matrix

For a complete baseline evaluation, run experiments with:

**Seeds:** 0, 1, 2 (3 runs for statistical significance)

**Split types:**
- `scaffold` (primary)
- `random` (for comparison)

**Features:**
- `morgan` (ECFP4, 2048 bits)
- `combined` (all fingerprints)
- `descriptors_basic` (13 physicochemical descriptors)

**Models:**
- `rf` (Random Forest)
- `xgb` (XGBoost)
- `lgbm` (LightGBM)

This gives: 3 seeds × 2 splits × 3 features × 3 models = **54 experiments**

### Experiment Naming Convention

Results are automatically organized by directory structure:

```
artifacts/models/baselines/
└── seed_{seed}/
    └── {split_type}/
        └── {feature}/
            ├── {model_name}/
            │   ├── {model_name}_model.joblib
            │   ├── train_predictions.csv
            │   ├── val_predictions.csv
            │   └── test_predictions.csv
            ├── comparison.json
            └── reports/
                ├── results_summary.csv
                ├── results_sorted_by_auc.csv
                ├── best_model.txt
                └── summary.json
```

### Result File Formats

#### 1. Individual Experiment Results

Each experiment creates a `comparison.json`:

```json
{
  "results": [
    {
      "model_name": "xgb",
      "feature_type": "morgan",
      "train_auc": 0.998,
      "train_accuracy": 0.981,
      "train_f1": 0.987,
      "val_auc": 0.945,
      "val_accuracy": 0.902,
      "val_f1": 0.931,
      "test_auc": 0.963,
      "test_accuracy": 0.931,
      "test_f1": 0.956
    }
  ],
  "summary": {
    "n_models": 1,
    "best_test_auc": 0.963,
    "best_model_name": "xgb",
    "mean_test_auc": 0.963,
    "std_test_auc": 0.0
  }
}
```

#### 2. Master Table (CSV)

```csv
seed,split_type,feature,model_name,train_auc,train_accuracy,train_f1,val_auc,val_accuracy,val_f1,test_auc,test_accuracy,test_f1
0,scaffold,morgan,xgb,0.998,0.981,0.987,0.945,0.902,0.931,0.963,0.931,0.956
0,scaffold,morgan,rf,0.999,0.984,0.990,0.947,0.910,0.938,0.961,0.928,0.953
```

## Running on CFFF

### Batch Experiment Script

Create `run_experiments.sh`:

```bash
#!/bin/bash

# BBB Baseline Experiments
# Run on CFFF with: sbatch run_experiments.sh

SEEDS=(0 1 2)
SPLITS=("scaffold" "random")
FEATURES=("morgan" "combined" "descriptors_basic")
MODELS=("rf" "xgb" "lgbm")

for SEED in "${SEEDS[@]}"; do
    echo "=== Seed $SEED ==="

    # Step 1: Preprocess
    python scripts/baseline/01_preprocess_b3db.py --seed $SEED

    for SPLIT in "${SPLITS[@]}"; do
        for FEATURE in "${FEATURES[@]}"; do
            echo "Processing: $SEED $SPLIT $FEATURE"

            # Step 2: Compute features
            python scripts/baseline/02_compute_features.py \
                --seed $SEED --split $SPLIT --feature $FEATURE

            # Step 3: Train models
            python scripts/baseline/03_train_baselines.py \
                --seed $SEED --split $SPLIT --feature $FEATURE \
                --models ${MODELS[@]}
        done
    done
done

# Aggregate all results
python scripts/analysis/aggregate_results.py
```

### Monitoring Progress

Check results as experiments complete:

```bash
# Count completed experiments
find artifacts/models/baselines -name "comparison.json" | wc -l

# View latest results
tail -20 artifacts/reports/baseline_results_master.csv

# Check for failed experiments
grep -r "Traceback" artifacts/models/baselines/
```

## Archiving Results

### Final Result Archive

When experiments are complete, create an archive:

```bash
# Create timestamped archive
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_DIR="archives/baseline_${TIMESTAMP}"

# Copy master results
mkdir -p $ARCHIVE_DIR
cp artifacts/reports/baseline_*.csv $ARCHIVE_DIR/

# Copy all experiment results
cp -r artifacts/models/baselines $ARCHIVE_DIR/

# Create summary
cat > $ARCHIVE_DIR/README.txt << EOF
BBB Baseline Results - $TIMESTAMP

Experiments: $(find $ARCHIVE_DIR -name "comparison.json" | wc -l)
Best Test AUC: $(cat $ARCHIVE_DIR/baseline_results_master.csv | awk -F, 'NR>1 {print $11}' | sort -rn | head -1)
EOF

# Compress
tar -czf ${ARCHIVE_DIR}.tar.gz $ARCHIVE_DIR
```

## Troubleshooting

### Issue: Missing results in master table

**Check:** Does `comparison.json` exist for the experiment?

```bash
find artifacts/models/baselines -name "comparison.json"
```

**Fix:** Re-run the experiment or check for errors in training output.

### Issue: Wrong path parsing

**Check:** Is the directory structure correct?

Expected: `artifacts/models/baselines/seed_{N}/{split}/{feature}/comparison.json`

**Fix:** Ensure experiments used correct output paths.

### Issue: Duplicate entries

**Check:** Are experiments being re-run with same seed/split/feature?

**Fix:** Delete old results or use different seeds before re-running.
