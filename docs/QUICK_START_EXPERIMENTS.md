# Quick Start: Baseline Experiments and Reporting

This guide shows how to run baseline experiments and generate reports.

## Quick Start (5 Minutes)

### 1. Run Single Experiment

```bash
# Preprocess data
python scripts/baseline/01_preprocess_b3db.py --seed 0

# Compute features
python scripts/baseline/02_compute_features.py --seed 0 --split scaffold --feature morgan

# Train models
python scripts/baseline/03_train_baselines.py --seed 0 --split scaffold --feature morgan --models rf,xgb,lgbm
```

### 2. Aggregate Results

```bash
python scripts/analysis/aggregate_results.py
```

### 3. View Results

```bash
# View master table
cat artifacts/reports/baseline_results_master.csv

# Or open in Excel/spreadsheet software
start artifacts/reports/baseline_results_master.csv  # Windows
xdg-open artifacts/reports/baseline_results_master.csv  # Linux
open artifacts/reports/baseline_results_master.csv  # macOS
```

## Running Multiple Experiments

### Option 1: Manual Loop

```bash
# Test different features
for feat in morgan maccs combined descriptors_basic; do
    python scripts/baseline/02_compute_features.py --seed 0 --split scaffold --feature $feat
    python scripts/baseline/03_train_baselines.py --seed 0 --split scaffold --feature $feat --models rf,xgb,lgbm
done

# Aggregate
python scripts/analysis/aggregate_results.py
```

### Option 2: Automated Matrix

```bash
# Run full matrix (3 seeds × 2 splits × 3 features × 3 models = 54 experiments)
python scripts/analysis/run_baseline_matrix.py \
    --seeds 0,1,2 \
    --splits scaffold,random \
    --features morgan,combined,descriptors_basic \
    --models rf,xgb,lgbm

# Takes ~30-60 minutes depending on hardware
```

### Option 3: Dry Run First

```bash
# See what will be executed without actually running
python scripts/analysis/run_baseline_matrix.py --seeds 0,1,2 --dry_run
```

## Understanding Results

### Master Table Format

```csv
seed,split_type,feature,model_name,test_auc,test_f1
0,scaffold,morgan,xgb,0.963,0.956
0,scaffold,morgan,rf,0.961,0.953
```

- **`test_auc`** = Primary metric (higher is better)
- Sort by this column to find best models

### Finding Best Model

```bash
# View top 10 results
head -11 artifacts/reports/baseline_results_master.csv | sort -t, -k11 -rn | head -11
```

Or use Python:

```python
import pandas as pd

df = pd.read_csv('artifacts/reports/baseline_results_master.csv')
print(df.nlargest(10, 'test_auc')[['seed', 'split_type', 'feature', 'model_name', 'test_auc', 'test_f1']])
```

## Result Files

### Individual Experiment Results

Location: `artifacts/models/baselines/seed_{seed}/{split}/{feature}/`

```
artifacts/models/baselines/seed_0/scaffold/morgan/
├── rf/
│   ├── rf_model.joblib              # Trained model
│   ├── train_predictions.csv        # Training predictions
│   ├── val_predictions.csv          # Validation predictions
│   └── test_predictions.csv         # Test predictions
├── xgb/
│   └── ...
├── comparison.json                  # All results for this experiment
└── reports/
    ├── results_summary.csv          # Summary table
    ├── best_model.txt               # Best model info
    └── summary.json                 # Statistics
```

### Aggregated Results

Location: `artifacts/reports/`

```
artifacts/reports/
├── baseline_results_master.csv           # ⭐ MASTER TABLE
├── baseline_summary_by_model.csv         # Grouped by model
├── baseline_summary_by_feature.csv        # Grouped by feature
└── README.md                              # Detailed documentation
```

## Updating Results

### After New Experiments

1. **Run new experiments** (see above)
2. **Re-aggregate results:**
   ```bash
   python scripts/analysis/aggregate_results.py
   ```
3. **Master table is automatically updated**

### Partial Updates

You can filter aggregation by specific conditions:

```bash
# Only seed 0
python scripts/analysis/aggregate_results.py --seed 0

# Only scaffold split
python scripts/analysis/aggregate_results.py --split scaffold

# Only Morgan features
python scripts/analysis/aggregate_results.py --feature morgan
```

## Common Workflows

### Workflow 1: Quick Baseline

```bash
# Run default (RF, XGB, LightGBM on Morgan)
python scripts/baseline/01_preprocess_b3db.py --seed 0
python scripts/baseline/02_compute_features.py --seed 0 --split scaffold --feature morgan
python scripts/baseline/03_train_baselines.py --seed 0 --split scaffold --feature morgan --models rf,xgb,lgbm
python scripts/analysis/aggregate_results.py
```

### Workflow 2: Compare Features

```bash
# Test multiple features with same model
for feat in morgan maccs combined descriptors_basic; do
    python scripts/baseline/02_compute_features.py --seed 0 --split scaffold --feature $feat
    python scripts/baseline/03_train_baselines.py --seed 0 --split scaffold --feature $feat --models rf
done
python scripts/analysis/aggregate_results.py
# Compare: cat artifacts/reports/baseline_summary_by_feature.csv
```

### Workflow 3: Statistical Significance

```bash
# Run multiple seeds
for seed in 0 1 2; do
    python scripts/baseline/01_preprocess_b3db.py --seed $seed
    python scripts/baseline/02_compute_features.py --seed $seed --split scaffold --feature morgan
    python scripts/baseline/03_train_baselines.py --seed $seed --split scaffold --feature morgan --models rf,xgb,lgbm
done
python scripts/analysis/aggregate_results.py
# Check std in: cat artifacts/reports/baseline_summary_by_model.csv
```

### Workflow 4: Scaffold vs Random Split

```bash
# Compare split types
python scripts/baseline/01_preprocess_b3db.py --seed 0 --split_type scaffold
python scripts/baseline/02_compute_features.py --seed 0 --split scaffold --feature morgan
python scripts/baseline/03_train_baselines.py --seed 0 --split scaffold --feature morgan --models rf,xgb,lgbm

python scripts/baseline/01_preprocess_b3db.py --seed 0 --split_type random
python scripts/baseline/02_compute_features.py --seed 0 --split random --feature morgan
python scripts/baseline/03_train_baselines.py --seed 0 --split random --feature morgan --models rf,xgb,lgbm

python scripts/analysis/aggregate_results.py
# Compare: cat artifacts/reports/baseline_results_master.csv | grep -E "scaffold|random"
```

## Running on CFFF

### Prepare Script

Create `run_baseline_cfff.sh`:

```bash
#!/bin/bash
# BBB Baseline Experiments for CFFF

module load python/3.10
pip install -r requirements.txt

# Run full matrix
python scripts/analysis/run_baseline_matrix.py \
    --seeds 0,1,2 \
    --splits scaffold \
    --features morgan,combined,descriptors_basic \
    --models rf,xgb,lgbm

# Copy results to output directory
cp -r artifacts/reports/ $OUTPUT_DIR/
cp -r artifacts/models/baselines/ $OUTPUT_DIR/
```

### Submit Job

```bash
sbatch run_baseline_cfff.sh
```

## Troubleshooting

### Issue: Missing results in master table

**Check:** Did the experiment complete successfully?

```bash
# Find all comparison.json files
find artifacts/models/baselines -name "comparison.json"

# Should see one file per (seed, split, feature) combination
```

**Fix:** Re-run failed experiments

### Issue: Duplicate entries

**Cause:** Re-running experiments with same seed/split/feature

**Fix:**
```bash
# Delete old results before re-running
rm -rf artifacts/models/baselines/seed_0/scaffold/morgan/
# Then re-run
```

### Issue: Wrong aggregation

**Cause:** Old comparison.json files from previous runs

**Fix:**
```bash
# Clean and re-aggregate
rm artifacts/reports/baseline_*.csv
python scripts/analysis/aggregate_results.py
```

## Next Steps

After baseline is established:

1. **Analyze results:** Find best model/feature combinations
2. **Interpretability:** Run SHAP analysis on best models
3. **Advanced models:** Try GNNs, Transformers
4. **Hyperparameter tuning:** Optimize best models
5. **Ensemble:** Combine top models
