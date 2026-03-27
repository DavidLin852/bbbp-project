# BBB Baseline Results

This directory contains aggregated results from baseline experiments.

## Files

### Master Result Table
- **`baseline_results_master.csv`** - Complete table of all experimental results
  - **This is the canonical baseline result table**
  - Primary metric: `test_auc` (Test ROC AUC)
  - Updated by running: `python scripts/analysis/aggregate_results.py`

### Summary Tables
- **`baseline_summary_by_model.csv`** - Results grouped by model type
- **`baseline_summary_by_feature.csv`** - Results grouped by feature type

## Current Results

### Summary Statistics

| Feature | Model | Test AUC | Test F1 |
|---------|-------|----------|---------|
| descriptors_basic | rf | 0.9542 | 0.9574 |
| morgan | lgbm | 0.9543 | 0.9624 |

### Complete Results

```
seed,split_type,feature,model_name,test_auc,test_f1
0,scaffold,descriptors_basic,rf,0.9542,0.9574
0,scaffold,morgan,lgbm,0.9543,0.9624
```

## Updating Results

### After Running New Experiments

1. **Run experiments:**
   ```bash
   python scripts/baseline/01_preprocess_b3db.py --seed 0
   python scripts/baseline/02_compute_features.py --seed 0 --split scaffold --feature morgan
   python scripts/baseline/03_train_baselines.py --seed 0 --split scaffold --feature morgan --models rf,xgb,lgbm
   ```

2. **Aggregate results:**
   ```bash
   python scripts/analysis/aggregate_results.py
   ```

3. **View updated master table:**
   ```bash
   cat artifacts/reports/baseline_results_master.csv
   ```

### Running Complete Experiment Matrix

```bash
# Run all experiments (recommended: 3 seeds × 2 splits × 3 features × 3 models = 54 experiments)
python scripts/analysis/run_baseline_matrix.py --seeds 0,1,2 --splits scaffold,random --features morgan,combined,descriptors_basic --models rf,xgb,lgbm

# Or run with dry-run first to see what will be executed
python scripts/analysis/run_baseline_matrix.py --seeds 0,1,2 --dry_run
```

## Result Format

### Master Table Columns

| Column | Description |
|--------|-------------|
| `seed` | Random seed (0, 1, 2, ...) |
| `split_type` | Data split method (`scaffold`, `random`) |
| `feature` | Feature type (`morgan`, `combined`, `descriptors_basic`, etc.) |
| `model_name` | Model (`rf`, `xgb`, `lgbm`, etc.) |
| `train_auc` | Training ROC AUC |
| `train_accuracy` | Training accuracy |
| `train_f1` | Training F1 score |
| `val_auc` | Validation ROC AUC |
| `val_accuracy` | Validation accuracy |
| `val_f1` | Validation F1 score |
| `test_auc` | **Test ROC AUC (primary metric)** |
| `test_accuracy` | Test accuracy |
| `test_f1` | Test F1 score |

### Primary Metric

**Test AUC** (`test_auc`) is the primary metric for:
- Model comparison
- Feature selection
- Reporting results

## Archiving

When experiments are complete, archive results:

```bash
# Create archive
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tar -czf archives/baseline_${TIMESTAMP}.tar.gz \
    artifacts/reports/baseline_*.csv \
    artifacts/models/baselines/
```

## Notes

- Results are automatically organized by seed/split/feature
- Each experiment creates a `comparison.json` file
- The aggregation script scans all `comparison.json` files
- Re-running aggregation updates all summary tables
