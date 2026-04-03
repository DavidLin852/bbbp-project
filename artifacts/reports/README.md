# Benchmark Reports Directory

This directory contains all benchmark results and summaries for the BBB permeability prediction project.

---

## Directory Structure

```
artifacts/reports/
├── benchmark_summary.csv                    # Classical baseline summary (main)
├── baseline_results_master.csv              # All classical results (detailed)
��── baseline_summary_by_feature.csv          # Classical results by feature
├── baseline_summary_by_model.csv            # Classical results by model
├── cls_benchmark_scaffold.csv               # Classification scaffold results
├── cls_results_master.csv                   # Classification master results
├── cls_results_scaffold.csv                 # Classification scaffold detailed
├── cls_summary_by_model.csv                 # Classification by model
├── reg_benchmark_scaffold.csv               # Regression scaffold results
├── reg_results_master.csv                   # Regression master results
├── reg_results_scaffold.csv                 # Regression scaffold detailed
├── reg_summary_by_model.csv                 # Regression by model
├── gnn/                                     # GNN baseline results
│   └── gnn_benchmark_scaffold_YYYYMMDD_HHMMSS.csv
└── transformer/                             # Transformer baseline results
    └── transformer_benchmark_scaffold_YYYYMMDD_HHMMSS.csv
```

---

## Report Files

### Classical Baselines

#### `benchmark_summary.csv`
**Main summary of classical baseline results**

Columns:
- `rank`: Ranking by test AUC
- `feature`: Feature type (morgan, descriptors_basic, etc.)
- `model_name`: Model type (rf, xgb, lgbm)
- `test_auc_mean`: Mean test AUC across seeds
- `test_auc_std`: Standard deviation of test AUC
- `test_f1_mean`: Mean test F1 across seeds
- `test_f1_std`: Standard deviation of test F1
- `train_auc_mean`: Mean training AUC
- `val_auc_mean`: Mean validation AUC
- `seed_count`: Number of seeds

**Best Result (Current):**
- Rank 1: Morgan + RF, Test AUC = 0.9401 ± 0.0454

#### `baseline_results_master.csv`
**Master table with all classical baseline results**

Contains per-seed results for all combinations of:
- Seeds: 0, 1, 2
- Features: morgan, descriptors_basic
- Models: rf, xgb, lgbm
- Tasks: classification

#### `baseline_summary_by_feature.csv`
**Results aggregated by feature type**

Shows performance of each feature type averaged across models and seeds.

#### `baseline_summary_by_model.csv`
**Results aggregated by model type**

Shows performance of each model averaged across features and seeds.

### Classification Results

#### `cls_benchmark_scaffold.csv`
**Classification results on scaffold split**

Aggregated classification results with mean ± std across seeds.

#### `cls_results_master.csv`
**Detailed per-seed classification results**

All classification experiments with individual seed results.

#### `cls_results_scaffold.csv`
**Scaffold split classification results**

Per-split and per-seed classification metrics.

#### `cls_summary_by_model.csv`
**Classification results summarized by model**

### Regression Results

#### `reg_benchmark_scaffold.csv`
**Regression results on scaffold split**

Aggregated regression results (R², RMSE, MAE) with mean ± std.

#### `reg_results_master.csv`
**Detailed per-seed regression results**

All regression experiments with individual seed results.

#### `reg_results_scaffold.csv`
**Scaffold split regression results**

Per-split and per-seed regression metrics.

#### `reg_summary_by_model.csv`
**Regression results summarized by model**

### GNN Baselines

#### `gnn/gnn_benchmark_scaffold_*.csv`
**GNN benchmark results on scaffold split**

Columns:
- `model`: gcn, gin, or gat
- `task`: classification or regression
- `n_seeds`: Number of seeds (5)
- `train_auc_mean/std`: Training AUC
- `val_auc_mean/std`: Validation AUC
- `test_auc_mean/std`: Test AUC
- `train_r2_mean/std`: Training R²
- `val_r2_mean/std`: Validation R²
- `test_r2_mean/std`: Test R²

**Best Results (Current):**
- Classification: GAT, Test AUC = 0.9356 ± 0.0314
- Regression: GIN, Test R² = 0.7062 ± 0.0473

### Transformer Baselines

#### `transformer/transformer_benchmark_scaffold_*.csv`
**Transformer benchmark results on scaffold split**

Columns:
- `task`: classification or regression
- `seed`: Random seed (0-4)
- `accuracy`, `precision`, `recall`, `f1`: Classification metrics
- `auc`: ROC-AUC for classification
- `r2`, `rmse`, `mae`: Regression metrics

**Best Results (Current):**
- Classification: Mean AUC = 0.8822 ± 0.0756
- Regression: Mean R² = 0.1911 ± 0.2014

---

## Result Integration

### Unified Leaderboard

To compare across all baseline categories, refer to `docs/CURRENT_BASELINE_SUMMARY.md` which contains:

**Classification:**
| Rank | Category | Model | Representation | Test AUC |
|------|----------|-------|----------------|----------|
| 1 | Classical | RF | Morgan | 0.9401 ± 0.0454 |
| 2 | Graph | GAT | Molecular Graph | 0.9356 ± 0.0314 |
| 3 | Sequence | Transformer | SMILES | 0.8822 ± 0.0756 |

**Regression:**
| Rank | Category | Model | Representation | Test R² |
|------|----------|-------|----------------|---------|
| 1 | Graph | GIN | Molecular Graph | 0.7062 ± 0.0473 |
| 2 | Graph | GAT | Molecular Graph | 0.6408 ± 0.0357 |
| 3 | Sequence | Transformer | SMILES | 0.1911 ± 0.2014 |

---

## Updating Reports

### After Running New Experiments

1. **Classical Baselines:**
   ```bash
   python scripts/analysis/aggregate_results.py
   python scripts/analysis/generate_benchmark_summary.py
   ```

2. **GNN Baselines:**
   - Results are automatically saved to `gnn/gnn_benchmark_scaffold_*.csv`
   - No additional aggregation needed

3. **Transformer Baselines:**
   - Results are automatically saved to `transformer/transformer_benchmark_scaffold_*.csv`
   - No additional aggregation needed

### Archiving Reports

When archiving results to CFFF or shared storage:

```bash
# Create timestamped archive
TIMESTAMP=$(date +%Y%m%d)
ARCHIVE_DIR="/path/to/archive/${TIMESTAMP}_results"
mkdir -p $ARCHIVE_DIR

# Copy all reports
cp -r artifacts/reports/* $ARCHIVE_DIR/

# Verify
ls -lh $ARCHIVE_DIR
```

---

## File Formats

All result files are in CSV format with the following conventions:

- **Mean ± Std:** Reported as separate columns (`_mean`, `_std`)
- **Seeds:** Results aggregated across multiple seeds (typically 3 or 5)
- **Splits:** Scaffold split is the primary evaluation protocol
- **Metrics:**
  - Classification: AUC (primary), F1, accuracy, precision, recall
  - Regression: R² (primary), RMSE, MAE

---

## Contact and Support

For questions about results or reporting:
- Refer to `docs/USAGE_GUIDE.md` for command-line usage
- Refer to `docs/CURRENT_BASELINE_SUMMARY.md` for baseline interpretation
- Refer to `CLAUDE.md` for comprehensive project documentation

---

**Last Updated:** 2026-04-03
**Project Status:** Baseline phase complete
