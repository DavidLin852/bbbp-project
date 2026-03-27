# BBB Baseline Benchmark - First Formal Results

## Summary

**Date:** 2025-03-27
**Dataset:** B3DB (Groups A+B)
**Split:** Scaffold split (80:10:10)
**Seeds:** 0, 1, 2
**Metric:** Test ROC AUC (primary)

---

## MAIN CONCLUSION (Stable Baseline)

### Strongest Classical Baseline

| Configuration | Value |
|---------------|-------|
| **Feature** | **Morgan (ECFP4)** |
| **Model** | **Random Forest** |
| **Test AUC** | **0.9401 ± 0.0454** |
| **Test F1** | **0.9391 ± 0.0270** |
| **Seeds** | 3 |

### How to Use This Baseline

This configuration should be used as:
- ✅ **Reference point** for all new model evaluations
- ✅ **Minimum performance threshold** (new models should beat 0.94 AUC)
- ✅ **Standard** for scaffold-split evaluation on B3DB
- ✅ **Baseline for ablation studies** (e.g., different splits, features)

---

## Complete Benchmark Summary

| Rank | Feature | Model | Test AUC (mean±std) | Test F1 (mean±std) | Train AUC | Val AUC |
|------|---------|-------|---------------------|-------------------|-----------|---------|
| 1 | morgan | rf | **0.9401 ± 0.0454** | 0.9391 ± 0.0270 | 0.9995 | 0.9577 |
| 2 | morgan | xgb | 0.9198 ± 0.0674 | 0.9335 ± 0.0310 | 0.9981 | 0.9393 |
| 3 | morgan | lgbm | 0.9195 ± 0.0619 | 0.9363 ± 0.0331 | 0.9998 | 0.9457 |
| 4 | descriptors_basic | rf | 0.9159 ± 0.0730 | 0.9397 ± 0.0270 | 0.9995 | 0.9500 |
| 5 | descriptors_basic | xgb | 0.9115 ± 0.0712 | 0.9370 ± 0.0308 | 0.9998 | 0.9465 |
| 6 | descriptors_basic | lgbm | 0.9114 ± 0.0815 | 0.9395 ± 0.0298 | 0.9998 | 0.9448 |

---

## TEMPORARY OBSERVATIONS

**Note:** These observations are based on the current experiment matrix and may change with additional data or configurations.

### 1. Feature Performance

| Feature | Average Test AUC | Verdict |
|---------|------------------|---------|
| **morgan** | **0.9265** | **Best feature** ⭐ |
| descriptors_basic | 0.9129 | Good, but 1.4% lower than Morgan |

**Conclusion:** Morgan fingerprints (ECFP4, 2048 bits) are the strongest feature type for this task.

### 2. Model Performance

| Model | Average Test AUC | Verdict |
|-------|------------------|---------|
| **rf** | **0.9280** | **Best model** ⭐ |
| xgb | 0.9156 | Good, but 1.4% lower than RF |
| lgbm | 0.9154 | Good, but 1.3% lower than RF |

**Conclusion:** Random Forest performs best on average, but all 3 models are competitive.

### 3. Stability Analysis

Lower standard deviation = more stable across random seeds.

| Configuration | Test AUC Std | Verdict |
|---------------|-------------|---------|
| **morgan + rf** | **0.0454** | **Most stable** ⭐ |
| morgan + lgbm | 0.0619 | Stable |
| morgan + xgb | 0.0674 | Moderate variation |
| descriptors_basic + rf | 0.0730 | Moderate variation |
| descriptors_basic + xgb | 0.0712 | Moderate variation |
| descriptors_basic + lgbm | 0.0815 | Highest variation |

**Conclusion:** The best baseline (RF + Morgan) is also the most stable across seeds.

### 4. Overfitting Analysis

Comparing train vs test AUC gap:

| Configuration | Train AUC | Test AUC | Gap | Verdict |
|---------------|-----------|----------|-----|---------|
| All configs | ~0.999 | 0.91-0.94 | ~0.06-0.09 | **Moderate overfitting** ⚠️ |

**Observation:** All models show some overfitting (train AUC >> test AUC). This is expected for:
- Small dataset (n=4,679 after filtering)
- Complex models (800-2000 trees)
- High-dimensional features (2048 dimensions)

**Recommendation:** Consider regularization or hyperparameter tuning to reduce overfitting.

---

## Statistical Significance

### Confidence Intervals (Approximate)

With 3 seeds, we can estimate 95% confidence intervals:

**Best baseline (RF + Morgan):**
- Mean Test AUC: 0.9401
- Std: 0.0454
- 95% CI: 0.9401 ± 0.0525 ≈ **[0.8876, 0.9926]**

**Interpretation:**
- With only 3 seeds, confidence intervals are wide
- True performance likely between 0.89 and 0.99
- Need more seeds (5-10) for tighter bounds

---

## Comparison to Individual Seeds

| Seed | morgan+rf Test AUC | descriptors_basic+rf Test AUC |
|------|-------------------|------------------------------|
| 0 | 0.9641 | 0.9542 |
| 1 | 0.8877 | 0.8317 |
| 2 | 0.9684 | 0.9619 |

**Observation:** Seed 1 shows significantly lower performance than seeds 0 and 2. This suggests:
- High variance due to scaffold split
- Some splits are "easier" than others
- Need more seeds to get stable estimate

---

## Result Files

### Generated Files

1. **`artifacts/reports/benchmark_summary.csv`** - ⭐ Main benchmark table
2. **`artifacts/reports/benchmark_report.txt`** - Detailed report
3. **`artifacts/reports/baseline_results_master.csv`** - All individual results

### How to Regenerate

```bash
# Re-aggregate from individual experiment results
python scripts/analysis/aggregate_results.py

# Regenerate benchmark summary
python scripts/analysis/generate_benchmark_summary.py
```

---

## Recommended Next Steps

### Priority 1: Baseline Refinement (Immediate)

**Goal:** Stabilize and improve the baseline

1. **Hyperparameter tuning**
   ```bash
   # Tune RF hyperparameters on Morgan features
   # Focus on reducing overfitting
   # Key params: max_depth, min_samples_split, n_estimators
   ```

2. **Additional features**
   ```bash
   # Test combined fingerprints (Morgan + MACCS + AtomPairs + FP2)
   # Test maccs keys
   # Compare: morgan vs combined vs maccs
   ```

3. **More seeds for stability**
   ```bash
   # Run seeds 0-9 (10 seeds total)
   # This will give tighter confidence intervals
   # Estimate: 10 seeds × 2 features × 3 models = 60 experiments
   ```

### Priority 2: Ablation Studies (Short-term)

**Goal:** Understand what matters most

1. **Scaffold vs Random split**
   ```bash
   # Compare scaffold split to random stratified split
   # Question: How much does scaffold split affect performance?
   ```

2. **Group filter sensitivity**
   ```bash
   # Test with groups A,B,C (more data, lower quality)
   # Compare to current A,B (less data, higher quality)
   # Question: Is more data better or is quality more important?
   ```

3. **Dataset size analysis**
   ```bash
   # Train on subsets (25%, 50%, 75%, 100%)
   # Question: How much data is needed for good performance?
   ```

### Priority 3: Advanced Models (Future)

**ONLY AFTER baseline is fully understood and stable:**

1. **GNN Models**
   - GAT (Graph Attention Network)
   - GCN (Graph Convolutional Network)
   - **Must beat 0.9401 AUC baseline**

2. **Transformer Models**
   - MolBERT
   - Graphormer
   - **Must beat 0.9401 AUC baseline**

3. **Ensemble Methods**
   - Voting ensemble
   - Stacking ensemble
   - **Must beat 0.9401 AUC baseline**

---

## Key Takeaways

### What We Know (Main Conclusions)

1. ✅ **Strongest baseline:** Random Forest + Morgan fingerprints (0.94 AUC)
2. ✅ **Best feature:** Morgan > descriptors_basic (1.4% better)
3. ✅ **Best model:** RF > XGB ≈ LightGBM (all competitive)
4. ✅ **Stability:** Best baseline is also most stable (lowest std)
5. ✅ **Threshold:** New models must achieve >0.94 AUC to be useful

### What We Don't Know Yet (Temporary Observations)

1. ❓ **Will hyperparameter tuning improve performance?**
   - Current models may be overfitting
   - Need to test regularization

2. ❓ **Will combined features perform better?**
   - Only tested individual features so far
   - Combined may capture complementary information

3. ❓ **Is scaffold split better than random?**
   - Haven't tested random split yet
   - Need ablation study

4. ❓ **Will more data improve performance?**
   - Only used groups A,B (n=4,679)
   - Groups A,B,C has n=6,203
   - Groups A,B,C,D has n=6,244

5. ❓ **Why does seed 1 perform so poorly?**
   - Seed 1: 0.88-0.89 AUC
   - Seeds 0,2: 0.96-0.97 AUC
   - High variance needs investigation

---

## Interpretation Guide

### How to Read the Master CSV

**File:** `artifacts/reports/baseline_results_master.csv`

**Columns:**
- `seed`: Random seed (0, 1, 2)
- `split_type`: Always "scaffold" (our main split)
- `feature`: Feature type (morgan, descriptors_basic)
- `model_name`: Model (rf, xgb, lgbm)
- `test_auc`: ⭐ **Primary metric** - higher is better
- `test_f1`: Secondary metric
- `val_auc`: Validation performance (check for overfitting)
- `train_auc`: Training performance (check for overfitting)

**Finding the best configuration:**
```bash
# Sort by test_auc
cat artifacts/reports/baseline_results_master.csv | sort -t, -k11 -rn

# Or use the summary table
cat artifacts/reports/benchmark_summary.csv
```

### Identifying Overfitting

**Signs of overfitting:**
- Train AUC >> Test AUC (gap > 0.05)
- Val AUC << Train AUC
- High variance across seeds

**Current status:**
- All models show moderate overfitting (gap ~0.06-0.09)
- This is acceptable for baseline
- Should be addressed in refinement phase

---

## Citation

If you use these baseline results, please cite:

```bibtex
@misc{bbb_baseline_2025,
  title={Blood-Brain Barrier Permeability Prediction Baseline},
  author={Your Name},
  year={2025},
  note={Dataset: B3DB, Split: Scaffold (80:10:10), Seeds: 3, Best Test AUC: 0.9401}
}
```

---

**Last Updated:** 2025-03-27
**Status:** First formal baseline established ✅
**Next Review:** After hyperparameter tuning or additional features
