# Sequence Baseline Comparison Methodology

**Last Updated:** 2026-04-03
**Status:** Implementation complete, ready for benchmarking

---

## Overview

This document explains how the SMILES Transformer (sequence baseline) will be compared against classical and graph baselines, and what the project state will be once the Transformer baseline is complete.

---

## 1. Experimental Design

### Consistent Evaluation Protocol

All three baseline categories (classical, graph, sequence) follow the **same evaluation protocol**:

| Component | Specification |
|-----------|----------------|
| **Dataset** | B3DB Groups A+B (n ≈ 1,900 molecules) |
| **Split Type** | Scaffold stratified split (80:10:10 train/val/test) |
| **Seeds** | 5 seeds (0, 1, 2, 3, 4) for robust comparison |
| **Metrics (Classification)** | AUC (primary), F1, Accuracy, Precision, Recall |
| **Metrics (Regression)** | R² (primary), RMSE, MAE |
| **Data Source** | Same pre-split CSV files in `data/splits/seed_{N}/` |

### Key Fairness Guarantees

1. **Identical Splits:** All models use the exact same train/val/test splits
2. **Same Seeds:** Random initialization is controlled with the same seeds
3. **Same Metrics:** Evaluation metrics are identical across all baselines
4. **Same Preprocessing:** SMILES canonicalization is done once and shared

---

## 2. Baseline Comparison Strategy

### 2.1 Classical vs Sequence

**Comparison Dimensions:**

| Aspect | Classical (RF + Morgan) | Sequence (Transformer) |
|--------|-------------------------|----------------------|
| **Input Representation** | Fixed-length fingerprint (2048 bits) | Variable-length SMILES sequence |
| **Feature Type** | Hand-crafted chemical features | Learned token embeddings |
| **Model Architecture** | Random Forest (tree ensemble) | Transformer (attention layers) |
| **Training** | No gradient descent | AdamW optimizer, learning rate scheduling |
| **Interpretability** | Feature importance | Attention weights (future work) |
| **Parameters** | ~100-500 trees | ~500K-2M parameters |
| **Training Time** | Seconds to minutes | Minutes to hours |

**Research Questions:**
- Can sequence-based learning match hand-crafted fingerprints?
- Does the attention mechanism capture structural patterns better than trees?
- What is the trade-off between interpretability and performance?

### 2.2 Graph vs Sequence

**Comparison Dimensions:**

| Aspect | Graph (GAT/GIN) | Sequence (Transformer) |
|--------|-----------------|----------------------|
| **Input Representation** | Molecular graph (nodes + edges) | SMILES string (tokens) |
| **Structure Awareness** | Explicit graph topology | Implicit (learned from sequences) |
| **Invariance** | Permutation invariant (by design) | Position-dependent (relaxed by attention) |
| **Message Passing** | Graph convolution | Self-attention |
| **Computational Cost** | O(|V| × |E|) | O(L² × d) where L = seq length |
| **Parameters** | ~100K-500K | ~500K-2M |

**Research Questions:**
- Is explicit graph structure necessary for BBB prediction?
- Can SMILES sequences capture sufficient structural information?
- Which representation generalizes better to new scaffolds?

### 2.3 Three-Way Comparison

**Performance Tiers:**

1. **Tier 1:** Best classical model (RF + Morgan: AUC 0.9401)
2. **Tier 2:** Best graph model (GAT: AUC 0.9356)
3. **Tier 3:** Sequence model (TBD)

**Expected Outcomes:**

| Scenario | Interpretation |
|----------|----------------|
| **Sequence > Classical & Graph** | Sequential learning is superior; prioritize for pretraining |
| **Sequence ≈ Classical > Graph** | Simpler representations suffice; GNNs not worth complexity |
| **Classical > Sequence ≈ Graph** | Hand-crafted features are hard to beat; focus on feature engineering |
| **Classical > Graph > Sequence** | Graph features help, but sequences insufficient for this task |

---

## 3. Evaluation Metrics

### Classification Task (BBB+ vs BBB-)

**Primary Metric:** ROC-AUC
- **Why AUC?** Threshold-independent, robust to class imbalance
- **Current Best:** RF + Morgan (0.9401 ± 0.0454)

**Secondary Metrics:**
- **F1 Score:** Balance of precision and recall
- **Accuracy:** Overall correctness (less informative with imbalance)
- **Precision/Recall:** Detailed performance analysis

**Statistical Comparison:**
- Mean ± standard deviation across 5 seeds
- Paired t-tests to assess significance (optional)
- Effect size (Cohen's d) for practical significance

### Regression Task (logBB prediction)

**Primary Metric:** R² (coefficient of determination)
- **Why R²?** Measures explained variance, scale-independent
- **Current Best:** GIN (0.7062 ± 0.0473)

**Secondary Metrics:**
- **RMSE:** Root mean squared error (in logBB units)
- **MAE:** Mean absolute error (more interpretable than RMSE)

---

## 4. Result Integration

### 4.1 Unified Results Table

Once Transformer benchmarking is complete, results will be integrated into a unified table:

**Classification Leaderboard:**

| Rank | Model Category | Model | Representation | Test AUC | Test F1 |
|------|---------------|-------|----------------|----------|---------|
| 1 | Classical | RF | Morgan | 0.9401 ± 0.0454 | 0.9391 ± 0.0270 |
| 2 | Graph | GAT | Molecular Graph | 0.9356 ± 0.0314 | 0.9231 ± 0.0185 |
| 3 | Sequence | Transformer | SMILES | TBD | TBD |

**Regression Leaderboard:**

| Rank | Model Category | Model | Representation | Test R² | Test RMSE |
|------|---------------|-------|----------------|----------|-----------|
| 1 | Graph | GIN | Molecular Graph | 0.7062 ± 0.0473 | 0.5472 ± 0.0410 |
| 2 | Classical | (TBD) | (TBD) | TBD | TBD |
| 3 | Sequence | Transformer | SMILES | TBD | TBD |

### 4.2 Report File Structure

```
artifacts/reports/
├── benchmark_summary.csv              # Classical baselines
├── gnn/
│   └── gnn_benchmark_scaffold_*.csv   # GNN baselines
└── transformer/
    └── transformer_benchmark_scaffold_*.csv  # Transformer baselines
```

Each CSV file contains:
- Per-seed results
- Mean ± standard deviation
- Training metadata (epochs, best epoch, etc.)

---

## 5. Project State After Transformer Baseline

### 5.1 Completed Components

✅ **Data Pipeline**
- B3DB preprocessing and canonicalization
- Scaffold stratified split generation
- Train/val/test splits for 5 seeds

✅ **Classical Baselines**
- 6 feature representations (Morgan, descriptors_basic, etc.)
- 3 model families (RF, XGB, LightGBM)
- Formal benchmarking with 3 seeds

✅ **Graph Baselines**
- 3 GNN architectures (GCN, GIN, GAT)
- Classification and regression tasks
- Formal benchmarking with 5 seeds

✅ **Sequence Baselines**
- SMILES tokenization
- Transformer architecture
- Training and evaluation pipeline
- Benchmarking script (ready to run)

### 5.2 Ready for Next Phase

**What's Complete:**
- All three representation categories (fingerprints, graphs, sequences)
- Reproducible evaluation protocol
- Baseline comparison framework
- Checkpoint management for pretrained models

**What's Next:**
- **ZINC22 Pretraining:** Use GCN, GIN, GAT, and Transformer as downstream models
- **Pretrained vs Non-pretrained:** Compare performance with and without ZINC22 pretraining
- **Representation Transfer:** Test which representation benefits most from pretraining

### 5.3 Decision Points

After Transformer baseline is complete, we can answer:

1. **Which representation is best for BBB prediction?**
   - If classical wins: Focus on feature engineering
   - If graph wins: Invest in GNN research
   - If sequence wins: Prioritize Transformer-based approaches

2. **Should we invest in pretraining?**
   - If sequence baseline is competitive: Pretraining is promising
   - If gap to classical is large: Pretraining may not close the gap

3. **Which models to use for pretraining?**
   - GCN, GIN, GAT are ready as graph baselines
   - Transformer is ready as sequence baseline
   - These will serve as downstream evaluation models

---

## 6. Running the Transformer Benchmark

### Quick Test (Dry Run)

```bash
# Single seed, 5 epochs only
python scripts/transformer/run_transformer_benchmark.py --dry_run
```

### Full Benchmark

```bash
# Classification only
python scripts/transformer/run_transformer_benchmark.py \
    --tasks classification \
    --seeds 0,1,2,3,4

# Regression only
python scripts/transformer/run_transformer_benchmark.py \
    --tasks regression \
    --seeds 0,1,2,3,4

# Both tasks
python scripts/transformer/run_transformer_benchmark.py \
    --tasks classification,regression \
    --seeds 0,1,2,3,4
```

### Custom Configuration

```bash
python scripts/transformer/run_transformer_benchmark.py \
    --tasks classification \
    --seeds 0,1,2,3,4 \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --max_smiles_length 128
```

---

## 7. Expected Timeline

### Transformer Benchmark Execution

| Step | Estimated Time | Notes |
|------|---------------|-------|
| Dry run test | ~10 minutes | 1 seed, 5 epochs |
| Tokenizer creation | ~5 minutes | One-time |
| Classification benchmark | ~2-4 hours | 5 seeds × ~30 min each |
| Regression benchmark | ~2-4 hours | 5 seeds × ~30 min each |
| **Total** | **~4-8 hours** | Depends on hardware |

### Hardware Recommendations

- **GPU:** NVIDIA GPU with 8GB+ VRAM (recommended)
- **CPU:** Modern multi-core CPU (fallback, slower)
- **RAM:** 16GB+ recommended

---

## 8. Success Criteria

The Transformer baseline will be considered successful if:

1. ✅ **Implementation Complete:** All code is tested and documented
2. ✅ **Benchmark Executed:** Results for both classification and regression
3. ✅ **Comparison Made:** Clear comparison against classical and graph baselines
4. ✅ **Decision Ready:** Clear recommendation on whether to proceed with pretraining

**Minimum Performance Thresholds:**

| Task | Minimum Viable | Target | Competitive |
|------|---------------|--------|-------------|
| Classification AUC | > 0.85 | > 0.90 | > 0.93 |
| Regression R² | > 0.50 | > 0.65 | > 0.70 |

---

## 9. Risk Mitigation

### Potential Issues

| Issue | Mitigation |
|-------|-----------|
| **Overfitting** (small dataset) | Early stopping, dropout, regularization |
| **Underfitting** (model too small) | Increase model capacity, train longer |
| **Training instability** | Learning rate scheduling, gradient clipping |
| **Poor performance** | Tune hyperparameters, try different architectures |

### Fallback Options

If Transformer performs poorly:
- Try different pooling strategies (mean, max, cls)
- Adjust model capacity (layers, hidden dim)
- Tune learning rate and batch size
- Consider pretrained molecular Transformers (MolBERT, ChemBERTa)

---

## 10. Summary

**What We Have:**
- ✅ Complete classical baseline pipeline
- ✅ Complete graph baseline pipeline (GCN, GIN, GAT)
- ✅ Complete sequence baseline implementation (Transformer)

**What We Need:**
- ⏳ Run Transformer benchmark to get actual numbers

**What Happens Next:**
1. Run Transformer benchmark (classification + regression)
2. Integrate results into unified comparison
3. Analyze which representation performs best
4. Make decision on pretraining strategy
5. Move to ZINC22 pretraining phase with all baselines as reference points

**Project Readiness:**
Once Transformer benchmarking is complete, the project will have:
- **3 representation categories** fully benchmarked
- **Formal baseline models** saved for pretraining comparison
- **Clear direction** for ZINC22 pretraining research
- **Reproducible pipeline** ready for CFFF execution

---

**End of Document**
