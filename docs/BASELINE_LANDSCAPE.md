# Current Baseline Landscape

**Last Updated:** 2026-04-03
**Project Status:** Classical and Graph baselines complete, moving to Sequence baseline

---

## Overview

This document provides a concise summary of the current baseline landscape for BBB permeability prediction on the B3DB dataset (Groups A+B, scaffold split).

### Baseline Categories

1. **Classical Baselines** - Traditional ML on molecular fingerprints/descriptors
2. **Graph Baselines** - Graph Neural Networks (GNNs) on molecular graphs
3. **Sequence Baselines** - Transformer models on SMILES sequences (IN PROGRESS)

---

## 1. Classical Baselines

### Status: ✅ COMPLETE

**Configuration:**
- Dataset: B3DB Groups A+B (n ≈ 1,900 after preprocessing)
- Split: Scaffold stratified split (80:10:10 train/val/test)
- Seeds: 3 seeds (0, 1, 2)
- Evaluation: AUC, F1 (classification); R², RMSE, MAE (regression)

**Feature Representations (6 independent baselines):**

| Feature | Dimension | Type | Status |
|---------|-----------|------|--------|
| Morgan (ECFP4) | 2048 | Circular fingerprint | ✅ Benchmarked |
| Descriptors (basic) | 13 | Physicochemical | ✅ Benchmarked |
| MACCS | 167 | Structural keys | ⚙️ Implemented, not benchmarked |
| FP2 | 2048 | Daylight fingerprint | ⚙️ Implemented, not benchmarked |
| AtomPairs | 1024 | Atom pair fingerprints | ⚙️ Implemented, not benchmarked |
| Descriptors (extended) | 30 | Extended physicochemical | ⚙️ Implemented, not benchmarked |

**Model Families (3 implemented, 3 pending):**

| Model | Status | Best Performance |
|-------|--------|------------------|
| Random Forest | ✅ Benchmarked | **AUC: 0.9401 ± 0.0454** (Morgan) |
| XGBoost | ✅ Benchmarked | AUC: 0.9198 ± 0.0674 (Morgan) |
| LightGBM | ✅ Benchmarked | AUC: 0.9195 ± 0.0619 (Morgan) |
| SVM | ⚙️ Implemented | Not benchmarked |
| Logistic Regression | ⚙️ Implemented | Not benchmarked |
| KNN | ⚙️ Implemented | Not benchmarked |

### Classical Baseline Leaderboard (Classification)

| Rank | Feature | Model | Test AUC | Test F1 |
|------|---------|-------|----------|---------|
| 1 | Morgan | RF | **0.9401 ± 0.0454** | 0.9391 ± 0.0270 |
| 2 | Morgan | XGB | 0.9198 ± 0.0674 | 0.9335 ± 0.0310 |
| 3 | Morgan | LGBM | 0.9195 ± 0.0619 | 0.9363 ± 0.0331 |
| 4 | Descriptors_basic | RF | 0.9159 ± 0.0730 | 0.9397 ± 0.0270 |
| 5 | Descriptors_basic | XGB | 0.9115 ± 0.0712 | 0.9370 ± 0.0308 |
| 6 | Descriptors_basic | LGBM | 0.9114 ± 0.0815 | 0.9395 ± 0.0298 |

**Key Findings:**
- **Best classical model:** Random Forest + Morgan fingerprints
- Morgan fingerprints consistently outperform physicochemical descriptors
- High variance across seeds suggests dataset sensitivity
- All models show signs of overfitting (train AUC > 0.99)

---

## 2. Graph Baselines

### Status: ✅ COMPLETE

**Configuration:**
- Same scaffold splits as classical baselines
- Seeds: 5 seeds (0, 1, 2, 3, 4)
- Graph featurization: 22-dim node features, 8-dim edge features
- Training: 300 epochs max, early stopping (patience=30)
- Hardware: GPU-first with CPU fallback

**GNN Architectures (3 models):**

| Model | Architecture | Key Features | Status |
|-------|-------------|--------------|--------|
| GCN | Graph Convolutional Network | Spectral-based, 3 layers | ✅ Benchmarked |
| GIN | Graph Isomorphism Network | MLP+SUM, 3 layers | ✅ Benchmarked |
| GAT | Graph Attention Network | Multi-head (4 heads), 3 layers | ✅ Benchmarked |

### Graph Baseline Leaderboard

**Classification (Test AUC):**

| Rank | Model | Test AUC | Test F1 | Train AUC |
|------|-------|----------|---------|-----------|
| 1 | GAT | **0.9356 ± 0.0314** | 0.9231 ± 0.0185 | 0.9753 ± 0.0160 |
| 2 | GIN | 0.9271 ± 0.0349 | 0.9269 ± 0.0156 | 0.9538 ± 0.0112 |
| 3 | GCN | 0.9255 ± 0.0384 | 0.9207 ± 0.0197 | 0.9509 ± 0.0135 |

**Regression (Test R²):**

| Rank | Model | Test R² | Test RMSE | Test MAE |
|------|-------|---------|-----------|----------|
| 1 | GIN | **0.7062 ± 0.0473** | 0.5472 ± 0.0410 | 0.4408 ± 0.0228 |
| 2 | GAT | 0.6408 ± 0.0357 | 0.5841 ± 0.0363 | 0.4620 ± 0.0280 |
| 3 | GCN | 0.3237 ± 0.1193 | 0.6646 ± 0.0549 | 0.5050 ± 0.0368 |

**Key Findings:**
- **Best graph model:** GAT for classification, GIN for regression
- GAT classification (0.9356) is **competitive with best classical** (0.9401)
- GNNs show lower overfitting than classical models (smaller train-test gap)
- High variance persists across seeds

---

## 3. Sequence Baselines

### Status: 🔄 IN PROGRESS

**Planned Configuration:**
- Representation: SMILES strings as token sequences
- Model: Transformer encoder (architecture TBD)
- Training: Same scaffold splits as classical/graph baselines
- Seeds: 5 seeds (matching GNN evaluation)
- Evaluation: Same metrics (AUC, F1, R², RMSE, MAE)

**Implementation Status:**

| Component | Status |
|-----------|--------|
| SMILES tokenization | ⏳ To implement |
| Transformer architecture | ⏳ To implement |
| Training script | ⏳ To implement |
| Benchmark evaluation | ⏳ To implement |
| Results integration | ⏳ To implement |

---

## 4. Cross-Baseline Comparison

### Performance Summary (Classification)

| Baseline Category | Best Model | Test AUC | Key Strength |
|-------------------|------------|----------|--------------|
| Classical | RF + Morgan | **0.9401 ± 0.0454** | Strongest overall, simple |
| Graph | GAT | 0.9356 ± 0.0314 | Lower variance, structural |
| Sequence | - | - | Pending |

### Performance Summary (Regression)

| Baseline Category | Best Model | Test R² | Key Strength |
|-------------------|------------|----------|--------------|
| Classical | (Pending full benchmark) | - | - |
| Graph | GIN | **0.7062 ± 0.0473** | Best regression performance |
| Sequence | - | - | Pending |

### Key Insights

1. **Classical models currently lead** in classification (RF + Morgan: 0.9401)
2. **GNNs are highly competitive** (GAT: 0.9356, within 0.5% of RF)
3. **GNNs show better generalization** (smaller train-test gap)
4. **Graph representation matters** - GIN excels at regression, GAT at classification
5. **High variance across seeds** - dataset size is a limitation

---

## 5. What's Already Benchmarked

### Completed Benchmarks

**Classical:**
- ✅ Morgan + RF/XGB/LGBM (6 experiments)
- ✅ Descriptors_basic + RF/XGB/LGBM (6 experiments)
- ✅ Scaffold split evaluation (3 seeds)
- ✅ Aggregated results and summary reports

**Graph:**
- ✅ GCN classification + regression (2 tasks × 5 seeds)
- ✅ GIN classification + regression (2 tasks × 5 seeds)
- ✅ GAT classification + regression (2 tasks × 5 seeds)
- ✅ Scaffold split evaluation (5 seeds)
- ✅ GNN benchmark summary reports

### Pending Benchmarks

**Classical:**
- ⏳ MACCS, FP2, AtomPairs, Descriptors_extended
- ⏳ SVM, Logistic Regression, KNN
- ⏳ Full 18-experiment matrix

**Sequence:**
- ⏳ SMILES Transformer architecture selection
- ⏳ Tokenization strategy
- ⏳ Training and evaluation
- ⏳ Cross-baseline comparison

---

## 6. Current Strongest Models

### Classification Task

**Overall Best: Random Forest + Morgan**
- Test AUC: 0.9401 ± 0.0454
- Test F1: 0.9391 ± 0.0270
- Strength: Consistent, interpretable, simple

**Runner-up: GAT**
- Test AUC: 0.9356 ± 0.0314
- Test F1: 0.9231 ± 0.0185
- Strength: Lower variance, structural learning

### Regression Task

**Overall Best: GIN**
- Test R²: 0.7062 ± 0.0473
- Test RMSE: 0.5472 ± 0.0410
- Strength: Best regression performance across all baselines

---

## 7. Next Steps

### Immediate Priority: Sequence Baseline

1. **Implement SMILES Transformer baseline**
   - Design architecture (encoder-only, pre-training ready)
   - Implement tokenization and data pipeline
   - Train on B3DB with scaffold splits
   - Evaluate with same metrics as classical/graph

2. **Prepare for pre-training reference**
   - Ensure GCN, GIN, GAT are saved as formal baselines
   - Document their architectures and checkpoints
   - These will serve as comparison points for ZINC22 pre-training

3. **After sequence baseline is complete:**
   - Cross-baseline comparison analysis
   - Move to ZINC22 pre-training phase
   - Compare pretrained vs non-pretrained models

---

## 8. File Locations

### Classical Baselines

- Scripts: `scripts/baseline/01_preprocess_b3db.py`, `02_compute_features.py`, `03_train_baselines.py`
- Results: `artifacts/reports/benchmark_summary.csv`
- Models: `artifacts/models/baselines/`

### Graph Baselines

- Scripts: `scripts/gnn/run_gnn_benchmark.py`
- Models: `src/gnn/models.py` (GCN, GIN, GAT)
- Results: `artifacts/reports/gnn/gnn_benchmark_scaffold_*.csv`
- Checkpoints: `artifacts/models/gnn/`

### Sequence Baselines (Pending)

- Scripts: `scripts/transformer/` (to create)
- Models: `src/transformer/` (to create)
- Results: `artifacts/reports/transformer/` (to create)
- Checkpoints: `artifacts/models/transformer/` (to create)

---

**Note:** All baselines use the same scaffold splits for fair comparison. The sequence baseline will follow the same evaluation protocol to ensure comparability.
