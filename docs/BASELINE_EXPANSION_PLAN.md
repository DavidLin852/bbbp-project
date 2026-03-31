# Comprehensive Baseline Expansion Plan

**SMBPP - BBB Permeability Prediction**

**Version:** 1.1
**Date:** 2026-03-31
**Scope:** Non-pretraining baseline suite expansion

---

## Executive Summary

This document outlines a phased plan to expand the current baseline pipeline into a comprehensive non-pretraining baseline suite, covering:

- **Classical/tabular baselines** (4 features × 6 models)
- **Graph neural networks** (GCN, GIN, GAT, MPNN)
- **Sequence models** (SMILES Transformer, optionally LSTM/GRU)
- **Both tasks** (classification + regression)
- **Both splits** (scaffold primary, random reference)
- **Multi-seed evaluation** with unified reporting

**Key principles:**
1. Preserve the current working baseline pipeline (protected)
2. Extend incrementally via new modules, not refactoring existing ones
3. Maintain compatibility with local + Git + CFFF workflow
4. No pretraining, no unconstrained generation

---

## Part 1: Current State Assessment

### 1.1 What Exists (Reusable)

| Component | Path | Status | Reuse Potential |
|-----------|------|--------|-----------------|
| **Data preprocessing** | `src/data/preprocessing.py` | ✅ Complete | 100% - direct reuse |
| **Scaffold split** | `src/data/scaffold_split.py` | ✅ Complete | 100% - supports both scaffold/random |
| **Fingerprints** | `src/features/fingerprints.py` | ✅ Complete | 100% - Morgan (2048), MACCS (167), AtomPairs (1024), FP2 (2048) |
| **Descriptors** | `src/features/descriptors.py` | ✅ Complete | 100% - basic (13), extended (30), all (45) |
| **Graph conversion** | `src/features/graph.py` | ✅ Complete | 90% - may need minor extensions |
| **Classical models** | `src/models/baseline_models.py` | ✅ Complete | 100% - RF, XGB, LGBM, SVM, KNN, LR, NB |
| **Trainer** | `src/train/trainer.py` | ✅ Complete | 80% - classification only, regression not yet integrated |
| **Metrics** | `src/utils/metrics.py` | ✅ Complete | 100% - comprehensive |
| **Configuration** | `src/config/` | ✅ Complete | 100% - frozen dataclasses |
| **GAT backbone** | `src/pretrain/backbone_gat.py` | ⚙️ Exists | Needs wrapping for baseline training loop |
| **Transformer** | `src/transformer/transformer_model.py` | ⚙️ Exists | Fingerprint-based; SMILES version needed for baseline |
| **Regression dataset** | `data/raw/B3DB_regression.tsv` | ✅ Available | 100% |

### 1.2 What Is Missing (Needs Implementation)

| Component | Description | Status |
|-----------|-------------|--------|
| **Regression task support** | Add regression trainer and run logBB benchmarks | Needed |
| **Extended model support** | SVM, LR, KNN not yet in train script | Needed |
| **Extended feature benchmarking** | MACCS, FP2, atom_pairs not benchmarked | Needed |
| **GCN model** | Graph Convolutional Network baseline | Future |
| **GIN model** | Graph Isomorphism Network baseline | Future |
| **MPNN model** | Message Passing Neural Network baseline | Future |
| **SMILES tokenizer** | Character/byte-pair encoding for SMILES | Future |
| **SMILES Transformer** | Transformer on SMILES tokens | Future |
| **LSTM/GRU baseline** | Recurrent SMILES model (optional) | Future |
| **Random split support** | Already exists, needs script integration | Low |
| **Unified reporting** | Cross-benchmark comparison format | Low |

### 1.3 Protected Areas (Do Not Modify)

```
scripts/baseline/
├── 01_preprocess_b3db.py    # PROTECTED
├── 02_compute_features.py   # PROTECTED
└── 03_train_baselines.py    # PROTECTED

src/config.py                # PROTECTED
src/config/baseline.py       # PROTECTED
src/data/preprocessing.py    # PROTECTED (bug fixes only)
src/data/scaffold_split.py   # PROTECTED (bug fixes only)
src/features/fingerprints.py # PROTECTED
src/models/baseline_models.py # PROTECTED
```

---

## Part 2: Target Architecture

### 2.1 Proposed Directory Structure

```
code/
├── scripts/
│   ├── baseline/                 # PROTECTED - current pipeline
│   │   ├── 01_preprocess_b3db.py
│   │   ├── 02_compute_features.py
│   │   └── 03_train_baselines.py
│   │
│   ├── extended/                 # NEW - extended baseline scripts
│   │   ├── 01_preprocess_extended.py    # Support regression + both splits
│   │   ├── 02_compute_features_extended.py  # All feature families
│   │   ├── 03_train_classical.py        # All classical models
│   │   ├── 04_train_graph.py            # GNN models
│   │   └── 05_train_sequence.py         # Transformer/LSTM models
│   │
│   └── analysis/
│       ├── run_full_baseline_matrix.py  # NEW - comprehensive experiment
│       ├── aggregate_all_results.py     # NEW - unified aggregation
│       └── generate_benchmark_report.py # NEW - unified report
│
├── src/
│   ├── data/                     # Existing (protected)
│   ├── features/                 # Existing (protected)
│   │   └── graph.py              # Existing, may add graph_dataset.py
│   │
│   ├── models/                   # Extended
│   │   ├── baseline_models.py    # PROTECTED
│   │   ├── regression_models.py  # NEW - regression wrappers
│   │   ├── graph_models.py       # NEW - GCN, GIN, GAT, MPNN
│   │   └── sequence_models.py    # NEW - SMILES models
│   │
│   ├── train/                    # Extended
│   │   ├── trainer.py            # Existing (protected)
│   │   ├── trainer_regression.py # NEW
│   │   ├── trainer_graph.py      # NEW
│   │   └── trainer_sequence.py   # NEW
│   │
│   ├── features/                 # Extended
│   │   ├── fingerprints.py       # PROTECTED
│   │   ├── descriptors.py        # PROTECTED
│   │   ├── graph.py              # PROTECTED
│   │   └── smiles_tokenizer.py   # NEW
│   │
│   └── evaluate/                 # Extended
│       ├── comparison.py         # Existing
│       ├── report.py             # Existing
│       └── benchmark_matrix.py   # NEW - unified comparison
│
└── artifacts/
    └── reports/
        ├── baseline_summary.csv          # Existing
        ├── extended_benchmark_full.csv   # NEW - all results
        └── figures/                      # NEW - comparison plots
```

### 2.2 New Module Interfaces

#### `src/models/graph_models.py`
```python
class GNNModel:
    """Unified interface for GNN models."""
    def __init__(self, model_type: str, config: GNNConfig):
        # model_type: "gcn", "gin", "gat", "mpnn"
        pass

    def fit(self, train_data, val_data, config) -> dict: pass
    def predict(self, data) -> np.ndarray: pass
    def save(self, path: str): pass
    def load(self, path: str): pass
```

#### `src/models/sequence_models.py`
```python
class SMILESModel:
    """Unified interface for SMILES-based models."""
    def __init__(self, model_type: str, config: SequenceConfig):
        # model_type: "transformer", "lstm"
        pass

    def fit(self, train_data, val_data, config) -> dict: pass
    def predict(self, data) -> np.ndarray: pass
    def save(self, path: str): pass
    def load(self, path: str): pass
```

#### `src/features/smiles_tokenizer.py`
```python
class SMILESTokenizer:
    """Character-level or BPE tokenizer for SMILES."""
    def __init__(self, vocab: str = "char", max_length: int = 128):
        pass

    def encode(self, smiles: str) -> List[int]: pass
    def decode(self, tokens: List[int]) -> str: pass
    def batch_encode(self, smiles_list: List[str]) -> torch.Tensor: pass
```

---

## Part 3: Baseline Matrix

### 3.1 Complete Baseline Coverage

#### Classical/Tabular Baselines

Each feature family is an **independent** baseline input. Feature concatenation is optional exploratory work and NOT part of the formal baseline suite.

| Feature | Dim | RF | XGB | LGBM | SVM | LR | KNN | Benchmark Status |
|---------|-----|----|----|------|-----|----|----|----------------|
| morgan | 2048 | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ✅ Benchmarked |
| descriptors_basic | 13 | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ✅ Benchmarked |
| maccs | 167 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⚙️ Planned |
| fp2 | 2048 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⚙️ Planned |
| **Subtotal (target)** | | **4** | **4** | **4** | **4** | **4** | **4** | **24** |

**Currently working:** morgan + RF/XGB/LGBM, descriptors_basic + RF/XGB/LGBM (6 experiments)
**To implement:** maccs, fp2, SVM, LR, KNN for all 4 features (18 more)

#### Graph Neural Network Baselines

| Model | Classification | Regression | Notes |
|-------|---------------|------------|-------|
| GCN | ⬜ | ⬜ | Basic graph convolution |
| GIN | ⬜ | ⬜ | Graph isomorphism |
| GAT | ⬜ | ⬜ | Attention (backbone exists) |
| MPNN | ⬜ | ⬜ | Message passing |
| **Subtotal** | 4 | 4 | **8** |

#### Sequence Baselines

| Model | Classification | Regression | Notes |
|-------|---------------|------------|-------|
| SMILES Transformer | ⬜ | ⬜ | Char/token-level |
| LSTM (optional) | ⬜ | ⬜ | Recurrent baseline |
| **Subtotal** | 2 | 2 | **4** |

### 3.2 Experiment Matrix Summary

| Category | Features/Models | Tasks | Splits | Seeds | Total Experiments |
|----------|-----------------|-------|--------|-------|-------------------|
| Classical | 24 | 2 | 2 | 3 | 288 |
| Graph | 4 | 2 | 2 | 3 | 48 |
| Sequence | 2 | 2 | 2 | 3 | 24 |
| **Total** | | | | | **360** |

Note: Not all combinations need to run initially. Focus on scaffold split + classification first.

---

## Part 4: Phased Implementation Roadmap

### Phase 0: Foundation (1-2 days)
**Goal:** Extend existing infrastructure without touching protected code.

**Tasks:**
1. Create `src/models/regression_models.py` - wrappers for regression
2. Create `src/train/trainer_regression.py`
3. Add random split flag support in extended scripts
4. Test regression on existing classical models (RF with Morgan)
5. Verify FP2 fingerprint support in extended feature script

**Files to create:**
```
src/models/regression_models.py
src/train/trainer_regression.py
scripts/extended/01_preprocess_extended.py
scripts/extended/02_compute_features_extended.py
```

**Validation:**
- Run RF on logBB regression with Morgan features
- Compare scaffold vs random split results

**Risk to protected code:** None (all new files)

---

### Phase 1: Complete Classical Baselines (2-3 days)
**Goal:** Full coverage of classical models with all independent feature families.

**Tasks:**
1. Extend train script to support SVM, LR, KNN (MACCS, FP2, and atom_pairs are already implemented in `src/features/fingerprints.py`)
2. Create unified `03_train_classical.py` script that runs all 4 features × 6 models
3. Add regression support for all model types
4. Create analysis script for classical baseline matrix

**Note:** `maccs`, `fp2`, and `atom_pairs` are already implemented in `src/features/fingerprints.py`. The work is extending the training script and running benchmarks.

**Files to create:**
```
scripts/extended/03_train_classical.py
scripts/analysis/run_classical_matrix.py
```

**Baseline matrix for Phase 1:** 4 features × 6 models × 2 tasks × 2 splits × 3 seeds = **288 experiments**

Note: Combined/concatenated features are optional exploratory and NOT included in the formal baseline matrix.

**Validation:**
- Verify RF+Morgan matches existing baseline (AUC ~0.94)
- Run full classical matrix on 3 seeds

**Risk to protected code:** None

---

### Phase 2: Graph Neural Network Baselines (3-4 days)
**Goal:** Add GCN, GIN, GAT, MPNN baselines.

**Tasks:**
1. Create `src/models/graph_models.py` with unified interface
2. Implement GCN (simplest first)
3. Implement GIN (for expressiveness)
4. Adapt existing GAT backbone into baseline
5. Implement MPNN-style model
6. Create `src/train/trainer_graph.py`
7. Create `scripts/extended/04_train_graph.py`
8. Test on both classification and regression

**Files to create:**
```
src/models/graph_models.py
src/train/trainer_graph.py
scripts/extended/04_train_graph.py
scripts/analysis/run_graph_matrix.py
```

**GNN Architecture Details:**
```
GCN:  3 layers, hidden=128, dropout=0.3
GIN:  3 layers, hidden=128, MLP for update
GAT:  3 layers, hidden=64, heads=4 (adapt from backbone_gat.py)
MPNN: 3 layers, hidden=128, edge features
```

**Validation:**
- Compare GAT against archived GAT results if available
- Verify graph models achieve reasonable performance (AUC > 0.85)

**Risk to protected code:** Low (new modules, graph.py unchanged)

---

### Phase 3: Sequence Baselines (3-4 days)
**Goal:** Add SMILES Transformer and optional LSTM baseline.

**Tasks:**
1. Create `src/features/smiles_tokenizer.py`
2. Create `src/models/sequence_models.py`
3. Implement character-level Transformer
4. Optionally implement LSTM baseline
5. Create `src/train/trainer_sequence.py`
6. Create `scripts/extended/05_train_sequence.py`
7. Test on both classification and regression

**Files to create:**
```
src/features/smiles_tokenizer.py
src/models/sequence_models.py
src/train/trainer_sequence.py
scripts/extended/05_train_sequence.py
scripts/analysis/run_sequence_matrix.py
```

**Sequence Model Details:**
```
SMILES Transformer:
  - Vocab: ~64 characters (or BPE with 256 tokens)
  - Max length: 128
  - Layers: 4, heads: 8, hidden: 256

LSTM (optional):
  - Embedding: 128
  - Hidden: 256, layers: 2, bidirectional
```

**Validation:**
- Verify tokenizer handles B3DB SMILES correctly
- Compare Transformer vs classical baselines

**Risk to protected code:** Low (new modules)

---

### Phase 4: Unified Reporting & Analysis (1-2 days)
**Goal:** Create comprehensive benchmark comparison.

**Tasks:**
1. Create `src/evaluate/benchmark_matrix.py`
2. Create unified aggregation script
3. Generate comparison tables and plots
4. Update documentation

**Files to create:**
```
src/evaluate/benchmark_matrix.py
scripts/analysis/run_full_baseline_matrix.py
scripts/analysis/aggregate_all_results.py
scripts/analysis/generate_benchmark_report.py
```

**Deliverables:**
- `artifacts/reports/extended_benchmark_full.csv`
- `artifacts/reports/figures/comparison_plots/`
- Updated `README.md` and `CLAUDE.md`

**Risk to protected code:** None

---

## Part 5: Code Module Mapping

### 5.1 Classical Baselines

All features are independent. Combined/concatenated features are optional exploratory only.

| Baseline | Feature Module | Model Module | Trainer | Script |
|----------|----------------|--------------|---------|--------|
| RF + Morgan | `features/fingerprints.py` | `models/baseline_models.py` | `train/trainer.py` | `baseline/03_train_baselines.py` |
| RF + MACCS | `features/fingerprints.py` | `models/baseline_models.py` | `train/trainer.py` | `extended/03_train_classical.py` |
| RF + FP2 | `features/fingerprints.py` | `models/baseline_models.py` | `train/trainer.py` | `extended/03_train_classical.py` |
| SVM + Morgan | `features/fingerprints.py` | `models/baseline_models.py` | `train/trainer.py` | `extended/03_train_classical.py` |
| LR + Morgan | `features/fingerprints.py` | `models/baseline_models.py` | `train/trainer.py` | `extended/03_train_classical.py` |
| Regression RF | `features/fingerprints.py` | `models/regression_models.py` (NEW) | `train/trainer_regression.py` (NEW) | `extended/03_train_classical.py` |

### 5.2 Graph Baselines

| Baseline | Graph Module | Model Module | Trainer | Script |
|----------|--------------|--------------|---------|--------|
| GCN (cls) | `features/graph.py` | `models/graph_models.py` (NEW) | `train/trainer_graph.py` (NEW) | `extended/04_train_graph.py` |
| GCN (reg) | `features/graph.py` | `models/graph_models.py` (NEW) | `train/trainer_graph.py` (NEW) | `extended/04_train_graph.py` |
| GIN (cls) | `features/graph.py` | `models/graph_models.py` (NEW) | `train/trainer_graph.py` (NEW) | `extended/04_train_graph.py` |
| GAT (cls) | `features/graph.py` | `models/graph_models.py` (NEW) | `train/trainer_graph.py` (NEW) | `extended/04_train_graph.py` |
| MPNN (cls) | `features/graph.py` | `models/graph_models.py` (NEW) | `train/trainer_graph.py` (NEW) | `extended/04_train_graph.py` |

### 5.3 Sequence Baselines

| Baseline | Tokenizer Module | Model Module | Trainer | Script |
|----------|------------------|--------------|---------|--------|
| SMILES Transformer (cls) | `features/smiles_tokenizer.py` (NEW) | `models/sequence_models.py` (NEW) | `train/trainer_sequence.py` (NEW) | `extended/05_train_sequence.py` |
| SMILES Transformer (reg) | `features/smiles_tokenizer.py` (NEW) | `models/sequence_models.py` (NEW) | `train/trainer_sequence.py` (NEW) | `extended/05_train_sequence.py` |
| LSTM (cls) | `features/smiles_tokenizer.py` (NEW) | `models/sequence_models.py` (NEW) | `train/trainer_sequence.py` (NEW) | `extended/05_train_sequence.py` |

---

## Part 6: Risk Assessment

### 6.1 Risks to Protected Pipeline

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Accidental modification of protected files | Low | High | Use new `scripts/extended/` directory |
| Configuration conflicts | Low | Medium | Separate config extensions in `src/config/extended.py` |
| Dependency version conflicts | Low | Medium | Test in isolated environment first |
| Graph feature changes breaking existing code | Low | Medium | Do not modify `features/graph.py`, create wrapper |

### 6.2 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GNN training instability | Medium | Medium | Use proven architectures, early stopping |
| SMILES tokenization edge cases | Medium | Low | Handle special characters, test on B3DB |
| Memory issues with large graphs | Medium | Low | Batch size tuning, gradient accumulation |
| CFFF compatibility issues | Low | Medium | Test scripts on cluster early |

### 6.3 Scope Creep Risks

| Risk | Mitigation |
|------|------------|
| Temptation to add pretraining | Explicitly defer, document for Phase 5+ |
| Temptation to add generation | Explicitly defer, keep VAE/GAN modules separate |
| Over-engineering abstractions | Keep interfaces simple, prefer copy over inheritance |

---

## Part 7: Implementation Priority

### Recommended Order

1. **Phase 0 (Foundation)** - Required for all subsequent work
2. **Phase 1 (Classical)** - Highest ROI, builds on working baseline
3. **Phase 2 (Graph)** - Different representation, valuable comparison
4. **Phase 3 (Sequence)** - Complementary approach, more complex
5. **Phase 4 (Reporting)** - Final integration and documentation

### First Three Deliverables (Concrete)

**Deliverable 1 (Phase 0):**
- `src/models/regression_models.py` - Regression wrappers
- `src/train/trainer_regression.py` - Regression training loop
- Working RF regression on logBB with scaffold split

**Deliverable 2 (Phase 1 partial):**
- All 4 features (morgan, maccs, fp2, descriptors_basic)
- All 6 models (rf, xgb, lgbm, svm, lr, knn)
- Classification only, scaffold split
- 4 × 6 × 3 = 72 experiments

**Deliverable 3 (Phase 1 complete):**
- Add regression task
- Add random split
- Full classical matrix: 4 × 6 × 2 × 2 × 3 = 288 experiments

---

## Part 8: Out of Scope

The following are **explicitly out of scope** for this baseline expansion:

### 8.1 Pretraining
- No ZINC22/PubChem pretraining
- No self-supervised learning
- No SMARTS-based pretraining
- All models trained from scratch on B3DB

### 8.2 Generation
- No VAE-based molecule generation
- No GAN-based molecule generation
- No diffusion models
- No de novo design

### 8.3 Advanced Features
- No ensemble methods (stacking, voting) - future phase
- No hyperparameter optimization (Optuna, etc.) - use defaults
- No interpretability analysis (SHAP, attention) - future phase
- No uncertainty quantification
- No active learning

### 8.4 Infrastructure
- No database integration
- No web interface
- No API endpoints
- No containerization changes

### 8.5 Data Expansion
- No external dataset integration (ChEMBL, Metrabase)
- No multi-source learning
- B3DB classification and regression only

---

## Part 9: Success Criteria

### Phase 0 Success
- [ ] Regression task works with existing classical models
- [ ] FP2 feature support verified in extended scripts
- [ ] No changes to protected files

### Phase 1 Success
- [ ] All 4 independent features × 6 models = 24 combinations work
- [ ] Both classification and regression tasks work
- [ ] Both scaffold and random splits work
- [ ] Results match existing baseline for overlapping configs

### Phase 2 Success
- [ ] GCN, GIN, GAT, MPNN all train without errors
- [ ] GNN models achieve AUC > 0.85 on classification
- [ ] Graph training stable (no NaN losses)

### Phase 3 Success
- [ ] SMILES Transformer trains on B3DB
- [ ] Tokenizer handles all SMILES in dataset
- [ ] Sequence models achieve reasonable performance

### Phase 4 Success
- [ ] Unified report with all baseline results
- [ ] Comparison tables and plots generated
- [ ] Documentation updated

---

## Appendix A: Estimated Effort

| Phase | Days | Complexity |
|-------|------|------------|
| Phase 0: Foundation | 1-2 | Low |
| Phase 1: Classical | 2-3 | Low-Medium |
| Phase 2: Graph | 3-4 | Medium |
| Phase 3: Sequence | 3-4 | Medium |
| Phase 4: Reporting | 1-2 | Low |
| **Total** | **10-15** | |

---

## Appendix B: Dependencies

### Current Dependencies (already satisfied)
- scikit-learn
- xgboost
- lightgbm
- rdkit
- torch
- torch-geometric
- pandas
- numpy

### No New Dependencies Required
All planned baselines can be implemented with existing dependencies.

---

## Appendix C: File Creation Checklist

### Phase 0
- [ ] `src/models/regression_models.py`
- [ ] `src/train/trainer_regression.py`
- [ ] `scripts/extended/01_preprocess_extended.py`
- [ ] `scripts/extended/02_compute_features_extended.py`

### Phase 1
- [ ] `scripts/extended/03_train_classical.py`
- [ ] `scripts/analysis/run_classical_matrix.py`

### Phase 2
- [ ] `src/models/graph_models.py`
- [ ] `src/train/trainer_graph.py`
- [ ] `scripts/extended/04_train_graph.py`
- [ ] `scripts/analysis/run_graph_matrix.py`

### Phase 3
- [ ] `src/features/smiles_tokenizer.py`
- [ ] `src/models/sequence_models.py`
- [ ] `src/train/trainer_sequence.py`
- [ ] `scripts/extended/05_train_sequence.py`
- [ ] `scripts/analysis/run_sequence_matrix.py`

### Phase 4
- [ ] `src/evaluate/benchmark_matrix.py`
- [ ] `scripts/analysis/run_full_baseline_matrix.py`
- [ ] `scripts/analysis/aggregate_all_results.py`
- [ ] `scripts/analysis/generate_benchmark_report.py`

---

**Document End**

*This plan should be reviewed before implementation begins. Once approved, Phase 0 can start immediately.*
