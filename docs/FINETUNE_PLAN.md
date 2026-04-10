# Fine-tuning Experiment Plan (Stage 4)

## Goal

Systematic evaluation: **which pretrained backbone × which downstream model × which feature representation** gives the best BBB prediction. No redundant "from scratch" runs — baseline results serve as controls.

## Existing Controls (Stage 1-2 Baselines, Already Done)

These are our no-pretrain baselines. **Do NOT re-run.**

| Category | Best Classification | Best Regression |
|----------|--------------------|-----------------|
| Classical (24 combos) | MACCS + LightGBM: AUC 0.9535 | Desc_basic + RF: R² 0.4488 |
| GNN (3 models) | GAT: AUC 0.9356 | GIN: R² 0.7062 |
| Transformer (1 model) | AUC 0.8820 | R² 0.1911 |

## Design: Three Dimensions

### Dimension 1: Pretrained Backbones (17)

| # | ID | Strategy | Backbone | Samples | Epochs | Dim |
|---|-----|----------|----------|---------|--------|-----|
| 1 | P_E10_GIN_100K | Property | GIN | 100K | 10 | 128 |
| 2 | P_E10_GIN_500K | Property | GIN | 500K | 10 | 128 |
| 3 | P_E10_GIN_1M | Property | GIN | 1M | 10 | 128 |
| 4 | P_E10_GIN_2M | Property | GIN | 2M | 10 | 128 |
| 5 | P_E10_GIN_5M | Property | GIN | 5M | 10 | 128 |
| 6 | P_E10_GAT_1M | Property | GAT | 1M | 10 | 128 |
| 7 | P_E20_GIN_256_5M | Property | GIN | 5M | 20 | 256 |
| 8 | D_E10_GIN_100K | Denoising | GIN | 100K | 10 | 128 |
| 9 | D_E10_GIN_1M | Denoising | GIN | 1M | 10 | 128 |
| 10 | D_E10_GIN_5M | Denoising | GIN | 5M | 10 | 128 |
| 11 | D_E10_GAT_1M | Denoising | GAT | 1M | 10 | 128 |
| 12 | D_E20_GIN_256_5M | Denoising | GIN | 5M | 20 | 256 |
| 13 | T_E10_TRANS_100K | MLM | Transformer | 100K | 10 | 256 |
| 14 | T_E10_TRANS_1M | MLM | Transformer | 1M | 10 | 256 |
| 15 | T_E10_TRANS_5M | MLM | Transformer | 5M | 10 | 256 |
| 16 | T_E10_TRANS_1M_L6 | MLM | Transformer | 1M | 10 | 256 |
| 17 | T_E20_TRANS_512_5M | MLM | Transformer | 5M | 20 | 512 |

### Dimension 2: Fine-tuning Approaches (4)

| Approach | How | Applicable Pretrains |
|----------|-----|---------------------|
| **A. GNN Fine-tune** | Load backbone → add head → full fine-tune | #1-12 (GIN/GAT) |
| **B. Embedding + Classical** | Extract graph/sequence embeddings → feed to LightGBM | #1-17 (all) |
| **C. Embedding + Feature Concat** | Embedding ⊕ fingerprint/descriptor → LightGBM | #1-17 (all) |
| **D. Transformer Fine-tune** | Load encoder → add head → fine-tune | #13-17 (Transformer) |

### Dimension 3: Feature Sets (for approach C)

| Feature | Dim | CLI Flag |
|---------|-----|----------|
| Morgan | 2048 | `morgan` |
| MACCS | 167 | `maccs` |
| FP2 | 2048 | `fp2` |
| Descriptors (basic) | 13 | `descriptors_basic` |

## Full Experiment Matrix

### Group A: GNN Fine-tuning (Backbone → Head → End-to-end)

| # | Pretrain | Backbone | Task | Count |
|---|----------|----------|------|-------|
| A1-A12 | #1-12 (each) | matches pretrain | CLS | 12 |
| A13-A24 | #1-12 (each) | matches pretrain | REG | 12 |

**Subtotal: 24 experiments**

### Group B: Embedding → LightGBM (Embedding only)

Extract graph-level or sequence-level embeddings from frozen pretrained models, train LightGBM.

| # | Pretrain | Embedding Dim | Tasks | Count |
|---|----------|---------------|-------|-------|
| B1-B17 | #1-17 (each) | 128/256/512 | CLS | 17 |
| B18-B34 | #1-17 (each) | 128/256/512 | REG | 17 |

**Subtotal: 34 experiments**

### Group C: Embedding + Feature Concat → LightGBM

Concatenate pretrained embeddings with classical features, train LightGBM.

| # | Pretrain | + Feature | Tasks | Count |
|---|----------|-----------|-------|-------|
| C1-C68 | #1-17 × 4 features | Morgan/MACCS/FP2/Desc | CLS | 17×4 = 68 |
| C69-C136 | #1-17 × 4 features | Morgan/MACCS/FP2/Desc | REG | 17×4 = 68 |

**Subtotal: 136 experiments**

### Group D: Transformer Fine-tuning

| # | Pretrain | Task | Count |
|---|----------|------|-------|
| D1-D5 | #13-17 (each) | CLS | 5 |
| D6-D10 | #13-17 (each) | REG | 5 |

**Subtotal: 10 experiments**

## Total Count

| Group | Description | Experiments |
|-------|-------------|-------------|
| A | GNN Fine-tune | 24 |
| B | Embedding → LightGBM | 34 |
| C | Embedding + Feature → LightGBM | 136 |
| D | Transformer Fine-tune | 10 |
| **Total** | | **204** |

× 5 seeds = **1020 runs**

## Priority / Phased Execution

Group C (136) 是最大的，且可能边际收益不大。建议分阶段：

### Phase 1 (Core, ~68 runs)
- Group A: GNN Fine-tune (24) — 最直接，看预训练是否提升GNN
- Group D: Transformer Fine-tune (10) — 同上
- Group B: Embedding → LightGBM (34) — 看预训练embedding单独的效果

### Phase 2 (Extended, ~136 runs)
- Group C: Embedding + Feature concat (136) — 看预训练+传统特征的协同效果

Phase 1完成后做一次分析，如果发现某些embedding效果明显，再决定Phase 2要不要全部跑。

## Analysis Plan

### Analysis 1: Same Backbone, Different Pretrain
```
GIN-128d (CLS): scratch(baseline=0.9271) → P_100K → P_500K → P_1M → P_2M → P_5M → D_100K → D_1M → D_5M
```
→ Which pretrain strategy and scale gives the biggest boost?

### Analysis 2: Same Pretrain, Different Downstream
```
P_E10_GIN_5M: GNN fine-tune vs Embedding+LGBM vs Embedding+MACCS+LGBM
```
→ Which downstream approach benefits most from this pretrain?

### Analysis 3: Feature Synergy (Group C)
```
LGBM(MACCS only) vs LGBM(embedding only) vs LGBM(embedding + MACCS)
```
→ Do pretrained embeddings complement classical features?

### Analysis 4: Final Selection for Ensemble
Pick top-K diverse models for stacking/voting:
- Classical: best baseline (MACCS + LightGBM, AUC 0.9535)
- GNN: best pretrained GIN (TBD)
- GNN: best pretrained GAT (TBD)
- Transformer: best pretrained encoder (TBD)
- Hybrid: best embedding + feature model (TBD)

## Training Configuration

- **Dataset:** B3DB Groups A+B, scaffold split (same splits as baselines)
- **Seeds:** 5 (seed 0-4, same as baselines)
- **GNN fine-tune:** 50 epochs, batch_size=64, AdamW lr=1e-4, cosine annealing, early stopping patience=10
- **Transformer fine-tune:** 30 epochs, batch_size=128, AdamW lr=5e-4, cosine annealing, early stopping patience=10
- **LightGBM:** same hyperparameters as baseline (from `03_train_baselines.py`)
- **Metrics:** AUC, F1, Acc (cls) / R², RMSE (reg)

## Output

Per-run CSV:
```
exp_id, pretrain_id, approach, model, feature, task, seed, test_auc, test_f1, test_acc, test_r2, test_rmse
```

Aggregated CSV (for final report):
```
exp_id, pretrain_id, approach, model, feature, task, test_auc_mean, test_auc_std, test_f1_mean, test_f1_std, test_r2_mean, test_r2_std, test_rmse_mean, test_rmse_std
```
