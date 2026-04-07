# Baseline Results — Scaffold Split (2026-04-02)

## Experiment Summary

Multi-seed classical ML baselines completed on the B3DB dataset using scaffold splitting (80/10/10, seeds 0–4). Five feature families and six model families were evaluated across both classification and regression tasks.

**Dataset:** B3DB Groups A+B, scaffold split
**Seeds:** 0, 1, 2, 3, 4
**Metrics:** AUC (classification), R² (regression); reported as mean ± std across seeds

---

## Classification Results

| Rank | Feature | Model | Test AUC (mean ± std) |
|------|---------|-------|----------------------|
| 1 | maccs | lgbm | 0.9535 |
| 2 | maccs | rf | 0.9504 |
| 3 | morgan | rf | 0.9504 |
| 4 | maccs | xgb | 0.9496 |
| 5 | fp2 | xgb | 0.9459 |

### Interpretation

- All top-5 combinations exceed **0.94 AUC**, indicating strong predictive performance for BBB permeability classification.
- MACCS structural keys (167 bits) slightly outperform Morgan fingerprints (2048 bits), possibly because MACCS encodes specific substructures more relevant to BBB transport while Morgan encodes broader connectivity patterns.
- Random Forest and LightGBM are the best-performing model families; this is consistent with the relatively small dataset size (700–750 training samples) where tree-based ensembles are robust to overfitting.
- Morgan + RF (0.9504) is consistent with the previously reported baseline (AUC: 0.9401 ± 0.0454), with the extended benchmark confirming the result across 5 seeds rather than 3.
- The standard deviations are not yet tabulated here; full statistical analysis is pending.

---

## Regression Results

| Rank | Feature | Model | Test R² (mean ± std) |
|------|---------|-------|----------------------|
| 1 | descriptors_basic | rf_reg | 0.4488 |
| 2 | maccs | rf_reg | 0.4397 |
| 3 | descriptors_basic | xgb_reg | 0.4347 |
| 4 | maccs | svm_reg | 0.4212 |
| 5 | descriptors_basic | ridge | 0.4139 |

### Interpretation

- R² values of ~0.44–0.45 indicate that **44–45% of logBB variance is explained**, which is moderate but expected for this task. logBB prediction is inherently difficult due to measurement noise and the complex biological mechanisms underlying transport.
- Physicochemical descriptors and MACCS fingerprints perform similarly, suggesting that simple molecular properties (LogP, TPSA, etc.) capture a meaningful fraction of BBB permeability variance.
- Tree-based models (RF, XGBoost) again outperform others on this dataset size.
- Regression performance should be interpreted cautiously: R² ≈ 0.45 means the model has predictive value but significant unexplained variance remains.

---

## Official Baseline Recommendations

The following combinations are recommended as official baselines for future model comparisons:

| Task | Feature | Model | Metric | Value |
|------|---------|-------|--------|-------|
| Classification | maccs | lgbm | AUC | 0.9535 |
| Classification | morgan | rf | AUC | 0.9504 |
| Regression | descriptors_basic | rf_reg | R² | 0.4488 |

**Rationale:**
- **maccs + lgbm** achieves the highest AUC; MACCS keys are compact (167 bits) and computationally cheap, making them a practical default.
- **morgan + rf** matches maccs + rf at AUC 0.9504 and is more widely used in literature, making it suitable for external comparison.
- **descriptors_basic + rf_reg** is the strongest regression baseline; descriptors are interpretable and the RF model provides a stable, well-understood reference point.

All three combinations should be used as the primary reference points when evaluating graph neural networks and other advanced models.

---

## Limitations

1. **Dataset size:** 700–750 training samples limits model complexity and generalization guarantees. Reported AUC/R² values may overestimate performance on truly novel chemical space.
2. **Scaffold split:** Scaffold-based splitting provides a conservative estimate (molecules in test sets are structurally distinct from training), which is appropriate for benchmarking but may understate performance in relaxed settings.
3. **Single dataset:** Results are specific to the B3DB dataset. Generalization to other BBB datasets has not been evaluated.
4. **Seed variance:** Standard deviations across seeds have not been fully tabulated; observed differences between top combinations (e.g., 0.9535 vs 0.9504) may not be statistically significant.

---

## Next Steps: Why GNNs

Classical ML baselines operate on fixed, handcrafted molecular representations. These representations have three fundamental limitations:

1. **Fixed encoding:** Morgan fingerprints and MACCS keys encode predefined substructures. They cannot discover novel, task-relevant patterns that were not anticipated in the fingerprint design.
2. **No molecular context:** Fingerprints treat atoms independently (or in fixed-radius neighborhoods). True molecular geometry and long-range interactions are not captured.
3. **No end-to-end learning:** Features are extracted separately from the model. The representation and the predictor are not jointly optimized for the task.

**Graph Neural Networks (GNNs)** address these limitations:
- GNNs operate on the molecular graph structure directly, learning task-specific representations end-to-end.
- Message-passing neural networks can capture long-range interactions between atoms that fingerprints miss.
- The molecular graph is a natural representation for this domain.

**Expected benefit:** GNNs should improve over classical baselines, particularly for molecules with complex transport mechanisms where local substructures alone are insufficient predictors. However, improvements may be modest on this dataset size — overfitting risk is high with graph-level models on 700-sample training sets.

**Recommended approach:**
- Start with a simple 3–4 layer GCN or GAT with standard regularization (dropout, weight decay).
- Use the scaffold split to evaluate whether GNNs generalize to unseen scaffold families.
- Compare directly against the official baselines above.

---

## Long-Term Research Trajectory

After GNN baselines are established:

### Transformer-style models

Graph Transformers (e.g., Graphormer, GPS) can capture global molecular context that message-passing GNNs may miss. They are the natural next step after simple GNNs if performance plateaus.

### ZINC22 pretraining

Pretraining on large molecular datasets (ZINC22, ~1B compounds) provides molecular representations that transfer to downstream tasks. This is the highest-risk, highest-reward step: pretraining may substantially improve both classification and regression, especially for underrepresented scaffold families in B3DB. However, it requires significant compute and a robust transfer learning protocol.

**Current priority order:** GNNs → Transformers → ZINC22 pretraining.

---

*Last Updated: 2026-04-02*
*Experiment: Scaffold-split classical baselines, 5 seeds, 5 features, 6 models (cls + reg)*
