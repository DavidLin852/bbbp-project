# Current Tasks

## High Priority
1. Build a standardized preprocessing pipeline for B3DB classification and regression datasets.
2. Support canonicalization, invalid molecule filtering, and deduplication.
3. Create scaffold split as the main split protocol.
4. Organize project structure into scripts + src modules.
5. Build reproducible baseline scripts for:
   - RF
   - XGBoost
   - LightGBM
6. Prepare the codebase for running on the CFFF platform.

## Medium Priority
1. Add graph data preparation pipeline for GNN models.
2. Add baseline GNN training entry.
3. Add experiment logging and output directory conventions.
4. Add descriptor / fingerprint generation utilities.

## Lower Priority
1. Draft ZINC22 pretraining pipeline.
2. Add Transformer baseline.
3. Add interpretability scripts such as SHAP and graph attribution.
4. Add ensemble logic after single-model baselines are stable.

## Explicit Non-Priority Right Now
- Do not start full-scale ZINC22 pretraining yet.
- Do not build unconstrained molecular generation now.
- Do not overfocus on clustering visualizations as the main result.
- Do not introduce overly complicated framework abstractions too early.

## First Immediate Deliverable
The first concrete deliverable should be:
- a clean B3DB preprocessing module
- clean output files for classification and regression
- scaffold split generation
- baseline-ready data outputs