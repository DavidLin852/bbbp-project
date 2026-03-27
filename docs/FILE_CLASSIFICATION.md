# Concrete File Classification for CFFF Cleanup

**Status:** PLANNING PHASE - NO DELETIONS YET
**Date:** 2025-03-27

---

## KEEP (Working Baseline - 50 Files)

### Core Scripts (6 files)

```
✅ scripts/baseline/01_preprocess_b3db.py
✅ scripts/baseline/02_compute_features.py
✅ scripts/baseline/03_train_baselines.py
✅ scripts/analysis/aggregate_results.py
✅ scripts/analysis/generate_benchmark_summary.py
✅ scripts/analysis/run_baseline_matrix.py
```

### Core Source Modules (15 modules)

```
✅ src/__init__.py
✅ src/config.py
✅ src/data/__init__.py
✅ src/data/preprocessing.py
✅ src/data/scaffold_split.py
✅ src/data/dataset.py
✅ src/features/__init__.py
✅ src/features/fingerprints.py
✅ src/features/descriptors.py
✅ src/features/graph.py
✅ src/models/__init__.py
✅ src/models/baseline_models.py
✅ src/models/model_factory.py
✅ src/train/__init__.py
✅ src/train/trainer.py
✅ src/evaluate/__init__.py
✅ src/evaluate/comparison.py
✅ src/evaluate/report.py
✅ src/utils/__init__.py
✅ src/utils/io.py
✅ src/utils/metrics.py
✅ src/utils/plotting.py
✅ src/utils/seed.py
✅ src/utils/split.py
```

### Data (2 files)

```
✅ data/raw/B3DB_classification.tsv
✅ data/raw/B3DB_regression.tsv
```

### Documentation (6 files)

```
✅ docs/PROJECT_CONTEXT.md
✅ docs/BASELINE_BENCHMARK.md
✅ docs/NEW_STRUCTURE.md
✅ docs/QUICK_REFERENCE.md
✅ docs/RESULTS_TRACKING.md
✅ docs/QUICK_START_EXPERIMENTS.md
✅ docs/CLEANUP_PLAN.md (this file)
✅ docs/STRUCTURE_COMPARISON.md
```

### Configuration (5 files)

```
✅ requirements.txt
✅ README.md
✅ .gitignore
✅ docker-compose.yml
✅ Dockerfile
```

---

## ARCHIVE (Legacy - Keep for Reference)

### Category A: Old Root Scripts (30+ files)

```
📦 scripts/01_prepare_splits.py
📦 scripts/02_featurize_all.py
📦 scripts/03_run_baselines.py
📦 scripts/04_run_gat_aux.py
📦 scripts/05_pretrain_smarts.py
📦 scripts/06_finetune_bbb_from_smarts.py
📦 scripts/07_train_vae.py
📦 scripts/08_train_gan.py
📦 scripts/09_generate_molecules.py
📦 scripts/09_generate_molecules_simple.py
📦 scripts/10_validate_chemical_space.py
📦 scripts/10_validate_chemical_space_lda.py
📦 scripts/10_validate_lda_simple.py
📦 scripts/11_validate_only_b3db.py
📦 scripts/12_improve_separation.py
📦 scripts/collect_mechanism_data.py
📦 scripts/create_auc_f1_scatter.py
📦 scripts/draw_all_smarts_from_json.py
📦 scripts/generate_final_comprehensive_heatmap.py
📦 scripts/visualize_molecule_predictions.py
📦 scripts/mechanism_training/ (entire directory)
```

### Category B: Legacy Source Modules (20 modules)

```
📦 src/baseline/ (entire directory)
   ├── eval_baselines.py
   ├── train_baselines.py
   ├── train_rf_xgb_lgb.py
   └── __init__.py

📦 src/featurize/ (entire directory)
   ├── __init__.py
   ├── fingerprints.py
   ├── graph_pyg.py
   └── rdkit_descriptors.py

📦 src/phys_aux/
   └── train_gat_aux.py

📦 src/finetune/
   └── train_gat_bbb_from_pretrain.py

📦 src/pretrain/ (entire directory)
   ├── backbone_gat.py
   ├── graph_pyg_smarts.py
   ├── smarts_labels.py
   ├── train_gat_multitask_cls_reg.py
   ├── train_gat_smarts.py
   ├── zinc20_loader.py
   └── zinc20_pretrain.py

📦 src/transformer/
   └── transformer_model.py

📦 src/vae/ (entire directory)
   ├── dataset.py
   ├── molecule_vae.py
   └── train_vae.py

📦 src/gan/ (entire directory)
   ├── molgan.py
   ├── reward.py
   └── train_molgan.py

📦 src/generation/ (entire directory)
   ├── filter_utils.py
   └── generate_molecules.py

📦 src/path_prediction/ (entire directory)
   ├── data_collector.py
   ├── feature_extractor.py
   ├── integrated_mechanism_predictor.py
   ├── mechanism_predictor.py
   └── mechanism_predictor_cornelissen.py

📦 src/explain/ (entire directory)
   ├── atom_grad.py
   ├── draw_rdkit.py
   ├── shap_analysis.py
   └── smarts_occlusion.py

📦 src/active_learning.py

📦 src/multi_model_predictor.py
```

### Category C: Web Interface (10 files)

```
📦 app_bbb_predict.py
📦 pages/0_prediction.py
📦 pages/1_smarts_analysis.py
📦 pages/2_model_comparison.py
📦 pages/3_active_learning.py
📦 pages/4_dim_reduction.py
📦 pages/6_ensemble_prediction.py
📦 pages/7_molecule_generation.py
📦 pages/9_mechanism_prediction.py
```

### Category D: Old Outputs (100+ files)

```
📦 outputs/ (entire directory)
   ├── b3db_analysis/
   ├── b3db_lda_analysis/
   ├── b3db_mechanism_analysis/
   ├── cornelissen_comprehensive_analysis/
   ├── data/
   ├── docs/
   ├── generated_molecules/
   ├── images/
   ├── logs/
   ├── mechanism_analysis/
   ├── mechanism_clustering/
   ├── molecule_predictions/
   └── proof1_analysis/
```

### Category E: Already Archived

```
✅ archive/ (entire directory - already archived)
   ├── root_backup/
   ├── scripts_backup/
   ├── analysis/
   ├── docs_backup/
   ├── images_backup/
   ├── old_docs/
   └── pretraining_analysis/
```

---

## REVIEW BEFORE DELETION (Risky Files)

### Artifacts Subdirectories (10 dirs - Manual Review)

```
⚠️ artifacts/ablation/
   Status: May contain useful ablation study results
   Action: Review, keep if useful, archive if not

⚠️ artifacts/active_learning_cache/
   Status: Cache files
   Action: Probably safe to delete

⚠️ artifacts/cache/
   Status: Feature cache
   Action: Probably safe to delete

⚠️ artifacts/explain/
   Status: OLD interpretability results
   Action: Review, keep SHAP results, archive rest

⚠️ artifacts/figures/
   Status: Old figures
   Action: Review, keep benchmark figures, archive rest

⚠️ artifacts/logs/
   Status: Old logs
   Action: Safe to delete

⚠️ artifacts/metrics/
   Status: OLD metrics (superseded by benchmark_summary.csv)
   Action: Safe to delete

⚠️ artifacts/predictions/
   Status: OLD predictions (superseded by model dirs)
   Action: Safe to delete

⚠️ artifacts/smarts_viz/
   Status: SMARTS analysis
   Action: Review, keep if useful, archive rest

⚠️ artifacts/temp_predict/
   Status: Temporary files
   Action: Safe to delete

⚠️ artifacts/analysis/
   Status: OLD analysis results
   Action: Review, keep useful analysis, archive rest
```

### External Data Directories (7 dirs - Manual Review)

```
⚠️ data/cns_drugs/
   Status: External data (not baseline)
   Action: Archive (not used yet)

⚠️ data/efflux/
   Status: External data (not baseline)
   Action: Archive (not used yet)

⚠️ data/influx/
   Status: External data (not baseline)
   Action: Archive (not used yet)

⚠️ data/mechanism/
   Status: Mechanism data (already in archive/)
   Action: Archive (duplicate)

⚠️ data/pampa/
   Status: External data (not baseline)
   Action: Archive (not used yet)

⚠️ data/transport_mechanisms/
   Status: External data (not baseline)
   Action: Archive (not used yet)

⚠️ data/zinc20/
   Status: ZINC22 data (not baseline yet)
   Action: Archive (for future use)
```

### Test Files (Safe to Archive/Delete)

```
⚠️ test_mechanism_prediction.py
   Status: Test script
   Action: Archive or delete

⚠️ tools/visualization_template/
   Status: Template
   Action: Archive or delete
```

---

## SUMMARY STATISTICS

### Current Project

```
Total Python files: ~250
├── Working baseline: 50 files (20%)
├── Legacy to archive: 150 files (60%)
├── Risky (need review): 20 files (8%)
└── Already archived: 30 files (12%)
```

### After Cleanup

```
Total Python files: ~80
├── Working baseline: 50 files (62%)
└── Archived: 30 files (38%)
```

---

## EXECUTION PLAN

### Step 1: Safe Archiving (No Data Loss)

```bash
# Create archive structure
mkdir -p archive/old_scripts/numbered_scripts
mkdir -p archive/old_scripts/mechanism
mkdir -p archive/old_scripts/visualization
mkdir -p archive/old_scripts/analysis
mkdir -p archive/old_src
mkdir -p archive/old_outputs
mkdir -p archive/old_web
mkdir -p archive/external_data

# Archive old root scripts
mv scripts/01_prepare_splits.py archive/old_scripts/numbered_scripts/
mv scripts/02_featurize_all.py archive/old_scripts/numbered_scripts/
mv scripts/03_run_baselines.py archive/old_scripts/numbered_scripts/
mv scripts/04_run_gat_aux.py archive/old_scripts/numbered_scripts/
mv scripts/05_pretrain_smarts.py archive/old_scripts/numbered_scripts/
mv scripts/06_finetune_bbb_from_smarts.py archive/old_scripts/numbered_scripts/
mv scripts/07_train_vae.py archive/old_scripts/numbered_scripts/
mv scripts/08_train_gan.py archive/old_scripts/numbered_scripts/
mv scripts/09_generate_molecules.py archive/old_scripts/numbered_scripts/
mv scripts/09_generate_molecules_simple.py archive/old_scripts/numbered_scripts/
mv scripts/10_validate*.py archive/old_scripts/numbered_scripts/
mv scripts/11_validate_only_b3db.py archive/old_scripts/numbered_scripts/
mv scripts/12_improve_separation.py archive/old_scripts/numbered_scripts/
mv scripts/mechanism_training archive/old_scripts/mechanism/
mv scripts/collect_mechanism_data.py archive/old_scripts/analysis/
mv scripts/create_auc_f1_scatter.py archive/old_scripts/analysis/
mv scripts/draw_all_smarts_from_json.py archive/old_scripts/analysis/
mv scripts/generate_final_comprehensive_heatmap.py archive/old_scripts/analysis/
mv scripts/visualize_molecule_predictions.py archive/old_scripts/analysis/

# Archive old source modules
mv src/baseline archive/old_src/
mv src/featurize archive/old_src/
mv src/phys_aux archive/old_src/
mv src/finetune archive/old_src/
mv src/pretrain archive/old_src/
mv src/transformer archive/old_src/
mv src/vae archive/old_src/
mv src/gan archive/old_src/
mv src/generation archive/old_src/
mv src/path_prediction archive/old_src/
mv src/explain archive/old_src/
mv src/active_learning.py archive/old_src/
mv src/multi_model_predictor.py archive/old_src/

# Archive old outputs
mv outputs archive/old_outputs/

# Archive external data
mv data/cns_drugs archive/external_data/
mv data/efflux archive/external_data/
mv data/influx archive/external_data/
mv data/mechanism archive/external_data/
mv data/pampa archive/external_data/
mv data/transport_mechanisms archive/external_data/
mv data/zinc20 archive/external_data/

# Archive test files
mv test_mechanism_prediction.py archive/
mv tools archive/
```

### Step 2: Review Artifacts (Manual)

```bash
# Review each artifact subdirectory
# For each one:
# 1. Check contents: ls -la artifacts/<dir>/
# 2. Decide: keep, archive, or delete
# 3. If archive: mv artifacts/<dir> archive/old_artifacts/<dir>/
# 4. If delete: rm -rf artifacts/<dir>/

# Example:
ls -la artifacts/ablation/
# If old/not useful: mv artifacts/ablation archive/old_artifacts/
```

### Step 3: Final Cleanup

```bash
# Clean up old artifacts (after review)
rm -rf artifacts/active_learning_cache/
rm -rf artifacts/cache/
rm -rf artifacts/logs/
rm -rf artifacts/metrics/
rm -rf artifacts/predictions/
rm -rf artifacts/temp_predict/
```

---

## FILES CREATED FOR CLEANUP PLAN

```
docs/CLEANUP_PLAN.md              # Full cleanup plan (this file)
docs/STRUCTURE_COMPARISON.md       # Current vs target structure
docs/FILE_CLASSIFICATION.md         # This file (concrete list)
```

---

## NEXT STEPS

1. ✅ Review cleanup plan
2. ⏳ Decide: approve cleanup plan?
3. ⏳ Execute Phase 1: Safe archiving
4. ⏳ Review: check archived content
5. ⏳ Execute Phase 2: Review artifacts
6. ⏳ Execute Phase 3: Final cleanup
7. ⏳ Test baseline pipeline still works
8. ⏳ Update .gitignore
9. ⏳ Prepare for CFFF migration

---

**IMPORTANT:** Do not proceed to execution phase without explicit approval.
