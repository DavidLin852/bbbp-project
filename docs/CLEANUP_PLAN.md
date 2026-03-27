# Codebase Cleanup Plan for CFFF Migration

**Date:** 2025-03-27
**Goal:** Clean up project before migrating to CFFF
**Status:** Planning phase - NO DELETIONS YET

---

## 1. WORKING BASELINE PIPELINE (Active - KEEP)

These files constitute the current working baseline and MUST be kept.

### Core Pipeline Scripts

**NEW modular pipeline (currently used):**
```
scripts/baseline/
├── 01_preprocess_b3db.py          # Data preprocessing
├── 02_compute_features.py          # Feature computation
└── 03_train_baselines.py          # Model training
```

**Analysis & Reporting:**
```
scripts/analysis/
├── aggregate_results.py            # Aggregate experiment results
├── generate_benchmark_summary.py   # Generate benchmark summary
└── run_baseline_matrix.py          # Run full experiment matrix
```

### Core Source Modules (NEW)

```
src/
├── config.py                       # Global configuration
├── data/                           # Data preprocessing
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── scaffold_split.py
│   └── dataset.py
├── features/                       # Feature extraction
│   ├── __init__.py
│   ├── fingerprints.py
│   ├── descriptors.py
│   └─��� graph.py
├── models/                         # Model definitions
│   ├── __init__.py
│   ├── baseline_models.py
│   └── model_factory.py
├── train/                          # Training logic
│   ├── __init__.py
│   └── trainer.py
├── evaluate/                       # Evaluation
│   ├── __init__.py
│   ├── comparison.py
│   └── report.py
└── utils/                          # Utilities
    ├── __init__.py
    ├── io.py
    ├── metrics.py
    ├── plotting.py
    ├── seed.py
    └── split.py
```

### Data Files (Active)

```
data/
├── raw/
│   ├── B3DB_classification.tsv     # Original B3DB data
│   └── B3DB_regression.tsv         # Original B3DB data
└── splits/                          # Generated splits
    └── seed_*/
        └── classification_scaffold/
            ├── train.csv
            ├── val.csv
            └── test.csv
```

### Artifacts (Active - Results)

```
artifacts/
├── features/                       # Computed features
│   └── seed_*/
│       └── scaffold/
│           ├── morgan/
│           └── descriptors_basic/
├── models/                         # Trained models
│   └── baselines/
│       └── seed_*/
│           └── scaffold/
│               ├── morgan/
│               │   ├── rf/
│               │   ├── xgb/
│               │   ├── lgbm/
│               │   ├── comparison.json
│               │   └── reports/
│               └── descriptors_basic/
└── reports/                        # Aggregated results
    ├── baseline_results_master.csv
    ├── baseline_summary_by_model.csv
    ├── baseline_summary_by_feature.csv
    ├── benchmark_summary.csv
    └── benchmark_report.txt
```

### Documentation (Active)

```
docs/
├── BASELINE_BENCHMARK.md          # Benchmark results (⭐ IMPORTANT)
├── NEW_STRUCTURE.md                # New structure documentation
├── QUICK_REFERENCE.md              # Quick reference
├── RESULTS_TRACKING.md             # Results tracking guide
└── QUICK_START_EXPERIMENTS.md      # Experiments quick start

PROJECT_CONTEXT.md                  # Project context (⭐ IMPORTANT)
CLAUDE.md                           # Legacy project docs
README.md                            # Project README
```

### Configuration

```
src/config.py                        # All configuration
requirements.txt                     # Dependencies
.env.example                         # Environment variables example
.gitignore                           # Git ignore
docker-compose.yml                   # Docker setup
Dockerfile                           # Docker image
```

### Web Interface (Optional - Can Archive)

```
app_bbb_predict.py                  # Streamlit app
pages/
├── 0_prediction.py
├── 1_smarts_analysis.py
├── 2_model_comparison.py
├── 3_active_learning.py
├── 4_dim_reduction.py
├── 6_ensemble_prediction.py
├── 7_molecule_generation.py
└── 9_mechanism_prediction.py
```

---

## 2. OLD SCRIPTS (Legacy - Archive)

These are the OLD scripts that have been superseded by the new baseline pipeline.

### Old Pipeline Scripts (in root scripts/)

```
scripts/
├── 01_prepare_splits.py            # OLD: Replaced by baseline/01_preprocess_b3db.py
├── 02_featurize_all.py             # OLD: Replaced by baseline/02_compute_features.py
├── 03_run_baselines.py             # OLD: Replaced by baseline/03_train_baselines.py
├── 04_run_gat_aux.py               # GNN training (NOT baseline)
├── 05_pretrain_smarts.py           # SMARTS pretraining (NOT baseline)
├── 06_finetune_bbb_from_smarts.py  # Fine-tuning (NOT baseline)
├── 07_train_vae.py                 # VAE training (NOT baseline)
├── 08_train_gan.py                 # GAN training (NOT baseline)
├── 09_generate_molecules.py        # Generation (NOT baseline)
├── 09_generate_molecules_simple.py # Generation (NOT baseline)
├── 10_validate_*.py                # Various validation scripts (NOT baseline)
├── 11_validate_only_b3db.py
├── 12_improve_separation.py
├── collect_mechanism_data.py        # Mechanism data (NOT baseline)
├── create_auc_f1_scatter.py         # Visualization (NOT baseline)
├── draw_all_smarts_from_json.py     # Visualization (NOT baseline)
├── generate_final_comprehensive_heatmap.py  # Visualization (NOT baseline)
└── visualize_molecule_predictions.py  # Visualization (NOT baseline)
```

### Mechanism Training Scripts

```
scripts/mechanism_training/           # Cornelissen mechanism prediction
├── *.py                             # All scripts (NOT baseline)
```

---

## 3. LEGACY SOURCE CODE (Archive - Not Currently Used)

These modules are NOT used by the current baseline pipeline.

### Legacy Source Modules

```
src/
├── baseline/                        # OLD baseline implementation
│   ├── eval_baselines.py
│   ├── train_baselines.py
│   └── train_rf_xgb_lgb.py
├── featurize/                       # OLD feature extraction
│   ├── fingerprints.py
│   ├── graph_pyg.py
│   └── rdkit_descriptors.py
├── phys_aux/                        # GNN auxiliary tasks
│   └── train_gat_aux.py
├── finetune/                        # GNN fine-tuning
│   └── train_gat_bbb_from_pretrain.py
├── pretrain/                        # GNN pre-training
│   ├── backbone_gat.py
│   ├── graph_pyg_smarts.py
│   ├── smarts_labels.py
│   ├── train_gat_multitask_cls_reg.py
│   ├── train_gat_smarts.py
│   ├── zinc20_loader.py
│   └── zinc20_pretrain.py
├── transformer/                     # Transformer models
│   └── transformer_model.py
├── vae/                             # VAE models
│   ├── dataset.py
│   ├── molecule_vae.py
│   └── train_vae.py
├── gan/                             # GAN models
│   ├── molgan.py
│   ├── reward.py
│   └── train_molgan.py
├── generation/                      # Molecule generation
│   ├── filter_utils.py
│   └── generate_molecules.py
├── path_prediction/                 # Transport mechanism prediction
│   ├── data_collector.py
│   ├── feature_extractor.py
│   ├── integrated_mechanism_predictor.py
│   ├── mechanism_predictor.py
│   └── mechanism_predictor_cornelissen.py
├── explain/                         # Interpretability
│   ├── atom_grad.py
│   ├── draw_rdkit.py
│   ├── shap_analysis.py
│   └── smarts_occlusion.py
├── active_learning.py               # Active learning (NOT baseline)
└── multi_model_predictor.py         # Multi-model predictor (NOT baseline)
```

---

## 4. ARCHIVE CONTENTS (Already Archived)

These are already in the `archive/` directory and should stay there.

```
archive/
├── root_backup/                     # Old root files
│   ├── app_bbb_predict_complete.py
│   ├── download_chembl.py
│   ├── download_zinc20_real.py
│   ├── explore_pretraining_data.py
│   ├── generate_diverse_molecules.py
│   ├── generate_plots_from_export.py
│   ├── get_real_zinc_data.py
│   ├── pretrain_zinc20.py
│   ├── run_gnn_pipeline.py
│   └── test_prediction.py
├── scripts_backup/                  # Old scripts
│   ├── 01_prepare_all_datasets.py
│   ├── 02_enhanced_featurize.py
│   ├── 04_run_all_baselines.py
│   ├── 05_run_complete_matrix.py
│   ├── 06_run_simplified_matrix.py
│   ├── 07_plot_final_roc.py
│   ├── 08_explain_atoms.py
│   ├── 09_explain_smarts.py
│   ├── 10_global_smarts_importance.py
│   ├── 11_global_smarts_interactions.py
│   ├── 12_plot_*.py
│   ├── 13_plot_*.py
│   ├── 14_predict_smiles.py
│   ├── 14_predict_smiles_cli.py
│   ├── 15_ablate_smarts_on_model.py
│   ├── 16_tsne_analysis.py
│   ├── 17_advanced_dim_reduction.py
│   ├── 18_compare_methods.py
│   ├── combine_all_results.py
│   ├── enhanced_shap_analysis.py
│   ├── enhanced_train_transformer.py
│   ├── generate_comprehensive_heatmap.py
│   ├── generate_final_heatmap.py
│   ├── generate_final_summary.py
│   ├── generate_model_heatmap.py
│   ├── generate_multi_metric_heatmap.py
│   ├── run_anova_analysis.py
│   ├── run_ensemble_models.py
│   └── run_missing_experiments.py
└── analysis/                        # Old analysis scripts
    ├── analyze_molecules_ensemble.py
    ├── create_priority_list.py
    ├── simple_analysis.py
    └── simple_ensemble_report.py
```

---

## 5. OUTPUT FILES (Cleanup Needed)

### Current Outputs Directory

```
outputs/
├── b3db_analysis/                  # Old analysis outputs (ARCHIVE)
├── b3db_lda_analysis/              # Old analysis outputs (ARCHIVE)
├── b3db_mechanism_analysis/         # Old analysis outputs (ARCHIVE)
├── cornelissen_comprehensive_analysis/  # Cornelissen analysis (ARCHIVE)
├── data/                            # Data copies (REMOVE?)
├── docs/                           # Documentation copies (REMOVE?)
├── generated_molecules/             # Generated molecules (ARCHIVE)
├── images/                          # Old figures (ARCHIVE)
├── logs/                           # Old logs (ARCHIVE)
├── mechanism_analysis/              # Mechanism analysis (ARCHIVE)
├── mechanism_clustering/            # Mechanism analysis (ARCHIVE)
├── molecule_predictions/            # Old predictions (ARCHIVE)
└── proof1_analysis/                # Proof1 analysis (ARCHIVE)
```

---

## 6. CLASSIFICATION PROPOSAL

### KEEP (Working Baseline - CFFF Migration)

**Files:** 42 Python files + data + artifacts + docs

```
bbb_project/
├── scripts/
│   ├── baseline/                   # ⭐ CORE PIPELINE
│   │   ├── 01_preprocess_b3db.py
│   │   ├── 02_compute_features.py
│   │   └── 03_train_baselines.py
│   └── analysis/                   # ⭐ REPORTING
│       ├── aggregate_results.py
│       ├── generate_benchmark_summary.py
│       └── run_baseline_matrix.py
│
├── src/
│   ├── config.py                   # ⭐ CONFIG
│   ├── data/                       # ⭐ NEW MODULES
│   ├── features/                   # ⭐ NEW MODULES
│   ├── models/                     # ⭐ NEW MODULES
│   ├── train/                      # ⭐ NEW MODULES
│   ├── evaluate/                   # ⭐ NEW MODULES
│   └── utils/                      # ⭐ UTILITIES
│
├── data/
│   ├── raw/                        # ⭐ INPUT DATA
│   │   ├── B3DB_classification.tsv
│   │   └── B3DB_regression.tsv
│   └── splits/                     # ⭐ GENERATED SPLITS
│
├── artifacts/
│   ├── features/                   # ⭐ COMPUTED FEATURES
│   ├── models/                     # ⭐ TRAINED MODELS
│   └── reports/                    # ⭐ RESULTS
│
├── docs/
│   ├── PROJECT_CONTEXT.md          # ⭐ CONTEXT
│   ├── BASELINE_BENCHMARK.md       # ⭐ RESULTS
│   ├── NEW_STRUCTURE.md
│   ├── QUICK_REFERENCE.md
│   ├── RESULTS_TRACKING.md
│   └── QUICK_START_EXPERIMENTS.md
│
├── requirements.txt                # ⭐ DEPENDENCIES
├── README.md
└── .gitignore
```

**Total: ~50 files** (clean, focused, ready for CFFF)

---

### ARCHIVE (Legacy - Keep for Reference)

**Category A: Old Scripts (Before Reorganization)**

```
archive/
└── old_scripts/
    ├── scripts/01-12*.py           # Old root scripts (before baseline/)
    ├── scripts/mechanism_training/ # Mechanism prediction scripts
    ├── scripts/collect_mechanism_data.py
    ├── scripts/create_auc_f1_scatter.py
    ├── scripts/draw_all_smarts_from_json.py
    ├── scripts/generate_final_comprehensive_heatmap.py
    └── scripts/visualize_molecule_predictions.py
```

**Category B: Legacy Source Code (Before Modularization)**

```
archive/
└── old_src/
    ├── src/baseline/               # Old baseline (pre-2025)
    ├── src/featurize/              # Old features (pre-2025)
    ├── src/phys_aux/               # GNN auxiliary
    ├── src/finetune/               # GNN fine-tuning
    ├── src/pretrain/               # GNN pre-training
    ├── src/transformer/            # Transformer models
    ├── src/vae/                    # VAE models
    ├── src/gan/                    # GAN models
    ├── src/generation/             # Generation pipeline
    ├── src/path_prediction/         # Mechanism prediction
    ├── src/explain/                # Interpretability
    ├── src/active_learning.py
    └── src/multi_model_predictor.py
```

**Category C: Old Output Files**

```
archive/
└── old_outputs/
    └── outputs/                    # Entire outputs/ directory
        ├── b3db_*/
        ├── cornelissen_comprehensive_analysis/
        ├── generated_molecules/
        ├── mechanism_*/
        └── proof1_analysis/
```

**Category D: Web Interface (Optional - Archive If Not Used on CFFF)**

```
archive/
└── web_interface/
    ├── app_bbb_predict.py
    └── pages/
        ├── 0_prediction.py
        ├── 1_smarts_analysis.py
        ├── 2_model_comparison.py
        ├── 3_active_learning.py
        ├── 4_dim_reduction.py
        ├── 6_ensemble_prediction.py
        ├── 7_molecule_generation.py
        └── 9_mechanism_prediction.py
```

---

### MAYBE REMOVE (After Review - Risky)

**These need careful review before removal:**

```
1. artifacts/ablation/              # Old ablation study results
2. artifacts/active_learning_cache/ # Active learning cache
3. artifacts/cache/                # Feature cache
4. artifacts/explain/              # OLD explainability results
5. artifacts/figures/              # Old figures
6. artifacts/logs/                 # Old logs
7. artifacts/metrics/              # OLD metrics
8. artifacts/predictions/          # OLD predictions
9. artifacts/smarts_viz/            # SMARTS visualizations
10. artifacts/temp_predict/        # Temporary predictions
11. artifacts/analysis/            # OLD analysis results
```

**Recommendation:** Review content, then either:
- Keep if contains useful analysis
- Archive to `archive/old_artifacts/` if old
- Delete if clearly temporary

---

### TEMPORARY/OBSOLETE (Safe to Archive or Delete)

**Test files:**
```
test_mechanism_prediction.py          # Test script
tools/visualization_template/         # Template
```

**Old documentation:**
```
archive/docs_backup/                  # Already archived
archive/images_backup/                # Already archived
archive/old_docs/                    # Already archived
```

**Data directories (not used by baseline):**
```
data/cns_drugs/                       # External data
data/efflux/                         # External data
data/influx/                         # External data
data/mechanism/                      # Mechanism data
data/pampa/                          # External data
data/transport_mechanisms/           # External data
data/zinc20/                         # ZINC20 data (not used yet)
```

---

## 7. PROPOSED CLEAN STRUCTURE FOR CFFF

### Target Structure (After Cleanup)

```
bbb_project/
│
├── 📁 scripts/                      # Entry point scripts
│   ├── baseline/                   # ⭐ Baseline pipeline
│   │   ├── 01_preprocess_b3db.py
│   │   ├── 02_compute_features.py
│   │   └── 03_train_baselines.py
│   └── analysis/                   # ⭐ Analysis & reporting
│       ├── aggregate_results.py
│       ├── generate_benchmark_summary.py
│       └── run_baseline_matrix.py
│
├── 📁 src/                         # Source code (modular)
│   ├── config.py                   # ⭐ Configuration
│   ├── data/                       # ⭐ Data preprocessing
│   ├── features/                   # ⭐ Feature extraction
│   ├── models/                     # ⭐ Model definitions
│   ├── train/                      # ⭐ Training logic
│   ├── evaluate/                   # ⭐ Evaluation
│   └── utils/                      # ⭐ Utilities
│
├── 📁 data/                        # Data
│   ├── raw/                        # ⭐ Input datasets
│   │   ├── B3DB_classification.tsv
│   │   └── B3DB_regression.tsv
│   └── splits/                     # ⭐ Generated splits
│       └── seed_*/
│
├── 📁 artifacts/                   # Generated artifacts
│   ├── features/                   # ⭐ Computed features
│   ├── models/                     # ⭐ Trained models
│   └── reports/                    # ⭐ Aggregated results
│       ├── baseline_results_master.csv
│       ├── benchmark_summary.csv
│       └── benchmark_report.txt
│
├── 📁 docs/                       # Documentation
│   ├── PROJECT_CONTEXT.md          # ⭐ Project context
│   ├── BASELINE_BENCHMARK.md       # ⭐ Benchmark results
│   ├── NEW_STRUCTURE.md            # Structure docs
│   ├── QUICK_REFERENCE.md          # Quick reference
│   ├── RESULTS_TRACKING.md         # Results guide
│   └── QUICK_START_EXPERIMENTS.md  # Experiments guide
│
├── 📁 archive/                     # ⭐ Archived legacy code
│   ├── old_scripts/               # Old root scripts
│   ├── old_src/                   # Old source modules
│   ├── old_outputs/               # Old output files
│   └── old_web/                   # Old web interface
│
├── 📄 requirements.txt             # ⭐ Dependencies
├── 📄 README.md                   # ⭐ Project README
├── 📄 .gitignore                  # ⭐ Git config
├── 📄 docker-compose.yml          # Docker setup
└── 📄 Dockerfile                  # Docker image
```

**Key Principles:**
1. **Flat structure** - minimal nesting
2. **Clear separation** - scripts, src, data, artifacts, docs
3. **Archive legacy** - old code in `archive/`, not in root
4. **Ready for CFFF** - only essential files for baseline experiments

---

## 8. RISKY FILES (Do NOT Remove Without Review)

### High Risk - Must Review First

```
1. src/transformer/              # Transformer implementation (future use)
2. src/vae/                      # VAE models (future use)
3. src/gan/                      # GAN models (future use)
4. src/pretrain/                 # Pre-training logic (future ZINC22 use)
5. src/finetune/                 # Fine-tuning logic (future use)
6. src/path_prediction/           # Mechanism prediction (research value)
7. data/zinc20/                  # ZINC22 data (future use)
8. artifacts/analysis/           # May contain useful analysis
9. artifacts/explain/            # Interpretability results
```

### Medium Risk - Review Content

```
1. artifacts/ablation/           # Ablation study results
2. artifacts/figures/            # Generated figures
3. artifacts/metrics/           # Old metrics
4. artifacts/smarts_viz/        # SMARTS analysis
```

### Low Risk - Safe to Archive

```
1. archive/*                      # Already archived
2. outputs/*                      # Old outputs
3. test_mechanism_prediction.py   # Test script
4. tools/*                        # Templates
```

---

## 9. CLEANUP EXECUTION PLAN

### Phase 1: Safe Archiving (No Deletions)

```bash
# 1. Create archive directories
mkdir -p archive/old_scripts
mkdir -p archive/old_src
mkdir -p archive/old_outputs
mkdir -p archive/old_web
mkdir -p archive/old_artifacts

# 2. Archive old scripts (root scripts/)
mv scripts/[0-9]*.py archive/old_scripts/ 2>/dev/null
mv/scripts/mechanism_training archive/old_scripts/
mv scripts/collect_mechanism_data.py archive/old_scripts/
mv scripts/create_auc_f1_scatter.py archive/old_scripts/
mv scripts/draw_all_smarts_from_json.py archive/old_scripts/
mv scripts/generate_final_comprehensive_heatmap.py archive/old_scripts/
mv scripts/visualize_molecule_predictions.py archive/old_scripts/

# 3. Archive old source modules
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

# 4. Archive old outputs
mv outputs archive/old_outputs/

# 5. Archive web interface (optional)
# mv app_bbb_predict.py archive/old_web/
# mv pages archive/old_web/
```

### Phase 2: Review & Clean Artifacts (Manual Review Required)

```bash
# Review artifacts subdirectories
# Decide: keep, archive, or delete
ls -la artifacts/ablation/
ls -la artifacts/active_learning_cache/
ls -la artifacts/cache/
ls -la artifacts/explain/
ls -la artifacts/figures/
ls -la artifacts/logs/
ls -la artifacts/metrics/
ls -la artifacts/predictions/
ls -la artifacts/smarts_viz/
ls -la artifacts/temp_predict/
ls -la artifacts/analysis/
```

### Phase 3: Clean Unused Data

```bash
# Archive external data not used by baseline
mkdir -p archive/external_data
mv data/cns_drugs archive/external_data/
mv data/efflux archive/external_data/
mv data/influx archive/external_data/
mv data/mechanism archive/external_data/
mv data/pampa archive/external_data/
mv data/transport_mechanisms archive/external_data/
mv data/zinc20 archive/external_data/
```

---

## 10. FINAL STRUCTURE FOR CFFF

After cleanup, the project will have:

```
Total Files: ~100 (down from ~250)
├── Working baseline: 50 files
├── Archived: 150 files
└── Git-tracked: ~50 files
```

**Git should track:**
- scripts/baseline/
- scripts/analysis/
- src/ (NEW modules only)
- data/raw/
- docs/
- requirements.txt
- README.md
- .gitignore

**Git should NOT track:**
- data/splits/ (generated)
- artifacts/ (generated)
- archive/ (archived)
- outputs/ (archived)

---

## SUMMARY

### Files to KEEP (50 files)
- **Scripts:** 6 (baseline + analysis)
- **Source:** 15 modules (NEW modular structure)
- **Data:** 2 input files
- **Docs:** 6 documentation files
- **Config:** 5 files (requirements, docker, etc.)

### Files to ARCHIVE (~200 files)
- **Old scripts:** ~30 files
- **Old source:** ~20 modules
- **Old outputs:** ~100 files
- **Web interface:** ~10 files
- **Test files:** ~5 files

### Files to REVIEW BEFORE DELETION (~20 directories)
- **Old artifacts:** 10 directories in `artifacts/`
- **External data:** 7 directories in `data/`

---

**RECOMMENDATION:** Start with Phase 1 (safe archiving), review archived content, then proceed to Phase 2-3 only after confirmation.
