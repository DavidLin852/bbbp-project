# Project Structure - Current vs Target

## CURRENT STRUCTURE (Before Cleanup)

```
bbb_project/
├── 📄 50+ Python files in root and subdirectories
├── 📁 15+ source modules (mixed new/old)
├── 📁 30+ scripts (mixed new/old)
├── 📁 Large outputs/ directory
├── 📁 Large archive/ directory
├── 📁 artifacts/ with 10+ subdirectories
└── 📁 Complex nesting
```

## TARGET STRUCTURE (After Cleanup)

```
bbb_project/
│
├── 📁 scripts/                      (6 files - CLEAN)
│   ├── baseline/
│   │   ├── 01_preprocess_b3db.py
│   │   ├── 02_compute_features.py
│   │   └── 03_train_baselines.py
│   └── analysis/
│       ├── aggregate_results.py
│       ├── generate_benchmark_summary.py
│       └── run_baseline_matrix.py
│
├── 📁 src/                          (15 modules - CLEAN)
│   ├── config.py
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── train/
│   ├── evaluate/
│   └── utils/
│
├── 📁 data/                         (CLEAN)
│   ├── raw/
│   │   ├── B3DB_classification.tsv
│   │   └── B3DB_regression.tsv
│   └── splits/
│
├── 📁 artifacts/                    (CLEAN)
│   ├── features/
│   ├── models/
│   └── reports/
│
├── 📁 docs/                         (6 files - CLEAN)
│   ├── PROJECT_CONTEXT.md
│   ├── BASELINE_BENCHMARK.md
│   ├── NEW_STRUCTURE.md
│   ├── QUICK_REFERENCE.md
│   ├── RESULTS_TRACKING.md
│   └── QUICK_START_EXPERIMENTS.md
│
├── 📁 archive/                      (ARCHIVED)
│   ├── old_scripts/
│   ├── old_src/
│   ├── old_outputs/
│   └── old_web/
│
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 .gitignore
├── 📄 docker-compose.yml
└── 📄 Dockerfile
```

---

## FILE CLASSIFICATION

### ✅ KEEP (Working Baseline)

**Core Pipeline - 50 files:**

| Category | Files | Purpose |
|----------|-------|---------|
| Scripts | 6 | Entry points for experiments |
| Source | 15 | Modular source code |
| Data | 2 | Input datasets (B3DB) |
| Docs | 6 | Documentation |
| Config | 5 | Dependencies, Docker, etc. |

### 📦 ARCHIVE (Legacy - Keep for Reference)

**Category A: Old Scripts (Before Reorganization) - ~30 files**

| Location | Files | Status |
|----------|-------|--------|
| `scripts/01-12*.py` | Old numbered scripts | Superseded by `scripts/baseline/` |
| `scripts/mechanism_training/` | Mechanism scripts | Not baseline (keep for reference) |
| `scripts/collect_*.py` | Data collection | Not baseline (keep for reference) |
| `scripts/create_*.py` | Visualization | Not baseline (keep for reference) |
| `scripts/draw_*.py` | Visualization | Not baseline (keep for reference) |
| `scripts/generate_*.py` | Old analysis | Not baseline (keep for reference) |
| `scripts/visualize_*.py` | Visualization | Not baseline (keep for reference) |

**Category B: Legacy Source Code - ~20 modules**

| Location | Files | Status |
|----------|-------|--------|
| `src/baseline/` | Old baseline implementation | Superseded by new `src/models/`, `src/train/` |
| `src/featurize/` | Old feature extraction | Superseded by new `src/features/` |
| `src/phys_aux/` | GNN auxiliary tasks | Future use (not baseline) |
| `src/finetune/` | GNN fine-tuning | Future use (not baseline) |
| `src/pretrain/` | GNN pre-training | Future use (not baseline) |
| `src/transformer/` | Transformer models | Future use (not baseline) |
| `src/vae/` | VAE models | Future use (not baseline) |
| `src/gan/` | GAN models | Future use (not baseline) |
| `src/generation/` | Generation pipeline | Future use (not baseline) |
| `src/path_prediction/` | Mechanism prediction | Research (not baseline) |
| `src/explain/` | Interpretability | Research (not baseline) |
| `src/active_learning.py` | Active learning | Research (not baseline) |
| `src/multi_model_predictor.py` | Old predictor | Superseded by new pipeline |

**Category C: Old Outputs - ~100 files**

| Location | Files | Status |
|----------|-------|--------|
| `outputs/b3db_*/` | Old analysis | Archive (old experiments) |
| `outputs/cornelissen_*/` | Cornelissen analysis | Archive (published) |
| `outputs/generated_molecules/` | Generated molecules | Archive (generation experiments) |
| `outputs/mechanism_*/` | Mechanism analysis | Archive (research) |
| `outputs/proof1_*/` | Proof1 analysis | Archive (research) |

**Category D: Web Interface - ~10 files**

| Location | Files | Status |
|----------|-------|--------|
| `app_bbb_predict.py` | Streamlit app | Optional (archive if not used on CFFF) |
| `pages/*.py` | Streamlit pages | Optional (archive if not used on CFFF) |

### ⚠️ REVIEW BEFORE DELETION

**Artifacts Subdirectories (~10 dirs):**

| Directory | Content | Recommendation |
|-----------|---------|----------------|
| `artifacts/ablation/` | Old ablation results | Review: keep useful analysis, archive rest |
| `artifacts/active_learning_cache/` | Active learning cache | Probably safe to delete |
| `artifacts/cache/` | Feature cache | Probably safe to delete |
| `artifacts/explain/` | OLD interpretability | Review: keep SHAP results, archive rest |
| `artifacts/figures/` | Old figures | Review: keep benchmark figures, archive rest |
| `artifacts/logs/` | Old logs | Safe to delete |
| `artifacts/metrics/` | OLD metrics | Safe to delete (in benchmark_summary.csv) |
| `artifacts/predictions/` | OLD predictions | Safe to delete (in model dirs) |
| `artifacts/smarts_viz/` | SMARTS analysis | Review: keep if useful, archive rest |
| `artifacts/temp_predict/` | Temporary | Safe to delete |
| `artifacts/analysis/` | OLD analysis | Review: keep useful analysis, archive rest |

**External Data (~7 dirs):**

| Directory | Content | Recommendation |
|-----------|---------|----------------|
| `data/cns_drugs/` | External data | Archive (not baseline) |
| `data/efflux/` | External data | Archive (not baseline) |
| `data/influx/` | External data | Archive (not baseline) |
| `data/mechanism/` | Mechanism data | Archive (not baseline, in archive/) |
| `data/pampa/` | External data | Archive (not baseline) |
| `data/transport_mechanisms/` | External data | Archive (not baseline) |
| `data/zinc20/` | ZINC22 data | Archive (not baseline yet) |

---

## RECOMMENDED ARCHIVE STRUCTURE

```
archive/
├── old_scripts/                   # Old root scripts
│   ├── numbered_scripts/          # scripts/01-12*.py
│   ├── mechanism_training/         # Mechanism scripts
│   ├── visualization/              # Visualization scripts
│   └── analysis/                  # Old analysis scripts
│
├── old_src/                       # Old source modules
│   ├── baseline/                   # Old baseline
│   ├── featurize/                  # Old features
│   ├── advanced_models/            # GNN, Transformer, VAE, GAN
│   ├── research/                   # Path prediction, explain
│   └── utility/                    # Active learning, multi-model
│
├── old_outputs/                   # Old experiment outputs
│   ├── b3db_analysis/
│   ├── cornelissen_analysis/
│   ├── generated_molecules/
│   ├── mechanism_analysis/
│   └── proof1_analysis/
│
├── old_web/                       # Streamlit app
│   ├── app_bbb_predict.py
│   └── pages/
│
├── old_artifacts/                 # Old artifacts (review first)
│   ├── ablation/
│   ├── figures/
│   ├── explain/
│   └── analysis/
│
└── external_data/                 # External datasets
    ├── cns_drugs/
    ├── efflux/
    ├── influx/
    ├── mechanism/
    ├── pampa/
    ├── transport_mechanisms/
    └── zinc20/
```

---

## CFFF MIGRATION CHECKLIST

### Before Migration

- [ ] Run `scripts/analysis/run_baseline_matrix.py` to verify baseline works
- [ ] Check `artifacts/reports/benchmark_summary.csv` has latest results
- [ ] Verify all 6 baseline scripts are working
- [ ] Document any issues or gotchas

### Cleanup Steps

- [ ] Run Phase 1: Archive old scripts (no deletions)
- [ ] Run Phase 2: Review artifacts subdirectories
- [ ] Run Phase 3: Archive external data
- [ ] Update `.gitignore` to exclude generated files
- [ ] Test that baseline pipeline still works after cleanup

### For CFFF

- [ ] Copy cleaned structure to CFFF
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Test baseline: `python scripts/baseline/01_preprocess_b3db.py --seed 0`
- [ ] Verify data files are present
- [ ] Run small test matrix

### After Migration

- [ ] Run full baseline matrix on CFFF
- [ ] Compare results with local machine
- [ ] Update documentation with CFFF-specific notes
