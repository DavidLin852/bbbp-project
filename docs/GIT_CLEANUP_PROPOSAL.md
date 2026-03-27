# Git Cleanup Proposal - Reduce Repository Size

**Date:** 2025-03-27
**Current Repository Size:** ~354 MB (511 files)
**Target Size:** ~50 MB (clean research code)

---

## Problem Analysis

### Current Tracking Issues

1. **Heavy Artifacts Tracked (~1.9 GB)**
   - `artifacts/ablation/` - 40 files (~1.6 GB) - ablation study results
   - `artifacts/figures/` - 48 files (~30 MB) - visualization outputs
   - `artifacts/explain/` - 13 files (~3.8 MB) - interpretability results
   - `artifacts/smarts_viz/` - 71 files (~848 KB) - SMARTS visualizations
   - `artifacts/metrics/` - 19 files (~1.4 MB) - old metrics (superseded)

2. **Heavy Archive Tracked (~27 MB)**
   - `archive/images_backup/` - 44 files (~26 MB) - backup of old figures
   - `archive/docs_backup/` - 37 files (~328 KB) - backup of old docs
   - Other archive subdirectories (small, acceptable)

3. **Large External Data (~285 MB)**
   - `data/external/structures.sdf` - 277 MB - external structure file
   - `data/external/lipidmaps_smiles.csv` - 5.9 MB - external dataset

4. **Bulky Analysis Files**
   - `20220103_table.csv` - 3.0 MB - unknown origin
   - Multiple large CSV files in artifacts/ (duplicates, old results)

---

## Cleanup Plan

### Phase 1: Remove Large Artifacts from Git (KEEP LOCALLY)

**Remove from Git tracking (keep on disk):**

1. **Ablation Study Results** (~1.6 GB)
   ```
   artifacts/ablation/
   ```
   - Reason: Research outputs, superseded by benchmark_summary.csv
   - Action: `git rm --cached -r artifacts/ablation/`

2. **Figures & Plots** (~30 MB)
   ```
   artifacts/figures/
   ```
   - Reason: Generated visualizations (regeneratable)
   - Action: `git rm --cached -r artifacts/figures/`

3. **Interpretability Results** (~3.8 MB)
   ```
   artifacts/explain/
   ```
   - Reason: Research outputs, not baseline
   - Action: `git rm --cached -r artifacts/explain/`

4. **SMARTS Visualizations** (~848 KB)
   ```
   artifacts/smarts_viz/
   ```
   - Reason: Generated visualizations (regeneratable)
   - Action: `git rm --cached -r artifacts/smarts_viz/`

5. **Old Metrics** (~1.4 MB)
   ```
   artifacts/metrics/
   ```
   - Reason: Superseded by artifacts/reports/benchmark_summary.csv
   - Action: `git rm --cached -r artifacts/metrics/`

**Subtotal saved:** ~1.6 GB

### Phase 2: Remove Archive Outputs from Git (KEEP LOCALLY)

**Remove from Git tracking (keep on disk):**

1. **Image Backups** (~26 MB)
   ```
   archive/images_backup/
   ```
   - Reason: Backup of old figures, not essential
   - Action: `git rm --cached -r archive/images_backup/`

**Subtotal saved:** ~26 MB

### Phase 3: Remove Large External Data from Git (KEEP LOCALLY)

**Remove from Git tracking (keep on disk):**

1. **External Structure File** (277 MB)
   ```
   data/external/structures.sdf
   ```
   - Reason: Large external dataset, not baseline
   - Action: `git rm --cached data/external/structures.sdf`

2. **External CSV** (5.9 MB)
   ```
   data/external/lipidmaps_smiles.csv
   ```
   - Reason: External dataset, not baseline
   - Action: `git rm --cached data/external/lipidmaps_smiles.csv`

3. **Unknown Large CSV** (3.0 MB)
   ```
   20220103_table.csv
   ```
   - Reason: Unknown origin, not documented
   - Action: `git rm --cached 20220103_table.csv`

**Subtotal saved:** ~286 MB

---

## What to KEEP Tracking

### ✅ Essential Files (DO NOT REMOVE)

1. **Source Code** (~5 MB)
   - `scripts/` - All scripts (baseline/, analysis/, mechanism_data_sources.json)
   - `src/` - All source modules (data/, features/, models/, train/, evaluate/, utils/, config.py)
   - Including research code: pretrain/, transformer/, explain/, vae/, gan/, path_prediction/

2. **Documentation** (~1 MB)
   - `docs/` - All documentation (PROJECT_CONTEXT.md, BASELINE_BENCHMARK.md, etc.)
   - `README.md`, `CLAUDE.md`, `TASK.md`
   - `requirements.txt`, `.gitignore`

3. **Configuration** (~1 MB)
   - `Dockerfile`, `docker-compose.yml`
   - `.env.example`

4. **Archived Source Code** (~829 KB)
   - `archive/old_scripts/` - Legacy scripts (numbered_scripts/, mechanism/, analysis/)
   - `archive/old_src/` - Legacy source modules (baseline/, featurize/, vae/, gan/, generation/)
   - `archive/old_web/` - Streamlit web interface
   - `archive/docs_backup/` - Documentation backups (small, useful reference)
   - `archive/old_docs/` - Old documentation
   - `archive/pretraining_analysis/` - Pre-training analysis
   - `archive/root_backup/`, `archive/scripts_backup/` - Small backups

5. **Essential Artifacts** (~50 KB)
   - `artifacts/reports/` - Benchmark reports (6 files, essential)
   - `artifacts/analysis/` - Current analysis results (2 files)

6. **Essential Data** (~3 MB)
   - `data/raw/B3DB_classification.tsv` - Primary dataset (2.6 MB) ✅
   - `data/raw/B3DB_regression.tsv` - Regression dataset (413 KB) ✅

7. **Archive Metadata**
   - `archive/README.md`
   - `archive/analysis/` - Small analysis scripts

---

## Revised .gitignore

```gitignore
# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/

# Virtual environments
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Generated outputs (heavy)
artifacts/features/
artifacts/models/
artifacts/logs/
artifacts/cache/
artifacts/temp_predict/
artifacts/predictions/
artifacts/active_learning_cache/

# Analysis outputs (regeneratable)
artifacts/ablation/
artifacts/figures/
artifacts/explain/
artifacts/smarts_viz/
artifacts/metrics/

# Keep only essential reports
!artifacts/reports/
!artifacts/analysis/

# Archive old outputs (heavy)
archive/old_outputs/
archive/external_data/
archive/images_backup/

# Keep archived source code and docs
!archive/old_scripts/
!archive/old_src/
!archive/old_web/
!archive/docs_backup/
!archive/old_docs/
!archive/pretraining_analysis/
!archive/root_backup/
!archive/scripts_backup/
!archive/analysis/

# Data splits and processed data
data/splits/
data/processed/
data/features/

# Keep raw data
!data/raw/

# External datasets (heavy)
data/external/
data/cns_drugs/
data/efflux/
data/influx/
data/pampa/
data/transport_mechanisms/

# Keep mechanism data for research
!data/mechanism/

# ZINC22 data (very large, optional)
data/zinc20/

# Model files
*.npy
*.npz
*.pkl
*.joblib
*.pt
*.pth
*.ckpt
*.h5
*.pb

# Large data files (untracked)
*.tsv.gz
*.csv.gz
*.zip
*.tar.gz

# Large CSV files (untracked)
20220103_table.csv

# Streamlit
.streamlit/

# Logs
*.log

# Temporary files
*.tmp
*.bak
*.swp

# OS
.DS_Store
Thumbs.db
```

---

## Execution Plan

### Option A: Amend Initial Commit (RECOMMENDED)

**Pros:** Clean history from start, no orphaned objects
**Cons:** Requires force push if already pushed (not applicable here)

**Steps:**

```bash
# 1. Remove large files from Git tracking (keep on disk)
git rm --cached -r artifacts/ablation/
git rm --cached -r artifacts/figures/
git rm --cached -r artifacts/explain/
git rm --cached -r artifacts/smarts_viz/
git rm --cached -r artifacts/metrics/
git rm --cached -r archive/images_backup/
git rm --cached data/external/structures.sdf
git rm --cached data/external/lipidmaps_smiles.csv
git rm --cached 20220103_table.csv

# 2. Update .gitignore
# (copy revised .gitignore)

# 3. Stage changes
git add .gitignore

# 4. Amend initial commit
git commit --amend -m "Initial commit: Cleaned-up BBB permeability prediction project

Project structure after Phase 1 cleanup:
- Core baseline pipeline: scripts/baseline/, src/data|features|models|train|evaluate|utils/
- Analysis scripts: scripts/analysis/
- Documentation: docs/ (8 files)
- Archived legacy code: archive/old_scripts/, archive/old_src/, archive/old_web/
- Benchmark results: artifacts/reports/ (RF + Morgan: 0.9401 ± 0.0454 AUC)

Baseline benchmark established:
- Dataset: B3DB (Groups A+B)
- Split: Scaffold split (80:10:10)
- Seeds: 0, 1, 2
- Best baseline: Random Forest + Morgan fingerprints
- Test AUC: 0.9401 ± 0.0454

Git configuration:
- Tracked: Source code, archived legacy, documentation, benchmark reports
- Ignored: Generated outputs, trained models, feature cache, data splits (~3.8 GB)
- Repository size: ~50 MB (clean research code)

Status: Ready for CFFF migration

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# 5. Verify
git status
git log --oneline -1
du -sh .git
```

**Expected result:** Repository reduced from ~354 MB to ~50 MB

### Option B: Create New Clean Commit (ALTERNATIVE)

**Pros:** Non-destructive, preserves original commit
**Cons:** Large files remain in Git history

**Steps:**

```bash
# Same removal steps as Option A
# Then create new commit instead of amend
git commit -m "Git cleanup: Remove large artifacts from tracking

Removed from Git tracking (kept on disk):
- artifacts/ablation/ (~1.6 GB)
- artifacts/figures/ (~30 MB)
- artifacts/explain/ (~3.8 MB)
- artifacts/smarts_viz/ (~848 KB)
- artifacts/metrics/ (~1.4 MB)
- archive/images_backup/ (~26 MB)
- data/external/structures.sdf (277 MB)
- data/external/lipidmaps_smiles.csv (5.9 MB)
- 20220103_table.csv (3.0 MB)

Repository size: ~354 MB -> ~50 MB

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**Note:** Large files still in `.git` history, so repository size won't decrease significantly.

### Option C: Recreate Repository (CLEANEST)

**Pros:** Completely clean history, smallest repository size
**Cons:** Destructive, must reinitialize

**Steps:**

```bash
# 1. Backup current repository
cp -r .git ../bbb_project_git_backup

# 2. Remove Git
rm -rf .git

# 3. Reinitialize
git init
git config user.email "claude@anthropic.com"
git config user.name "Claude Sonnet"

# 4. Update .gitignore first
# (copy revised .gitignore)

# 5. Stage only essential files
git add .gitignore
git add scripts/
git add src/
git add docs/
git add requirements.txt README.md CLAUDE.md Dockerfile docker-compose.yml
git add archive/old_scripts/ archive/old_src/ archive/old_web/
git add archive/docs_backup/ archive/old_docs/ archive/pretraining_analysis/
git add archive/root_backup/ archive/scripts_backup/ archive/analysis/
git add artifacts/reports/
git add data/raw/

# 6. Create clean initial commit
git commit -m "Initial commit: Clean BBB permeability prediction research code

Core baseline pipeline:
- scripts/baseline/ - 3 core scripts
- scripts/analysis/ - 3 analysis scripts
- src/ - 15 modules (data, features, models, train, evaluate, utils)
- docs/ - 8 documentation files

Archived legacy code:
- archive/old_scripts/ - Legacy scripts
- archive/old_src/ - Legacy source modules
- archive/old_web/ - Streamlit web interface

Benchmark results:
- artifacts/reports/ - Benchmark summary (RF + Morgan: 0.9401 ± 0.0454 AUC)

Essential data:
- data/raw/B3DB_classification.tsv
- data/raw/B3DB_regression.tsv

Repository size: ~50 MB (clean research code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# 7. Verify
du -sh .git
git status
```

---

## Recommendation

**✅ Use Option A: Amend Initial Commit**

**Rationale:**
- Cleanest history (no large files in Git)
- Repository size: ~354 MB → ~50 MB
- No destructive operations on working directory
- Safe to proceed (not pushed to remote yet)

**Next Steps:**
1. Review this proposal
2. Approve cleanup plan
3. Execute Option A (amend initial commit)
4. Verify repository size
5. Test baseline pipeline still works
6. Ready for CFFF migration

---

## Summary Statistics

### Before Cleanup
- **Tracked files:** 511
- **Repository size:** ~354 MB
- **Large artifacts:** ~1.9 GB
- **Large data:** ~286 MB

### After Cleanup
- **Tracked files:** ~150
- **Repository size:** ~50 MB
- **Large artifacts:** 0 (all ignored)
- **Large data:** 0 (all ignored)

### Files Removed from Git (KEPT ON DISK)
- `artifacts/ablation/` - 40 files (~1.6 GB)
- `artifacts/figures/` - 48 files (~30 MB)
- `artifacts/explain/` - 13 files (~3.8 MB)
- `artifacts/smarts_viz/` - 71 files (~848 KB)
- `artifacts/metrics/` - 19 files (~1.4 MB)
- `archive/images_backup/` - 44 files (~26 MB)
- `data/external/structures.sdf` - 1 file (277 MB)
- `data/external/lipidmaps_smiles.csv` - 1 file (5.9 MB)
- `20220103_table.csv` - 1 file (3.0 MB)

**Total removed:** ~238 files, ~1.9 GB

### Files Still Tracked
- Source code: 90 files
- Archived code: 63 files
- Documentation: 39 files
- Essential data: 2 files
- Essential artifacts: 8 files (reports/)

**Total tracked:** ~202 files, ~50 MB

---

**Status:** Awaiting user approval
**Risk:** LOW (files kept on disk, only Git tracking changed)
