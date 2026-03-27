# CFFF Outputs Directory Transition Plan

**Date:** 2025-03-27
**Target:** Systematic use of CFFF outputs directory
**Status:** Planning phase - No code changes yet

---

## Overview

**Current State:**
- All outputs written to repository-local `artifacts/` directory
- Mixed with code in same directory tree
- Hardcoded paths throughout codebase

**Target State:**
- Code under `code/` (or `bbb_project/`)
- Data under `data/`
- Outputs redirected to CFFF shared storage: `/cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/`

**Approach:**
- Phase 1: Manual archiving (no code changes)
- Phase 2: Gradual path parameterization
- Phase 3: Full redirection to CFFF outputs

---

## Phase 1: Manual Archiving (Immediate - No Code Changes)

**Goal:** Use `outputs/` as experiment result archive

**Action:** Copy selected results from `artifacts/` to CFFF outputs directory

**No code changes required** - just manual archiving after experiments complete.

---

### Recommended Outputs Directory Structure

```
/cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/
├── experiments/                    # Organized by experiment date
│   ├── 2025-03-27_baseline_seed0/
│   │   ├── config.json            # Experiment configuration
│   │   ├── results/               # Experiment results
│   │   │   ├── baseline_results_master.csv
│   │   │   ├── benchmark_summary.csv
│   │   │   └── benchmark_report.txt
│   │   ├── models/                 # Trained models (optional)
│   │   │   └── seed_0_scaffold_morgan_rf.joblib
│   │   └── logs/                   # Training logs (optional)
│   │       └── training.log
│   └── 2025-03-27_baseline_full_matrix/
│       ├── config.json
│       ├── results/
│       └── models/
├── benchmarks/                     # Published benchmark results
│   ├── baseline_v1/
│   │   ├── README.md               # Benchmark description
│   │   ├── summary.csv              # Benchmark summary table
│   │   ├── details.csv              # All experiment details
│   │   └── report.txt               # Human-readable report
│   └── baseline_v2/
│       └── (same structure)
└── figures/                         # Publication figures
    ├── baseline_comparison.png
    ├── feature_importance.png
    └── confusion_matrix.png
```

---

### What to Copy Now (Manual Archiving)

#### 1. Current Benchmark Results (Priority: HIGH)

**Source:** `artifacts/reports/`

**Destination:** `outputs/benchmarks/baseline_v1/`

**Command:**
```bash
# Create directory structure
mkdir -p /cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/benchmarks/baseline_v1/

# Copy benchmark results
cp artifacts/reports/benchmark_summary.csv \
   /cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/benchmarks/baseline_v1/summary.csv

cp artifacts/reports/baseline_results_master.csv \
   /cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/benchmarks/baseline_v1/details.csv

cp artifacts/reports/benchmark_report.txt \
   /cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/benchmarks/baseline_v1/report.txt

# Create README
cat > /cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/benchmarks/baseline_v1/README.md << 'EOF'
# BBB Baseline Benchmark v1

**Date:** 2025-03-27
**Dataset:** B3DB (Groups A+B)
**Split:** Scaffold split (80:10:10)
**Seeds:** 0, 1, 2

## Best Baseline

- **Feature:** Morgan fingerprints (ECFP4, 2048 bits)
- **Model:** Random Forest
- **Test AUC:** 0.9401 ± 0.0454
- **Test F1:** 0.9391 ± 0.0270

## Files

- `summary.csv` - Aggregated results (mean ± std across seeds)
- `details.csv` - All individual experiment results
- `report.txt` - Human-readable benchmark report

## Usage

To reproduce these results:

```bash
# Activate environment
conda activate bbb-baseline

# Run full benchmark
python scripts/analysis/run_baseline_matrix.py

# Generate summary
python scripts/analysis/generate_benchmark_summary.py
```

## Next Steps

- Hyperparameter tuning on best baseline
- Ablation studies (scaffold vs random split)
- Additional features (combined fingerprints)

EOF
```

**Size:** ~50 KB

**Priority:** HIGH - These are the published baseline results

---

#### 2. Trained Models (Priority: MEDIUM)

**Source:** `artifacts/models/baselines/`

**Destination:** `outputs/experiments/2025-03-27_baseline_full_matrix/models/`

**Command:**
```bash
# Create directory
mkdir -p /cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/experiments/2025-03-27_baseline_full_matrix/models/

# Copy all trained models
cp -r artifacts/models/baselines/seed_* \
   /cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/experiments/2025-03-27_baseline_full_matrix/models/

# Create model manifest
cd /cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/experiments/2025-03-27_baseline_full_matrix/models/
find . -name "*.joblib" -o -name "*.json" > model_manifest.txt
```

**Size:** ~100 MB (18 experiments × 3 models × 2 seeds)

**Priority:** MEDIUM - Useful for reproducibility but large

---

#### 3. Experiment Configuration (Priority: HIGH)

**Source:** Generate from command line

**Destination:** `outputs/experiments/2025-03-27_baseline_full_matrix/config.json`

**Command:**
```bash
# Create experiment config
cat > /cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/experiments/2025-03-27_baseline_full_matrix/config.json << 'EOF'
{
  "experiment_name": "baseline_full_matrix",
  "date": "2025-03-27",
  "dataset": {
    "name": "B3DB",
    "groups": ["A", "B"],
    "filename": "B3DB_classification.tsv"
  },
  "split": {
    "type": "scaffold",
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1
  },
  "features": {
    "types": ["morgan", "descriptors_basic"],
    "morgan_bits": 2048,
    "morgan_radius": 2
  },
  "models": {
    "types": ["rf", "xgb", "lgbm"],
    "hyperparameters": "default"
  },
  "seeds": [0, 1, 2],
  "total_experiments": 18,
  "command": "python scripts/analysis/run_baseline_matrix.py"
}
EOF
```

**Size:** ~1 KB

**Priority:** HIGH - Essential for reproducibility

---

#### 4. Analysis Scripts (Priority: LOW)

**Source:** `scripts/analysis/` (core scripts only)

**Destination:** `outputs/experiments/2025-03-27_baseline_full_matrix/scripts/`

**Command:**
```bash
# Create directory
mkdir -p /cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/experiments/2025-03-27_baseline_full_matrix/scripts/

# Copy core analysis scripts
cp scripts/analysis/aggregate_results.py \
   scripts/analysis/generate_benchmark_summary.py \
   scripts/analysis/run_baseline_matrix.py \
   /cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/experiments/2025-03-27_baseline_full_matrix/scripts/
```

**Size:** ~30 KB

**Priority:** LOW - Scripts are in Git, but useful for version tracking

---

### Archiving Workflow (Manual)

#### After Each Experiment Run

```bash
#!/bin/bash
# archive_experiment.sh - Archive experiment results to CFFF outputs

EXPERIMENT_NAME=$1  # e.g., "baseline_seed0"
DATE=$(date +%Y-%m-%d)

# Create output directory
OUTPUT_DIR="/cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/experiments/${DATE}_${EXPERIMENT_NAME}"
mkdir -p "${OUTPUT_DIR}"/{results,models,logs,scripts}

# Copy benchmark results
cp artifacts/reports/benchmark_summary.csv "${OUTPUT_DIR}/results/"
cp artifacts/reports/baseline_results_master.csv "${OUTPUT_DIR}/results/"
cp artifacts/reports/benchmark_report.txt "${OUTPUT_DIR}/results/"

# Copy trained models (optional, comment out if not needed)
# cp -r artifacts/models/baselines/ "${OUTPUT_DIR}/models/"

# Copy analysis scripts
cp scripts/analysis/*.py "${OUTPUT_DIR}/scripts/"

# Create config.json (manual or generate)
cat > "${OUTPUT_DIR}/config.json" << EOF
{
  "experiment_name": "${EXPERIMENT_NAME}",
  "date": "${DATE}",
  "command": "python scripts/baseline/03_train_baselines.py --seed 0 --feature morgan --models rf"
}
EOF

echo "Experiment archived to: ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"
```

**Usage:**
```bash
bash archive_experiment.sh baseline_seed0
```

---

## Phase 2: Gradual Path Parameterization (Future Code Changes)

**Goal:** Make output paths configurable without breaking existing code

**Approach:** Add environment variables and config options

---

### Step 1: Add Environment Variables (No Code Changes)

**File:** `.env` or `configs/outputs.env`

```bash
# Output directory configuration
OUTPUT_BASE_DIR=/cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs
EXPERIMENT_NAME=baseline_v1

# Optional: Override local artifacts directory
# ARTIFACTS_DIR=${OUTPUT_BASE_DIR}/artifacts
```

**Usage:**
```bash
# Load environment variables
source configs/outputs.env

# Run experiment (outputs still go to artifacts/ for now)
python scripts/baseline/01_preprocess_b3db.py --seed 0

# Manually archive to outputs/
bash archive_experiment.sh baseline_seed0
```

---

### Step 2: Update Configuration Classes

**File:** `src/config/paths.py` (future change)

**Current:**
```python
@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[2]
    artifacts: Path = root / "artifacts"
    features: Path = artifacts / "features"
    models: Path = artifacts / "models"
    reports: Path = artifacts / "reports"
```

**Proposed:**
```python
import os
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[2]

    # Base output directory (configurable via environment variable)
    base_output_dir: Path = Path(os.getenv(
        "BBB_OUTPUT_DIR",
        root / "artifacts"  # Default to local artifacts/
    ))

    # Local artifacts (for intermediate results)
    artifacts: Path = root / "artifacts"
    features: Path = artifacts / "features"
    models: Path = artifacts / "models"
    reports: Path = artifacts / "reports"

    # CFFF outputs (for archived results)
    outputs: Path = base_output_dir / "outputs"
    benchmarks: Path = outputs / "benchmarks"
    experiments: Path = outputs / "experiments"
```

**Benefits:**
- ✅ Backward compatible (defaults to local artifacts/)
- ✅ Configurable via environment variable
- ✅ No code changes required initially

---

### Step 3: Add Output Manager Utility (Future)

**File:** `src/utils/output.py` (future addition)

```python
"""
Output manager for organizing experiment results.

This utility helps organize outputs into the CFFF outputs directory
structure while maintaining backward compatibility with local artifacts/.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from src.config import Paths


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    date: str
    description: str = ""
    command: str = ""
    parameters: dict = None


class OutputManager:
    """
    Manages output organization for experiments.

    Usage:
        # Initialize
        manager = OutputManager()

        # Archive current experiment
        manager.archive_experiment(
            name="baseline_seed0",
            description="Single experiment with seed 0"
        )
    """

    def __init__(self, paths: Paths = None):
        self.paths = paths or Paths()
        self.outputs_dir = self.paths.outputs
        self.experiments_dir = self.outputs_dir / "experiments"

    def get_experiment_dir(self, name: str, date: str = None) -> Path:
        """Get directory for a specific experiment."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        exp_dir = self.experiments_dir / f"{date}_{name}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def archive_benchmark_results(
        self,
        benchmark_name: str = "baseline_v1",
        description: str = ""
    ):
        """
        Archive benchmark results to CFFF outputs.

        Args:
            benchmark_name: Name of the benchmark (e.g., "baseline_v1")
            description: Description of the benchmark
        """
        date = datetime.now().strftime("%Y-%m-%d")
        benchmark_dir = self.paths.benchmarks / benchmark_name
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        # Copy benchmark results
        shutil.copy(
            self.paths.reports / "benchmark_summary.csv",
            benchmark_dir / "summary.csv"
        )
        shutil.copy(
            self.paths.reports / "baseline_results_master.csv",
            benchmark_dir / "details.csv"
        )
        shutil.copy(
            self.paths.reports / "benchmark_report.txt",
            benchmark_dir / "report.txt"
        )

        # Create README
        readme_content = f"""# BBB Baseline Benchmark: {benchmark_name}

**Date:** {date}
{description}

## Files

- `summary.csv` - Aggregated results (mean ± std)
- `details.csv` - All individual results
- `report.txt` - Human-readable report

## Best Baseline

See `report.txt` for details.
"""
        (benchmark_dir / "README.md").write_text(readme_content)

        print(f"Benchmark archived to: {benchmark_dir}")
        return benchmark_dir

    def archive_experiment(
        self,
        name: str,
        config: ExperimentConfig,
        copy_models: bool = False,
        copy_scripts: bool = True
    ):
        """
        Archive a complete experiment to CFFF outputs.

        Args:
            name: Experiment name
            config: Experiment configuration
            copy_models: Whether to copy trained models
            copy_scripts: Whether to copy analysis scripts
        """
        exp_dir = self.get_experiment_dir(name, config.date)

        # Create subdirectories
        (exp_dir / "results").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)

        if copy_models:
            (exp_dir / "models").mkdir(exist_ok=True)

        if copy_scripts:
            (exp_dir / "scripts").mkdir(exist_ok=True)

        # Copy results
        for file in ["benchmark_summary.csv", "baseline_results_master.csv", "benchmark_report.txt"]:
            src = self.paths.reports / file
            if src.exists():
                shutil.copy(src, exp_dir / "results" / file)

        # Copy models
        if copy_models:
            shutil.copytree(
                self.paths.models / "baselines",
                exp_dir / "models" / "baselines",
                dirs_exist_ok=True
            )

        # Copy scripts
        if copy_scripts:
            scripts_dir = Path("scripts/analysis")
            for script in scripts_dir.glob("*.py"):
                shutil.copy(script, exp_dir / "scripts" / script.name)

        # Save config
        config_dict = {
            "name": config.name,
            "date": config.date,
            "description": config.description,
            "command": config.command,
            "parameters": config.parameters or {}
        }
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"Experiment archived to: {exp_dir}")
        return exp_dir
```

**Usage (future):**
```python
from src.utils.output import OutputManager, ExperimentConfig

# After running experiments
manager = OutputManager()
manager.archive_benchmark_results(
    benchmark_name="baseline_v1",
    description="RF + Morgan: 0.9401 ± 0.0454 AUC"
)

# Archive specific experiment
config = ExperimentConfig(
    name="baseline_seed0",
    date="2025-03-27",
    description="Single experiment with seed 0",
    command="python scripts/baseline/03_train_baselines.py --seed 0"
)
manager.archive_experiment(name="baseline_seed0", config=config)
```

---

## Phase 3: Full Redirection (Future Code Changes)

**Goal:** Redirect all outputs to CFFF directory by default

**Approach:** Change default paths in configuration

---

### Update Path Defaults (Future)

**File:** `src/config/paths.py` (future change)

```python
@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[2]

    # CFFF outputs directory (new default)
    base_output_dir: Path = Path(os.getenv(
        "BBB_OUTPUT_DIR",
        Path("/cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs")
    ))

    # Local artifacts (for intermediate/cached results)
    artifacts: Path = root / "artifacts"
    features: Path = artifacts / "features"
    models: Path = artifacts / "models"

    # CFFF outputs (for final results)
    outputs: Path = base_output_dir
    reports: Path = outputs / "reports"  # Changed from artifacts/reports/
    benchmarks: Path = outputs / "benchmarks"
    experiments: Path = outputs / "experiments"
```

---

### Update Scripts to Use Output Manager (Future)

**File:** `scripts/analysis/generate_benchmark_summary.py` (future change)

```python
# At end of script, add automatic archiving

from src.utils.output import OutputManager

def main():
    # ... existing code ...

    # Generate benchmark summary
    # ... existing code ...

    # NEW: Archive to CFFF outputs
    manager = OutputManager()
    manager.archive_benchmark_results(
        benchmark_name="baseline_v1",
        description="RF + Morgan: 0.9401 ± 0.0454 AUC"
    )

    print(f"Benchmark results archived to: {manager.paths.benchmarks / 'baseline_v1'}")
```

---

## Summary of Transition Plan

### Phase 1: Manual Archiving (Immediate) ⭐

**Action:** Copy results to CFFF outputs manually

**What to do:**
1. Create directory structure in `/cpfs01/projects-HDD/.../outputs/`
2. Copy benchmark results after each run
3. Use `archive_experiment.sh` script for automation

**Benefits:**
- ✅ No code changes
- ✅ Immediate use of CFFF outputs
- ✅ Organized experiment tracking

**When:** NOW

---

### Phase 2: Path Parameterization (Future)

**Action:** Make output paths configurable

**What to do:**
1. Add `BBB_OUTPUT_DIR` environment variable
2. Update `src/config/paths.py` to support environment variable
3. Create `OutputManager` utility class
4. Add manual archiving step to scripts

**Benefits:**
- ✅ Flexible output location
- ✅ Backward compatible
- ✅ Gradual migration

**When:** AFTER Phase 1 is established

---

### Phase 3: Full Redirection (Future)

**Action:** Redirect all outputs to CFFF directory

**What to do:**
1. Change default paths in `src/config/paths.py`
2. Update all scripts to use new paths
3. Add automatic archiving to workflow
4. Keep local `artifacts/` for cache only

**Benefits:**
- ✅ All outputs in CFFF shared storage
- ✅ Automatic archiving
- ✅ Clean separation: code, data, outputs

**When:** AFTER Phase 2 is tested

---

## Immediate Actions (Phase 1)

### 1. Create Outputs Directory Structure

```bash
# On CFFF
mkdir -p /cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/{benchmarks,experiments,figures}
```

### 2. Archive Current Benchmark Results

```bash
# Copy benchmark v1 results
OUTPUT_DIR="/cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/benchmarks/baseline_v1"
mkdir -p "${OUTPUT_DIR}"

cp artifacts/reports/benchmark_summary.csv "${OUTPUT_DIR}/summary.csv"
cp artifacts/reports/baseline_results_master.csv "${OUTPUT_DIR}/details.csv"
cp artifacts/reports/benchmark_report.txt "${OUTPUT_DIR}/report.txt"

# Create README
cat > "${OUTPUT_DIR}/README.md" << 'EOF'
# BBB Baseline Benchmark v1

**Date:** 2025-03-27
**Best Baseline:** RF + Morgan (0.9401 ± 0.0454 AUC)

See `report.txt` for details.
EOF
```

### 3. Create Archiving Script

```bash
# Save as: scripts/archive_experiment.sh
chmod +x scripts/archive_experiment.sh
```

(Use script from Phase 1 section above)

### 4. Update Workflow

```bash
# After running experiments
python scripts/analysis/generate_benchmark_summary.py

# Archive to CFFF outputs
bash scripts/archive_experiment.sh baseline_$(date +%Y%m%d)
```

---

## Recommended Outputs Directory Structure

```
/cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs/
│
├── benchmarks/                     # Published benchmark results
│   ├── baseline_v1/                # First official baseline
│   │   ├── README.md               # Description
│   │   ├── summary.csv             # Aggregated results
│   │   ├── details.csv             # All experiments
│   │   └── report.txt              # Human-readable
│   └── baseline_v2/                # Future improvements
│
├── experiments/                    # Individual experiment runs
│   ├── 2025-03-27_baseline_full_matrix/
│   │   ├── config.json             # Experiment config
│   │   ├── results/               # Output files
│   │   ├── models/                 # Trained models
│   │   ├── logs/                   # Training logs
│   │   └── scripts/                # Code versions
│   └── 2025-03-28_hyperparameter_tuning/
│
└── figures/                         # Publication figures
    ├── baseline_comparison.png
    └── feature_importance.png
```

---

## What to Copy Now (Phase 1)

| Source | Destination | Priority | Size |
|--------|-------------|----------|------|
| `artifacts/reports/benchmark_summary.csv` | `outputs/benchmarks/baseline_v1/summary.csv` | HIGH | 1 KB |
| `artifacts/reports/baseline_results_master.csv` | `outputs/benchmarks/baseline_v1/details.csv` | HIGH | 50 KB |
| `artifacts/reports/benchmark_report.txt` | `outputs/benchmarks/baseline_v1/report.txt` | HIGH | 10 KB |
| `artifacts/models/baselines/` | `outputs/experiments/2025-03-27_baseline_full_matrix/models/` | MEDIUM | 100 MB |
| Scripts config | `outputs/experiments/.../config.json` | HIGH | 1 KB |
| Analysis scripts | `outputs/experiments/.../scripts/` | LOW | 30 KB |

**Total:** ~150 MB (with models) or ~50 KB (without models)

---

## What Could Be Redirected Later (Phase 2-3)

| Current Location | Future Location | Priority |
|------------------|-----------------|----------|
| `artifacts/reports/` | `outputs/reports/` | HIGH |
| `artifacts/models/` | `outputs/experiments/.../models/` | MEDIUM |
| `artifacts/logs/` | `outputs/experiments/.../logs/` | LOW |
| `artifacts/features/` | Keep local (cache) | N/A |
| `data/splits/` | Keep local (regenerable) | N/A |

---

## Benefits of This Approach

### 1. Separation of Concerns

- **Code:** Remains in repository
- **Data:** Remains in `data/` (raw and splits)
- **Outputs:** Archived to CFFF shared storage

### 2. No Breaking Changes

- Phase 1: No code changes, manual archiving
- Phase 2: Configurable paths, backward compatible
- Phase 3: Full redirection when ready

### 3. Reproducibility

- Each experiment gets dated directory
- Config saved with results
- Scripts version tracked
- Models preserved (optional)

### 4. CFFF-Friendly

- Large outputs in shared storage (not repo)
- Easy to share results with team
- Organized by experiment
- Persistent across sessions

---

## Next Steps

### Immediate (Phase 1)

1. ✅ Create outputs directory structure
2. ✅ Archive current benchmark results
3. ✅ Create `archive_experiment.sh` script
4. ✅ Update workflow to archive after runs

### Future (Phase 2)

1. Add `BBB_OUTPUT_DIR` environment variable
2. Update `src/config/paths.py` to support it
3. Create `OutputManager` utility
4. Add manual archiving step to scripts

### Later (Phase 3)

1. Change default paths to CFFF outputs
2. Update all scripts to use OutputManager
3. Automatic archiving after each run
4. Keep local `artifacts/` for cache only

---

**Status:** Planning complete ✅
**Phase 1:** Ready to implement (no code changes)
**Phase 2:** Future (configurable paths)
**Phase 3:** Future (full redirection)
