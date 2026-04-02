#!/usr/bin/env python
"""
Run the full formal classical baseline sweep.

Executes the complete experiment matrix:
  - 2 tasks (classification, regression)
  - 4 features (morgan, descriptors_basic, maccs, fp2)
  - 6 models per task
  - 5 seeds (0,1,2,3,4)
  - 1 split (scaffold)

Then runs aggregation and generates benchmark tables.

Usage:
    # Dry run - show all commands without executing
    python scripts/analysis/run_classical_sweep.py --dry_run

    # Full run
    python scripts/analysis/run_classical_sweep.py

    # Custom seeds
    python scripts/analysis/run_classical_sweep.py --seeds 0,1,2

    # Skip stages
    python scripts/analysis/run_classical_sweep.py --skip_preprocess --skip_features

Platform note:
    On CFFF, submit as a batch job that runs this script.
    On local, run directly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"
BASELINE = SCRIPTS / "baseline"
ANALYSIS = SCRIPTS / "analysis"

FEATURES = ["morgan", "descriptors_basic", "maccs", "fp2"]
CLS_MODELS = ["rf", "xgb", "lgbm", "svm", "lr", "knn"]
REG_MODELS = ["rf_reg", "xgb_reg", "lgbm_reg", "svm_reg", "ridge", "knn_reg"]
TASKS = ["classification", "regression"]


def run(cmd: list[str], desc: str, dry_run: bool, capture: bool = True):
    """Run a command, printing status."""
    if dry_run:
        print(f"  [DRY] {' '.join(cmd)}")
        return True

    full_cmd = [sys.executable] + cmd
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(
        full_cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=capture,
        text=True,
    )
    if result.returncode != 0:
        print(f"  FAILED: {desc}")
        if result.stdout:
            print(result.stdout[-500:])
        if result.stderr:
            print(result.stderr[-500:])
        return False
    print(f"  OK: {desc}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full classical baseline sweep")
    parser.add_argument(
        "--seeds", type=str, default="0,1,2,3,4",
        help="Comma-separated seeds (default: 0,1,2,3,4)"
    )
    parser.add_argument(
        "--features", type=str, default=None,
        help="Comma-separated features (default: all 4)"
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Comma-separated tasks (default: both)"
    )
    parser.add_argument(
        "--skip_preprocess", action="store_true",
        help="Skip preprocessing"
    )
    parser.add_argument(
        "--skip_features", action="store_true",
        help="Skip feature computation"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing"
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    features = [f.strip() for f in (args.features or ",".join(FEATURES)).split(",")]
    tasks = [t.strip() for t in (args.tasks or ",".join(TASKS)).split(",")]

    n_cls = len(tasks) * len(seeds) * len(features) * len(CLS_MODELS)
    n_reg = len(tasks) * len(seeds) * len(features) * len(REG_MODELS)

    print("=" * 70)
    print("CLASSICAL BASELINE SWEEP")
    print("=" * 70)
    print(f"Seeds:    {seeds}")
    print(f"Features: {features}")
    print(f"Tasks:    {tasks}")
    print(f"CLS runs: {n_cls}  |  REG runs: {n_reg}")
    print(f"Total:    {n_cls + n_reg} training runs")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:\n")

    failures = []

    # Stage 1: Preprocess
    if not args.skip_preprocess:
        print("\n[Stage 1/4] Preprocessing...")
        for seed in seeds:
            for task in tasks:
                for split in ["scaffold"]:
                    cmd = [
                        "python", str(BASELINE / "01_preprocess_b3db.py"),
                        "--seed", str(seed),
                        "--split_type", split,
                        "--task", task,
                    ]
                    ok = run(cmd, f"preprocess seed={seed} task={task}", args.dry_run)
                    if not ok:
                        failures.append(f"preprocess_seed_{seed}_{task}")
    else:
        print("\n[Stage 1/4] Preprocessing SKIPPED")

    # Stage 2: Compute features
    if not args.skip_features:
        print("\n[Stage 2/4] Computing features...")
        for seed in seeds:
            for task in tasks:
                for feature in features:
                    cmd = [
                        "python", str(BASELINE / "02_compute_features.py"),
                        "--seed", str(seed),
                        "--split", "scaffold",
                        "--task", task,
                        "--feature", feature,
                    ]
                    ok = run(cmd, f"features seed={seed} task={task} feature={feature}", args.dry_run)
                    if not ok:
                        failures.append(f"features_seed_{seed}_{task}_{feature}")
    else:
        print("\n[Stage 2/4] Features SKIPPED")

    # Stage 3: Train classification models
    if "classification" in tasks:
        print("\n[Stage 3/4a] Training classification models...")
        for seed in seeds:
            for feature in features:
                cmd = [
                    "python", str(BASELINE / "04_train_all.py"),
                    "--seed", str(seed),
                    "--task", "classification",
                    "--feature", feature,
                    "--models", ",".join(CLS_MODELS),
                ]
                ok = run(cmd, f"cls seed={seed} feature={feature}", args.dry_run)
                if not ok:
                    failures.append(f"cls_seed_{seed}_{feature}")

    # Stage 3b: Train regression models
    if "regression" in tasks:
        print("\n[Stage 3/4b] Training regression models...")
        for seed in seeds:
            for feature in features:
                cmd = [
                    "python", str(BASELINE / "04_train_all.py"),
                    "--seed", str(seed),
                    "--task", "regression",
                    "--feature", feature,
                    "--models", ",".join(REG_MODELS),
                ]
                ok = run(cmd, f"reg seed={seed} feature={feature}", args.dry_run)
                if not ok:
                    failures.append(f"reg_seed_{seed}_{feature}")

    # Stage 4: Aggregate and report
    print("\n[Stage 4/4] Aggregating results...")
    agg_cmd = [
        "python", str(ANALYSIS / "aggregate_results.py"),
        "--task", "all",
    ]
    run(agg_cmd, "aggregate_results.py", args.dry_run, capture=False)

    # Summary
    print("\n" + "=" * 70)
    if args.dry_run:
        print("DRY RUN COMPLETE - no commands were executed")
    elif failures:
        print(f"SWEEP COMPLETE with {len(failures)} failures:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("SWEEP COMPLETE - all runs succeeded")
    print("=" * 70)
    print("\nResults:")
    print("  artifacts/reports/cls_benchmark_scaffold.csv   -- Classification benchmark")
    print("  artifacts/reports/reg_benchmark_scaffold.csv   -- Regression benchmark")
    print("  artifacts/reports/cls_results_master.csv      -- All classification results")
    print("  artifacts/reports/reg_results_master.csv     -- All regression results")


if __name__ == "__main__":
    main()
