#!/usr/bin/env python
"""
Aggregate baseline experiment results.

Scans both classification (baselines/) and regression (regression/) directories
and produces task-specific master tables.

Usage:
    python scripts/analysis/aggregate_results.py --task all
    python scripts/analysis/aggregate_results.py --task classification
    python scripts/analysis/aggregate_results.py --task regression
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Field mappings
# ---------------------------------------------------------------------------
CLS_FIELDS = [
    "train_auc", "train_accuracy", "train_f1",
    "val_auc", "val_accuracy", "val_f1",
    "test_auc", "test_accuracy", "test_f1",
]

REG_FIELDS = [
    "train_mse", "train_rmse", "train_mae", "train_r2",
    "val_mse", "val_rmse", "val_mae", "val_r2",
    "test_mse", "test_rmse", "test_mae", "test_r2",
]


def load_comparison_json(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def extract_result(comparison: dict, fields: list[str]) -> list[dict]:
    results = []
    for r in comparison.get("results", []):
        row = {
            "model_name": r.get("model_name"),
            "feature_type": r.get("feature_type"),
        }
        for f in fields:
            row[f] = r.get(f)
        results.append(row)
    return results


def scan_directory(base_dir: Path, task: str) -> list[dict]:
    """
    Recursively scan for comparison.json files.

    Directory structure:
      artifacts/models/baselines/seed_N/split/feature/     -> classification
      artifacts/models/regression/seed_N/split/feature/   -> regression
    """
    all_results = []
    model_base = base_dir / "models"
    if not model_base.exists():
        return all_results

    for comp_file in model_base.rglob("comparison.json"):
        parts = comp_file.parts
        is_regression = "baselines" not in parts

        if task == "classification" and is_regression:
            continue
        if task == "regression" and not is_regression:
            continue

        try:
            seed_idx = next(i for i, p in enumerate(parts) if p.startswith("seed_"))
            seed = int(parts[seed_idx].replace("seed_", ""))
            split = parts[seed_idx + 1]
            feature = parts[seed_idx + 2]
        except (StopIteration, IndexError, ValueError):
            print(f"Warning: Skipping unexpected path: {comp_file}")
            continue

        comparison = load_comparison_json(comp_file)
        fields = REG_FIELDS if is_regression else CLS_FIELDS
        results = extract_result(comparison, fields)

        for r in results:
            r["seed"] = seed
            r["split_type"] = split
            r["feature"] = feature
            r["task"] = "regression" if is_regression else "classification"

        all_results.extend(results)

    return all_results


def make_master(results: list[dict], fields: list[str], sort_key: str, sort_asc: bool) -> pd.DataFrame:
    df = pd.DataFrame(results)
    cols = ["seed", "split_type", "feature", "model_name"] + fields
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(sort_key, ascending=sort_asc)
    return df


def make_summary(df: pd.DataFrame, agg_metrics: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg_dict = {m: ["mean", "std", "count"] for m in agg_metrics}
    agg = df.groupby(["feature", "model_name"]).agg(agg_dict).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    first_metric = agg_metrics[0]
    agg = agg.sort_values(f"{first_metric}_mean", ascending=False)
    return agg


def main():
    parser = argparse.ArgumentParser(description="Aggregate baseline results")
    parser.add_argument("--base_dir", type=str, default="artifacts")
    parser.add_argument("--output_dir", type=str, default="artifacts/reports")
    parser.add_argument(
        "--task", type=str, default="all",
        choices=["classification", "regression", "all"],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--feature", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = scan_directory(base_dir, args.task)
    print(f"Found {len(results)} results")

    if not results:
        print("No results found.")
        return

    df_all = pd.DataFrame(results)

    if args.seed is not None:
        df_all = df_all[df_all["seed"] == args.seed]
    if args.split is not None:
        df_all = df_all[df_all["split_type"] == args.split]
    if args.feature is not None:
        df_all = df_all[df_all["feature"] == args.feature]

    print(f"Filtered to {len(df_all)} results")

    # Classification
    cls_df = df_all[df_all["task"] == "classification"].copy()
    if not cls_df.empty:
        cls_master = make_master(cls_df.to_dict("records"), CLS_FIELDS, "test_auc", False)
        cls_master.to_csv(output_dir / "cls_results_master.csv", index=False)

        agg_cls = make_summary(cls_master, ["test_auc", "test_f1"])
        agg_cls.to_csv(output_dir / "cls_summary_by_model.csv", index=False)

        scaffold_cls = cls_master[cls_master["split_type"] == "scaffold"]
        scaffold_cls.to_csv(output_dir / "cls_results_scaffold.csv", index=False)

        agg_scaffold_cls = make_summary(scaffold_cls, ["test_auc", "test_f1"])
        agg_scaffold_cls.to_csv(output_dir / "cls_benchmark_scaffold.csv", index=False)

        print(f"\n=== Classification (scaffold) ===")
        print(agg_scaffold_cls.head(5).to_string(index=False))

    # Regression
    reg_df = df_all[df_all["task"] == "regression"].copy()
    if not reg_df.empty:
        reg_master = make_master(reg_df.to_dict("records"), REG_FIELDS, "test_r2", False)
        reg_master.to_csv(output_dir / "reg_results_master.csv", index=False)

        agg_reg = make_summary(reg_master, ["test_r2", "test_rmse", "test_mae"])
        agg_reg.to_csv(output_dir / "reg_summary_by_model.csv", index=False)

        scaffold_reg = reg_master[reg_master["split_type"] == "scaffold"]
        scaffold_reg.to_csv(output_dir / "reg_results_scaffold.csv", index=False)

        agg_scaffold_reg = make_summary(scaffold_reg, ["test_r2", "test_rmse", "test_mae"])
        agg_scaffold_reg.to_csv(output_dir / "reg_benchmark_scaffold.csv", index=False)

        print(f"\n=== Regression (scaffold) ===")
        print(agg_scaffold_reg.head(5).to_string(index=False))

    print(f"\nOutputs in: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
