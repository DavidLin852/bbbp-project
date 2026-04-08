#!/usr/bin/env python
"""
Run supervised GNN baselines for BBB permeability prediction.

Executes GNN experiments across:
  - 3 models: GCN, GIN, GAT
  - 2 tasks: classification, regression
  - 5 seeds: 0, 1, 2, 3, 4
  - Scaffold split

CUDA-first design: GPU is the intended execution environment.
CPU fallback is supported but not the primary target.

Usage:
    # Dry run - show all commands without executing
    python scripts/gnn/run_gnn_benchmark.py --dry_run

    # Full run (classification only)
    python scripts/gnn/run_gnn_benchmark.py --tasks classification

    # Full run (regression only)
    python scripts/gnn/run_gnn_benchmark.py --tasks regression

    # Full run (both tasks)
    python scripts/gnn/run_gnn_benchmark.py

    # Custom seeds (faster iteration)
    python scripts/gnn/run_gnn_benchmark.py --seeds 0,1

    # Single model
    python scripts/gnn/run_gnn_benchmark.py --models gcn

    # Skip existing (resume interrupted runs)
    python scripts/gnn/run_gnn_benchmark.py --skip_existing

    # Custom batch size (reduce if GPU OOM)
    python scripts/gnn/run_gnn_benchmark.py --batch_size 32

    # Use specific GPU
    python scripts/gnn/run_gnn_benchmark.py --gpu 0

CFFF / Cluster usage:
    # Interactive GPU job on CFFF
    python scripts/gnn/run_gnn_benchmark.py --tasks classification --gpu 0

    # Or submit as a batch script
    # See docs/GNN_STAGE.md for CFFF batch job examples
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import Paths
from src.gnn.models import GNNConfig
from src.gnn.train import train_gnn, get_device, print_device_info, clear_gpu_memory

# GNN models
GNN_MODELS = ["gcn", "gin", "gat"]
TASKS = ["classification", "regression"]


def run_single(
    model_type: str,
    split_dir: Path,
    seed: int,
    task: str,
    config: GNNConfig,
    output_base: Path,
    device: str,
    num_workers: int,
    skip_existing: bool,
    pretrained_encoder_path: str | None = None,
) -> dict | None:
    """
    Run a single GNN experiment.

    Returns result dict on success, None on skip/failure.
    """
    model_dir = output_base / f"seed_{seed}" / task / model_type

    if skip_existing:
        result_file = model_dir / "result.json"
        if result_file.exists():
            try:
                json.loads(result_file.read_text())
                print(f"  SKIPPED (exists): {model_type}/{task}/seed={seed}")
                return {}
            except Exception:
                pass

    # Clear GPU memory between experiments
    clear_gpu_memory()

    try:
        from src.gnn.dataset import B3DBGNNDataset

        dataset = B3DBGNNDataset(
            split_dir=split_dir,
            task=task,
        )

        print(f"  Dataset: train={dataset.train_size}, val={dataset.val_size}, "
              f"test={dataset.test_size}, features={dataset.get_input_dim()}")

        result = train_gnn(
            model_type=model_type,
            dataset=dataset,
            seed=seed,
            task=task,
            config=config,
            output_dir=output_base,
            device=device,
            num_workers=num_workers,
            verbose=True,
            pretrained_encoder_path=pretrained_encoder_path,
        )

        return result.to_dict()

    except Exception as e:
        print(f"  FAILED: {model_type}/{task}/seed={seed} - {e}")
        import traceback
        traceback.print_exc()
        return None


def aggregate_results(output_base: Path, tasks: list[str]) -> pd.DataFrame:
    """
    Aggregate results across all seeds and models.

    Returns a DataFrame with mean ± std across seeds.
    """
    rows = []

    for task in tasks:
        for model in GNN_MODELS:
            seed_results: list[dict] = []

            for seed in range(5):
                result_file = output_base / f"seed_{seed}" / task / model / "result.json"
                if result_file.exists():
                    try:
                        data = json.loads(result_file.read_text())
                        seed_results.append(data)
                    except Exception:
                        continue

            if not seed_results:
                continue

            row = {"model": model, "task": task, "n_seeds": len(seed_results)}

            if task == "classification":
                for prefix in ["train", "val", "test"]:
                    for metric in ["auc", "f1", "loss"]:
                        key = f"{prefix}_{metric}"
                        values = [r[key] for r in seed_results if key in r]
                        if values:
                            row[f"{prefix}_{metric}_mean"] = np.mean(values)
                            row[f"{prefix}_{metric}_std"] = np.std(values)
            else:
                for prefix in ["train", "val", "test"]:
                    for metric in ["r2", "rmse", "mae"]:
                        key = f"{prefix}_{metric}"
                        values = [r[key] for r in seed_results if key in r]
                        if values:
                            row[f"{prefix}_{metric}_mean"] = np.mean(values)
                            row[f"{prefix}_{metric}_std"] = np.std(values)

            rows.append(row)

    df = pd.DataFrame(rows)

    if "test_auc_mean" in df.columns:
        df = df.sort_values("test_auc_mean", ascending=False)
    elif "test_r2_mean" in df.columns:
        df = df.sort_values("test_r2_mean", ascending=False)

    return df


def print_comparison_table(
    gnn_df: pd.DataFrame | None,
    baseline_path: Path,
    task: str,
    metric_col: str,
    title: str,
):
    """Print GNN results alongside classical baseline reference."""
    print(f"\n--- {title} ---")

    cols = [c for c in gnn_df.columns if c in ["model", "n_seeds", metric_col, metric_col.replace("_mean", "_std")]]
    if gnn_df is not None and len(gnn_df) > 0:
        print(gnn_df[cols].to_string(index=False))

    if baseline_path.exists():
        try:
            ref_df = pd.read_csv(baseline_path)
            ref_cols = [c for c in ref_df.columns
                        if c in ["model_name", metric_col, metric_col.replace("_mean", "_std")]]
            top_ref = ref_df.sort_values(metric_col, ascending=False).head(5)
            print(f"\nClassical Baseline Reference:")
            print(top_ref[ref_cols].to_string(index=False))
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Run supervised GNN baselines")
    parser.add_argument(
        "--seeds", type=str, default="0,1,2,3,4",
        help="Comma-separated seeds (default: 0,1,2,3,4)"
    )
    parser.add_argument(
        "--models", type=str, default="gcn,gin,gat",
        help="Comma-separated models (default: gcn,gin,gat)"
    )
    parser.add_argument(
        "--tasks", type=str, default="classification,regression",
        help="Comma-separated tasks (default: classification,regression)"
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip experiments with existing result.json files"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print plan without executing"
    )
    # ---- CUDA / Platform options ----
    parser.add_argument(
        "--device", type=str, default="auto",
        help="'cuda' (GPU), 'cpu', or 'auto' (default: auto-detect)"
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="CUDA GPU device ID (e.g., 0, 1). Overrides --device."
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader num_workers. 0=single process (safe, default). "
             "Increase on cluster for speed but test first."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size (default: 64). Reduce to 32 or 16 if GPU OOM."
    )
    # ---- Model hyperparameters ----
    parser.add_argument(
        "--hidden_dim", type=int, default=128,
        help="GNN hidden dimension (default: 128)"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3,
        help="Number of GNN layers (default: 3)"
    )
    parser.add_argument(
        "--heads", type=int, default=4,
        help="Attention heads for GAT (default: 4)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout rate (default: 0.3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=300,
        help="Maximum training epochs (default: 300)"
    )
    parser.add_argument(
        "--patience", type=int, default=30,
        help="Early stopping patience (default: 30)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Weight decay (default: 1e-4)"
    )
    # ---- Output ----
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory (default: artifacts/models/gnn)"
    )
    parser.add_argument(
        "--pretrained_encoder", type=str, default=None,
        help="Path to pretrained backbone state_dict (e.g. artifacts/models/pretrain/gin_full/gin_pretrained_backbone.pt). "
             "Only supported for 'gin' model."
    )

    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]

    # Validate
    for m in models:
        if m not in GNN_MODELS:
            print(f"ERROR: Unknown model '{m}'. Available: {GNN_MODELS}")
            sys.exit(1)

    # ---- Device setup ----
    import torch

    if args.gpu is not None:
        if not torch.cuda.is_available():
            print("ERROR: --gpu specified but CUDA is not available")
            sys.exit(1)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device_str = "cuda"
    elif args.device == "auto":
        device_str = "auto"
    else:
        device_str = args.device

    dev = get_device(device_str)

    # Config
    config = GNNConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Output directory
    P = Paths()
    output_base = Path(args.output_dir) if args.output_dir else P.models / "gnn"

    print("=" * 70)
    print("GNN BASELINE BENCHMARK")
    print("=" * 70)
    print(f"Models:    {models}")
    print(f"Tasks:     {tasks}")
    print(f"Seeds:     {seeds}")
    print(f"Output:    {output_base}")
    print(f"Workers:   {args.num_workers}")
    print(f"Device:    {dev}")
    if args.pretrained_encoder:
        print(f" Pretrained: {args.pretrained_encoder}")
    print(f"Config:    hidden={config.hidden_dim}, layers={config.num_layers}, "
          f"heads={config.heads}, dropout={config.dropout}")
    print(f"Training:  lr={config.lr}, wd={config.weight_decay}, "
          f"epochs={config.epochs}, patience={config.patience}, batch={config.batch_size}")
    print("=" * 70)

    # Print CUDA info if available
    if dev.type == "cuda":
        print_device_info(dev)
        print(f"  Visible GPUs: {torch.cuda.device_count()}")
        print(f"  GPU set by user: {args.gpu}")
        print()

    if args.dry_run:
        print("\n[DRY RUN] Experiments that would be executed:\n")
        for seed in seeds:
            for task_name in tasks:
                split_path = P.data_splits / f"seed_{seed}" / f"{task_name}_scaffold"
                for model in models:
                    print(f"  {model.upper()} | {task_name} | seed={seed} | "
                          f"split={split_path.name} | device={dev}")
        return

    total = len(seeds) * len(tasks) * len(models)
    current = 0
    all_results: list[dict] = []

    for seed in seeds:
        for task_name in tasks:
            split_path = P.data_splits / f"seed_{seed}" / f"{task_name}_scaffold"

            if not split_path.exists():
                print(f"\nWARNING: Split not found at {split_path}, skipping {task_name} seed={seed}")
                continue

            for model in models:
                current += 1
                print(f"\n[{current}/{total}] {model.upper()} | {task_name} | seed={seed}")

                result = run_single(
                    model_type=model,
                    split_dir=split_path,
                    seed=seed,
                    task=task_name,
                    config=config,
                    output_base=output_base,
                    device=device_str,
                    num_workers=args.num_workers,
                    skip_existing=args.skip_existing,
                    pretrained_encoder_path=args.pretrained_encoder,
                )

                if result:
                    all_results.append(result)

    # ---- Aggregate and report ----
    print(f"\n{'=' * 70}")
    print("AGGREGATING RESULTS")
    print("=" * 70)

    gnn_df = aggregate_results(output_base, tasks)

    # Save
    report_dir = P.reports / "gnn"
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"gnn_benchmark_scaffold_{timestamp}.csv"
    gnn_df.to_csv(report_file, index=False)
    print(f"\nReport: {report_file}")

    # Print comparison tables
    if "classification" in tasks:
        cls_df = gnn_df[gnn_df["task"] == "classification"].copy() if len(gnn_df) > 0 else None
        print_comparison_table(
            cls_df,
            P.reports / "cls_benchmark_scaffold.csv",
            "classification",
            "test_auc_mean",
            "Classification Results (GNN vs Classical Baselines)",
        )

    if "regression" in tasks:
        reg_df = gnn_df[gnn_df["task"] == "regression"].copy() if len(gnn_df) > 0 else None
        print_comparison_table(
            reg_df,
            P.reports / "reg_benchmark_scaffold.csv",
            "regression",
            "test_r2_mean",
            "Regression Results (GNN vs Classical Baselines)",
        )

    print(f"\n{'=' * 70}")
    print("GNN BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import os
    main()
