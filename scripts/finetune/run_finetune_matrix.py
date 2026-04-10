#!/usr/bin/env python
"""
Run the fine-tuning experiment matrix (Stage 4).

Organizes and executes all 204 fine-tuning experiments across:
  Group A: GNN Fine-tuning (pretrained GNN → head, end-to-end)
  Group B: Embedding → LightGBM (pretrained GNN/Transformer → frozen emb → LGBM)
  Group C: Embedding + Feature → LightGBM (embedding + fingerprints → LGBM)
  Group D: Transformer Fine-tuning (pretrained encoder → head, end-to-end)

Phase 1 (68 experiments × 5 seeds):
  Groups A, B, D
Phase 2 (136 experiments × 5 seeds):
  Group C

Usage:
    # List all experiments
    python scripts/finetune/run_finetune_matrix.py --list

    # Dry run
    python scripts/finetune/run_finetune_matrix.py --dry_run

    # Run Phase 1 (Groups A+B+D)
    python scripts/finetune/run_finetune_matrix.py --phase 1

    # Run Phase 2 (Group C)
    python scripts/finetune/run_finetune_matrix.py --phase 2

    # Run specific group
    python scripts/finetune/run_finetune_matrix.py --group A

    # Run single experiment
    python scripts/finetune/run_finetune_matrix.py --run 1

    # Custom seeds
    python scripts/finetune/run_finetune_matrix.py --seeds 0,1

    # Skip existing (for resuming)
    python scripts/finetune/run_finetune_matrix.py --phase 1 --skip_existing

    # Extract embeddings first (required for Groups B and C)
    python scripts/finetune/run_finetune_matrix.py --extract_only
    python scripts/finetune/run_finetune_matrix.py --extract_only --group B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

# Add project root
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.finetune.finetune_config import (
    build_experiment_matrix,
    count_experiments,
    get_pretrain_config,
)
from src.finetune.embedding_pipeline import train_lgbm_on_embeddings


# ============================================================
# GNN Config (for Group A)
# ============================================================

class GNNConfig:
    """Minimal GNN config for fine-tuning."""
    def __init__(
        self,
        hidden_dim=128,
        num_layers=3,
        heads=4,
        dropout=0.3,
        lr=1e-4,
        weight_decay=1e-5,
        epochs=200,
        patience=25,
        batch_size=64,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size


# ============================================================
# Group A: GNN Fine-tuning
# ============================================================

def run_group_a(exp: dict, seed: int, output_dir: Path, device: str, skip_existing: bool) -> dict | None:
    """Fine-tune a pretrained GNN on BBB (Group A)."""
    from src.gnn.train import train_gnn
    from src.gnn.dataset import B3DBGNNDataset

    pretrain_id = exp["pretrain_id"]
    task = exp["task"]
    model_type = exp["model_type"]

    backbone_path = project_root / exp["backbone_path"]
    if not backbone_path.exists():
        print(f"  [SKIP] Backbone not found: {backbone_path}")
        return None

    split_dir = project_root / "data" / "splits" / f"seed_{seed}" / f"{task}_scaffold"
    if not split_dir.exists():
        print(f"  [SKIP] Split not found: {split_dir}")
        return None

    model_dir = output_dir / "A" / f"seed_{seed}" / task / exp["exp_id"]
    if skip_existing and (model_dir / "result.json").exists():
        print(f"  SKIP: already done")
        return json.loads((model_dir / "result.json").read_text())

    print(f"  GNN-FT {pretrain_id} ({model_type}), seed={seed}, {task}")

    config = GNNConfig(
        hidden_dim=exp["hidden_dim"],
        num_layers=exp["num_layers"],
        heads=exp["heads"],
        lr=1e-4,
        weight_decay=1e-5,
        epochs=200,
        patience=25,
    )

    # train_gnn saves to output_dir / f"seed_{seed}" / task / model_type
    # We need it at output_dir / f"seed_{seed}" / task / exp["exp_id"]
    # So: call train_gnn with real model_type, then copy result.json
    tmp_output_dir = output_dir / "A" / f"seed_{seed}" / task / f"_tmp_{exp['exp_id']}"
    tmp_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = B3DBGNNDataset(split_dir=split_dir, task=task)
        result = train_gnn(
            model_type=model_type,
            dataset=dataset,
            seed=seed,
            task=task,
            config=config,
            output_dir=tmp_output_dir,
            device=device,
            num_workers=0,
            verbose=False,
            pretrained_encoder_path=str(backbone_path),
        )

        result_dict = result.to_dict()
        result_dict["exp_id"] = exp["exp_id"]
        result_dict["group"] = "A"
        result_dict["pretrain_id"] = pretrain_id
        result_dict["strategy"] = exp["strategy"]
        result_dict["feature_type"] = None

        # Move result to correct location
        model_dir = output_dir / "A" / f"seed_{seed}" / task / exp["exp_id"]
        model_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        # Copy model.pt and result.json
        for fname in ["model.pt", "result.json"]:
            src = tmp_output_dir / fname
            if src.exists():
                shutil.copy2(src, model_dir / fname)
        # Write updated result.json with enriched fields
        with open(model_dir / "result.json", "w") as f:
            json.dump(result_dict, f, indent=2)
        # Cleanup tmp
        shutil.rmtree(tmp_output_dir)

        return result_dict
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# Embedding Extraction (Groups B and C)
# ============================================================

def ensure_embeddings_extracted(
    pretrain_id: str,
    seed: int,
    task: str,
    embedding_dir: Path,
    device: str,
) -> bool:
    """Ensure embeddings are extracted for a given experiment. Returns True if available."""
    cfg = get_pretrain_config(pretrain_id)
    out_subdir = embedding_dir / pretrain_id / f"seed_{seed}" / task
    check_file = out_subdir / "X_train.npy"

    if check_file.exists():
        return True

    # Need to extract
    if cfg["model_type"] in ("gin", "gat"):
        from scripts.finetune.extract_embeddings import extract_gnn_embeddings
        return extract_gnn_embeddings(
            pretrain_id=pretrain_id,
            model_type=cfg["model_type"],
            seed=seed,
            task=task,
            output_dir=str(embedding_dir),
            device=device,
            skip_existing=True,
        )
    else:
        from scripts.finetune.extract_embeddings import extract_transformer_embeddings
        return extract_transformer_embeddings(
            pretrain_id=pretrain_id,
            seed=seed,
            task=task,
            output_dir=str(embedding_dir),
            device=device,
            skip_existing=True,
        )


def run_group_b(
    exp: dict, seed: int, output_dir: Path, embedding_dir: Path, device: str, skip_existing: bool
) -> dict | None:
    """Train LightGBM on pretrained embeddings (Group B)."""
    pretrain_id = exp["pretrain_id"]
    task = exp["task"]

    # Ensure embeddings are extracted first
    ok = ensure_embeddings_extracted(pretrain_id, seed, task, embedding_dir, device)
    if not ok:
        print(f"  [SKIP] Embeddings not available for {pretrain_id}")
        return None

    model_dir = output_dir / "B" / f"seed_{seed}" / task / pretrain_id
    if skip_existing and (model_dir / "result.json").exists():
        print(f"  SKIP: already done")
        return json.loads((model_dir / "result.json").read_text())

    return train_lgbm_on_embeddings(
        pretrain_id=pretrain_id,
        seed=seed,
        task=task,
        feature_type=None,  # Group B: embedding only
        output_dir=output_dir / "B",
        embedding_dir=embedding_dir,
        verbose=True,
    )


def run_group_c(
    exp: dict, seed: int, output_dir: Path, embedding_dir: Path, skip_existing: bool
) -> dict | None:
    """Train LightGBM on embedding + classical features (Group C)."""
    pretrain_id = exp["pretrain_id"]
    task = exp["task"]
    feature_type = exp["feature_type"]

    # Ensure embeddings are extracted
    ok = ensure_embeddings_extracted(pretrain_id, seed, task, embedding_dir, "auto")
    if not ok:
        print(f"  [SKIP] Embeddings not available for {pretrain_id}")
        return None

    # Ensure classical features exist
    feat_dir = project_root / "artifacts" / "features" / f"seed_{seed}" / "scaffold" / task / feature_type
    if not (feat_dir / "X_train.npy").exists():
        print(f"  [SKIP] Features not found: {feat_dir}")
        return None

    model_dir = output_dir / "C" / f"seed_{seed}" / task / f"{pretrain_id}+{feature_type}"
    if skip_existing and (model_dir / "result.json").exists():
        print(f"  SKIP: already done")
        return json.loads((model_dir / "result.json").read_text())

    return train_lgbm_on_embeddings(
        pretrain_id=pretrain_id,
        seed=seed,
        task=task,
        feature_type=feature_type,  # Group C: embedding + feature
        output_dir=output_dir / "C",
        embedding_dir=embedding_dir,
        verbose=True,
    )


# ============================================================
# Group D: Transformer Fine-tuning
# ============================================================

def run_group_d(
    exp: dict, seed: int, output_dir: Path, device: str, skip_existing: bool
) -> dict | None:
    """Fine-tune a pretrained Transformer on BBB (Group D)."""
    from scripts.finetune.finetune_transformer import finetune_transformer

    pretrain_id = exp["pretrain_id"]
    task = exp["task"]

    print(f"  TRANS-FT {pretrain_id}, seed={seed}, {task}")

    return finetune_transformer(
        pretrain_id=pretrain_id,
        seed=seed,
        task=task,
        output_dir=str(output_dir),
        device=device,
        epochs=200,
        patience=25,
        lr=5e-4,
        batch_size=128,
        verbose=True,
        skip_existing=skip_existing,
    )


# ============================================================
# Extraction Only Mode
# ============================================================

def extract_all_embeddings(matrix: list[dict], seeds: list[int], embedding_dir: Path, device: str):
    """Extract all embeddings needed for the matrix (Groups B and C)."""
    print("\n" + "=" * 60)
    print("EXTRACTING EMBEDDINGS")
    print("=" * 60)

    # Deduplicate: we need embeddings for each (pretrain_id, task) per seed
    needed = {}
    for exp in matrix:
        if exp["group"] in ("B", "C"):
            pid = exp["pretrain_id"]
            for task in ["classification", "regression"]:
                key = (pid, task)
                if key not in needed:
                    needed[key] = (exp["model_type"], task)

    print(f"Need embeddings for {len(needed)} pretrain_id × task combinations")

    total = len(needed) * len(seeds)
    done = 0

    for (pretrain_id, task), (model_type, _) in sorted(needed.items()):
        for seed in seeds:
            done += 1
            print(f"\n[{done}/{total}] {pretrain_id} ({model_type}), seed={seed}, {task}")

            ok = ensure_embeddings_extracted(pretrain_id, seed, task, embedding_dir, device)
            if not ok:
                print(f"  FAILED to extract embeddings")

    print(f"\nEmbedding extraction complete.")


# ============================================================
# Aggregation
# ============================================================

def aggregate_results(output_dir: Path, phases: list[int]) -> None:
    """Aggregate all result.json files into a master CSV."""
    import pandas as pd

    rows = []

    for group_dir in (output_dir).iterdir():
        if not group_dir.is_dir():
            continue
        group = group_dir.name
        if group not in ("A", "B", "C", "D"):
            continue

        for seed_dir in group_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            seed = int(seed_dir.name.replace("seed_", ""))

            for task_dir in seed_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                task = task_dir.name

                for exp_dir in task_dir.iterdir():
                    if not exp_dir.is_dir():
                        continue
                    result_file = exp_dir / "result.json"
                    if not result_file.exists():
                        continue

                    try:
                        r = json.loads(result_file.read_text())
                        r["group"] = group
                        r["seed"] = seed
                        r["task"] = task
                        r["exp_dir"] = str(exp_dir)
                        rows.append(r)
                    except Exception:
                        continue

    if not rows:
        print("No results found to aggregate.")
        return

    df = pd.DataFrame(rows)

    # Save master
    master_path = project_root / "artifacts" / "reports" / "finetune_results_master.csv"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(master_path, index=False)
    print(f"Master results: {master_path} ({len(df)} rows)")

    # Summary by experiment (mean ± std across seeds)
    if "test_auc" in df.columns:
        summary_cols = ["group", "pretrain_id", "model_type", "strategy", "task", "feature_type"]
        available = [c for c in summary_cols if c in df.columns]
        agg = df.groupby(available).agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_std=("test_auc", "std"),
            test_f1_mean=("test_f1", "mean"),
            test_f1_std=("test_f1", "std"),
            n_seeds=("seed", "count"),
        ).reset_index()
        agg = agg.sort_values("test_auc_mean", ascending=False)
        summary_path = project_root / "artifacts" / "reports" / "finetune_cls_summary.csv"
        agg.to_csv(summary_path, index=False)
        print(f"Classification summary: {summary_path}")

    if "test_r2" in df.columns:
        summary_cols = ["group", "pretrain_id", "model_type", "strategy", "task", "feature_type"]
        available = [c for c in summary_cols if c in df.columns]
        agg = df.groupby(available).agg(
            test_r2_mean=("test_r2", "mean"),
            test_r2_std=("test_r2", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            n_seeds=("seed", "count"),
        ).reset_index()
        agg = agg.sort_values("test_r2_mean", ascending=False)
        summary_path = project_root / "artifacts" / "reports" / "finetune_reg_summary.csv"
        agg.to_csv(summary_path, index=False)
        print(f"Regression summary: {summary_path}")

    # Print top results
    print("\n--- Classification Top 10 ---")
    if "test_auc_mean" in dir():
        print(agg.head(10).to_string(index=False))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning experiment matrix")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--dry_run", action="store_true", help="Show commands without executing")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="Run Phase 1 (Groups A+B+D) or Phase 2 (Group C)")
    parser.add_argument("--group", type=str, choices=["A", "B", "C", "D"], default=None,
                        help="Run specific group only")
    parser.add_argument("--run", type=int, default=None,
                        help="Run specific experiment by index (1-based)")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4",
                        help="Comma-separated seeds (default: 0,1,2,3,4)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip experiments with existing result.json (default: True)")
    parser.add_argument("--no_skip", action="store_true",
                        help="Force re-run even if result.json exists")
    parser.add_argument("--extract_only", action="store_true",
                        help="Only extract embeddings, don't train")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda, cpu, auto (default: auto)")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="Only aggregate existing results")
    parser.add_argument("--output_dir", type=str, default="artifacts/models/finetune")
    parser.add_argument("--embedding_dir", type=str, default="artifacts/embeddings")

    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    skip = not args.no_skip
    output_dir = Path(args.output_dir)
    embedding_dir = Path(args.embedding_dir)

    # --- Build matrix ---
    phase = args.phase or 1
    groups = [args.group] if args.group else None
    matrix = build_experiment_matrix(phase=phase, groups=groups)

    # --- Filter by --run ---
    if args.run:
        matrix = [matrix[args.run - 1]] if args.run <= len(matrix) else []

    # --- List or dry run ---
    if args.list:
        print(f"\n{'#':>3} {'Group':>6} {'PretrainID':<25} {'Model':>12} {'Task':<14} {'Feature':<20} Phase")
        print("-" * 95)
        for i, exp in enumerate(matrix):
            feat = exp["feature_type"] or ""
            print(f"{i+1:>3} {exp['group']:>6} {exp['pretrain_id']:<25} {exp['model_type']:>12} "
                  f"{exp['task']:<14} {feat:<20} {exp['phase']}")
        print(f"\nTotal: {len(matrix)} experiments")
        counts = count_experiments(phase=phase, groups=groups)
        print(f"Per group: {counts}")
        print(f"Total × {len(seeds)} seeds = {len(matrix) * len(seeds)} runs")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] {len(matrix)} experiments × {len(seeds)} seeds\n")
        for i, exp in enumerate(matrix):
            feat = exp.get("feature_type", "") or ""
            for seed in seeds:
                print(f"  [{i+1}/{len(matrix)}] seed={seed} {exp['group']} {exp['pretrain_id']} "
                      f"({exp['model_type']}) {exp['task']} {feat}")
        return

    # --- Aggregate only ---
    if args.aggregate_only:
        print("Aggregating results...")
        aggregate_results(output_dir, [1, 2])
        return

    # --- Extract only ---
    if args.extract_only:
        extract_all_embeddings(matrix, seeds, embedding_dir, args.device)
        return

    # --- Execute ---
    total = len(matrix) * len(seeds)
    current = 0
    results = []

    print(f"\n{'=' * 70}")
    print(f"FINE-TUNING MATRIX (Phase {phase})")
    print(f"{'=' * 70}")
    print(f"Experiments: {len(matrix)}, Seeds: {seeds}")
    print(f"Total runs: {total}")
    print(f"Output: {output_dir}")
    print(f"Embeddings: {embedding_dir}")
    print(f"Skip existing: {skip}")
    print(f"{'=' * 70}")

    start_time = time.time()

    for i, exp in enumerate(matrix):
        for seed in seeds:
            current += 1
            group = exp["group"]
            pretrain = exp["pretrain_id"]
            task = exp["task"]

            print(f"\n[{current}/{total}] {group} {pretrain} ({exp['model_type']}) {task} seed={seed}")

            result = None
            try:
                if group == "A":
                    result = run_group_a(exp, seed, output_dir, args.device, skip)
                elif group == "B":
                    result = run_group_b(exp, seed, output_dir, embedding_dir, args.device, skip)
                elif group == "C":
                    result = run_group_c(exp, seed, output_dir, embedding_dir, skip)
                elif group == "D":
                    result = run_group_d(exp, seed, output_dir, args.device, skip)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

            if result:
                results.append(result)

    elapsed = time.time() - start_time

    # --- Aggregate ---
    print(f"\n{'=' * 70}")
    print(f"FINE-TUNING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total runs: {total}")
    print(f"Successful: {len(results)}")
    print(f"Elapsed: {elapsed/60:.1f} min")

    print("\nAggregating results...")
    aggregate_results(output_dir, [phase])

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
