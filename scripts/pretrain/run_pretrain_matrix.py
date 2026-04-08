#!/usr/bin/env python
"""
Run a matrix of pretraining experiments.

Design space:
- Samples:   100K / 500K / 1M / 2M / 5M
- Epochs:    5 / 10 / 20
- Model:     gin / gat
- Strategy:  attr_masking / edge_masking / property / contrastive / denoising / context

Not all combinations are run — see PRIORITY_EXPERIMENTS below.

Usage:
    # Run all priority experiments (recommended: run on cluster)
    python scripts/pretrain/run_pretrain_matrix.py

    # Run specific experiment by ID
    python scripts/pretrain/run_pretrain_matrix.py --run 1

    # List all experiments
    python scripts/pretrain/run_pretrain_matrix.py --list

    # Dry run (show what would be run)
    python scripts/pretrain/run_pretrain_matrix.py --dry_run
"""

from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


# ============================================================
# Priority experiments: well-designed ablation matrix
# ============================================================
#
# Format: {
#     "id": "EXP_ID",
#     "strategy": "attr_masking" | "property" | "contrastive" | ...,
#     "samples": int,
#     "epochs": int,
#     "model": "gin" | "gat",
#     "hidden_dim": 128 | 256,
#     "notes": str,
# }
#
# Strategy -> script mapping:
#   attr_masking   -> scripts/pretrain/pretrain_attr_masking.py
#   edge_masking   -> scripts/pretrain/pretrain_edge_masking.py
#   property       -> scripts/pretrain/pretrain_graph.py
#   contrastive    -> scripts/pretrain/pretrain_contrastive.py
#   denoising      -> scripts/pretrain/pretrain_denoising.py
#   context        -> scripts/pretrain/pretrain_context.py

PRIORITY_EXPERIMENTS = [
    # === Baseline: Property Prediction (existing) ===
    {
        "id": "S4_E10_GIN_100K",
        "strategy": "property",
        "samples": 100_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Property prediction baseline: 100K samples",
    },
    {
        "id": "S4_E10_GIN_1M",
        "strategy": "property",
        "samples": 1_000_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Property prediction: 1M samples",
    },
    {
        "id": "S4_E10_GIN_5M",
        "strategy": "property",
        "samples": 5_000_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Property prediction: 5M samples (large-scale)",
    },

    # === Epochs ablation on attr_masking ===
    {
        "id": "S1_E5_GIN_100K",
        "strategy": "attr_masking",
        "samples": 100_000,
        "epochs": 5,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Attr masking: 5 epochs, 100K — fast sanity check",
    },
    {
        "id": "S1_E10_GIN_100K",
        "strategy": "attr_masking",
        "samples": 100_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Attr masking: 10 epochs, 100K",
    },
    {
        "id": "S1_E20_GIN_100K",
        "strategy": "attr_masking",
        "samples": 100_000,
        "epochs": 20,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Attr masking: 20 epochs, 100K",
    },
    {
        "id": "S1_E10_GIN_1M",
        "strategy": "attr_masking",
        "samples": 1_000_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Attr masking: 10 epochs, 1M",
    },
    {
        "id": "S1_E10_GIN_5M",
        "strategy": "attr_masking",
        "samples": 5_000_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Attr masking: 10 epochs, 5M (large-scale)",
    },

    # === Strategy comparison @ 100K, 10 epochs ===
    {
        "id": "S2_E10_GIN_100K",
        "strategy": "edge_masking",
        "samples": 100_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Edge masking: strategy comparison baseline",
    },
    {
        "id": "S3_E10_GIN_100K",
        "strategy": "contrastive",
        "samples": 100_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Contrastive: strategy comparison",
    },
    {
        "id": "S5_E10_GIN_100K",
        "strategy": "context",
        "samples": 100_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Context prediction: strategy comparison",
    },
    {
        "id": "S6_E10_GIN_100K",
        "strategy": "denoising",
        "samples": 100_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Denoising: strategy comparison",
    },

    # === Best strategy @ different scales ===
    {
        "id": "S1_E10_GIN_500K",
        "strategy": "attr_masking",
        "samples": 500_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Attr masking: 500K medium scale",
    },
    {
        "id": "S1_E10_GIN_2M",
        "strategy": "attr_masking",
        "samples": 2_000_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Attr masking: 2M large scale",
    },

    # === GAT vs GIN comparison ===
    {
        "id": "S1_E10_GAT_100K",
        "strategy": "attr_masking",
        "samples": 100_000,
        "epochs": 10,
        "model": "gat",
        "hidden_dim": 128,
        "notes": "Attr masking GAT: architecture comparison",
    },
    {
        "id": "S1_E10_GAT_1M",
        "strategy": "attr_masking",
        "samples": 1_000_000,
        "epochs": 10,
        "model": "gat",
        "hidden_dim": 128,
        "notes": "Attr masking GAT: 1M scale",
    },

    # === Hidden dim comparison ===
    {
        "id": "S1_E10_GIN_256_100K",
        "strategy": "attr_masking",
        "samples": 100_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 256,
        "notes": "Attr masking: larger hidden dim (256)",
    },
    {
        "id": "S1_E10_GIN_256_1M",
        "strategy": "attr_masking",
        "samples": 1_000_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 256,
        "notes": "Attr masking: larger hidden dim (256), 1M",
    },

    # === Large-scale: best strategy, best model ===
    {
        "id": "S1_E20_GIN_256_5M",
        "strategy": "attr_masking",
        "samples": 5_000_000,
        "epochs": 20,
        "model": "gin",
        "hidden_dim": 256,
        "notes": "Attr masking: full scale (5M, 20 epochs, 256d) — flagship",
    },
    {
        "id": "S4_E20_GIN_256_5M",
        "strategy": "property",
        "samples": 5_000_000,
        "epochs": 20,
        "model": "gin",
        "hidden_dim": 256,
        "notes": "Property prediction: full scale — flagship",
    },
]

# Strategy -> script mapping
STRATEGY_SCRIPTS = {
    "attr_masking": "scripts/pretrain/pretrain_attr_masking.py",
    "edge_masking": "scripts/pretrain/pretrain_edge_masking.py",
    "property": "scripts/pretrain/pretrain_graph.py",
    "contrastive": "scripts/pretrain/pretrain_contrastive.py",
    "denoising": "scripts/pretrain/pretrain_denoising.py",
    "context": "scripts/pretrain/pretrain_context.py",
}

# Save directory base
SAVE_DIR_BASE = "artifacts/models/pretrain/exp_matrix"


def build_command(exp: dict) -> list[str]:
    """Build the command line for an experiment."""
    script = STRATEGY_SCRIPTS[exp["strategy"]]
    cmd = [
        "python", str(PROJECT_ROOT / script),
        "--num_samples", str(exp["samples"]),
        "--epochs", str(exp["epochs"]),
        "--model", exp["model"],
        "--hidden_dim", str(exp["hidden_dim"]),
        "--save_dir", f"{SAVE_DIR_BASE}/{exp['id']}",
    ]
    return cmd


def run_experiment(exp: dict, dry_run: bool = False) -> dict:
    """Run a single experiment."""
    cmd = build_command(exp)
    save_path = Path(f"{SAVE_DIR_BASE}/{exp['id']}")

    # Check if already done
    backbone_path = save_path / f"{exp['model']}_pretrained_backbone.pt"
    if backbone_path.exists():
        return {"exp_id": exp["id"], "status": "SKIPPED (already exists)", "cmd": " ".join(cmd)}

    if dry_run:
        return {"exp_id": exp["id"], "status": "DRY_RUN", "cmd": " ".join(cmd)}

    print(f"\n{'='*60}")
    print(f"Running: {exp['id']}")
    print(f"Strategy: {exp['strategy']}, Samples: {exp['samples']:,}, Epochs: {exp['epochs']}")
    print(f"Model: {exp['model']}, hidden_dim: {exp['hidden_dim']}")
    print(f"Notes: {exp['notes']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        return {"exp_id": exp["id"], "status": "SUCCESS", "cmd": " ".join(cmd)}
    except subprocess.CalledProcessError as e:
        return {"exp_id": exp["id"], "status": f"FAILED ({e.returncode})", "cmd": " ".join(cmd)}


def parse_args():
    parser = argparse.ArgumentParser(description="Run pretraining experiment matrix")
    parser.add_argument("--run", type=int, default=None,
                        help="Run specific experiment by index (1-based)")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=list(STRATEGY_SCRIPTS.keys()),
                        help="Run only experiments of this strategy")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be run without executing")
    parser.add_argument("--continue_on_error", action="store_true", default=True,
                        help="Continue if an experiment fails")
    parser.add_argument("--save_results", type=str, default=None,
                        help="Path to save results JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print(f"\n{'ID':<30} {'Strategy':<15} {'Samples':>8} {'Epochs':>6} {'Model':<5} {'Dim':>4}  Notes")
        print("-" * 100)
        for i, exp in enumerate(PRIORITY_EXPERIMENTS):
            print(f"{exp['id']:<30} {exp['strategy']:<15} {exp['samples']:>8,} {exp['epochs']:>6} "
                  f"{exp['model']:<5} {exp['hidden_dim']:>4}  {exp['notes']}")
        print(f"\nTotal: {len(PRIORITY_EXPERIMENTS)} experiments")
        return

    # Filter experiments
    exps = PRIORITY_EXPERIMENTS
    if args.run:
        exps = [PRIORITY_EXPERIMENTS[args.run - 1]]
    elif args.strategy:
        exps = [e for e in exps if e["strategy"] == args.strategy]

    print(f"\n{'='*60}")
    print(f"Pretraining Experiment Matrix")
    print(f"{'='*60}")
    print(f"Experiments: {len(exps)}")
    print(f"Strategy filter: {args.strategy or 'all'}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*60}")

    results = []
    for i, exp in enumerate(exps):
        print(f"\n[{i+1}/{len(exps)}] ", end="", flush=True)
        result = run_experiment(exp, dry_run=args.dry_run)
        results.append(result)
        print(f"  {result['status']}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['exp_id']:<30} {r['status']}")
    print(f"\nTotal: {len(results)} experiments")

    # Save results
    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump({"experiments": PRIORITY_EXPERIMENTS, "results": results}, f, indent=2)
        print(f"Saved results to {args.save_results}")

    # Failed count
    failed = [r for r in results if "FAILED" in r["status"]]
    skipped = [r for r in results if "SKIPPED" in r["status"]]
    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  {r['exp_id']}: {r['status']}")


if __name__ == "__main__":
    main()
