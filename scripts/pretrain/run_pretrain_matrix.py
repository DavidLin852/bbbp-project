#!/usr/bin/env python
"""
Run a matrix of pretraining experiments.

Two core strategies for molecular graphs:
- Property Prediction: learn structure-property relationships (SAR)
- Denoising: learn chemical constraints and valid structures

Design space:
- Samples:   100K / 500K / 1M / 2M / 5M
- Epochs:    10 / 20
- Model:     gin / gat
- Hidden dim: 128 / 256

Usage:
    # List all experiments
    python scripts/pretrain/run_pretrain_matrix.py --list

    # Dry run (show commands without executing)
    python scripts/pretrain/run_pretrain_matrix.py --dry_run

    # Run specific experiment by index
    python scripts/pretrain/run_pretrain_matrix.py --run 1

    # Run all property experiments
    python scripts/pretrain/run_pretrain_matrix.py --strategy property

    # Run all
    python scripts/pretrain/run_pretrain_matrix.py
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
# Experiment Matrix
# ============================================================
# Two strategies, focused on molecular graph learning:
#
# Strategy A - Property Prediction:
#   Input: clean molecular graph
#   Target: 11 chemical properties (logP, TPSA, MW, HBD, HBA, ...)
#   Learns: structure -> physicochemical property mapping (SAR)
#
# Strategy B - Denoising:
#   Input: graph with Gaussian noise on node features
#   Target: reconstruct original clean features
#   Learns: what feature combinations are chemically valid

PRIORITY_EXPERIMENTS = [
    # ============================================================
    # Strategy A: Property Prediction (构效关系)
    # ============================================================

    # A1: Sample size ablation (GIN, 128d, 10 epochs)
    {
        "id": "P_E10_GIN_100K",
        "strategy": "property",
        "samples": 100_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Property 100K: small scale baseline",
    },
    {
        "id": "P_E10_GIN_500K",
        "strategy": "property",
        "samples": 500_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Property 500K: medium scale",
    },
    {
        "id": "P_E10_GIN_1M",
        "strategy": "property",
        "samples": 1_000_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Property 1M: standard scale",
    },
    {
        "id": "P_E10_GIN_2M",
        "strategy": "property",
        "samples": 2_000_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Property 2M: large scale",
    },
    {
        "id": "P_E10_GIN_5M",
        "strategy": "property",
        "samples": 5_000_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Property 5M: max scale",
    },

    # A2: Architecture comparison (1M, 10 epochs)
    {
        "id": "P_E10_GAT_1M",
        "strategy": "property",
        "samples": 1_000_000,
        "epochs": 10,
        "model": "gat",
        "hidden_dim": 128,
        "notes": "Property GAT: GIN vs GAT comparison",
    },

    # A3: Flagship (5M, 20 epochs, 256d)
    {
        "id": "P_E20_GIN_256_5M",
        "strategy": "property",
        "samples": 5_000_000,
        "epochs": 20,
        "model": "gin",
        "hidden_dim": 256,
        "notes": "Property flagship: full scale (5M, 20ep, 256d)",
    },

    # ============================================================
    # Strategy B: Denoising (化学约束)
    # ============================================================

    # B1: Sample size ablation (GIN, 128d, 10 epochs)
    {
        "id": "D_E10_GIN_100K",
        "strategy": "denoising",
        "samples": 100_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Denoising 100K: small scale baseline",
    },
    {
        "id": "D_E10_GIN_1M",
        "strategy": "denoising",
        "samples": 1_000_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Denoising 1M: standard scale",
    },
    {
        "id": "D_E10_GIN_5M",
        "strategy": "denoising",
        "samples": 5_000_000,
        "epochs": 10,
        "model": "gin",
        "hidden_dim": 128,
        "notes": "Denoising 5M: max scale",
    },

    # B2: Architecture comparison (1M, 10 epochs)
    {
        "id": "D_E10_GAT_1M",
        "strategy": "denoising",
        "samples": 1_000_000,
        "epochs": 10,
        "model": "gat",
        "hidden_dim": 128,
        "notes": "Denoising GAT: GIN vs GAT comparison",
    },

    # B3: Flagship (5M, 20 epochs, 256d)
    {
        "id": "D_E20_GIN_256_5M",
        "strategy": "denoising",
        "samples": 5_000_000,
        "epochs": 20,
        "model": "gin",
        "hidden_dim": 256,
        "notes": "Denoising flagship: full scale (5M, 20ep, 256d)",
    },

    # ============================================================
    # Strategy C: Transformer MLM (SMILES masked language modeling)
    # ============================================================

    # C1: Sample size ablation (256d, 4 layers, 8 heads, 10 epochs)
    {
        "id": "T_E10_TRANS_100K",
        "strategy": "transformer",
        "samples": 100_000,
        "epochs": 10,
        "model": "transformer",
        "hidden_dim": 256,
        "notes": "Transformer MLM 100K: small scale baseline",
    },
    {
        "id": "T_E10_TRANS_1M",
        "strategy": "transformer",
        "samples": 1_000_000,
        "epochs": 10,
        "model": "transformer",
        "hidden_dim": 256,
        "notes": "Transformer MLM 1M: standard scale",
    },
    {
        "id": "T_E10_TRANS_5M",
        "strategy": "transformer",
        "samples": 5_000_000,
        "epochs": 10,
        "model": "transformer",
        "hidden_dim": 256,
        "notes": "Transformer MLM 5M: max scale",
    },

    # C2: Depth comparison (1M, 10 epochs)
    {
        "id": "T_E10_TRANS_1M_L6",
        "strategy": "transformer",
        "samples": 1_000_000,
        "epochs": 10,
        "model": "transformer",
        "hidden_dim": 256,
        "n_layers": 6,
        "notes": "Transformer MLM 6 layers: depth comparison",
    },

    # C3: Flagship (5M, 20 epochs, 512d, 8 layers)
    {
        "id": "T_E20_TRANS_512_5M",
        "strategy": "transformer",
        "samples": 5_000_000,
        "epochs": 20,
        "model": "transformer",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 12,
        "notes": "Transformer flagship: full scale (5M, 20ep, 512d, 8L)",
    },
]

# Strategy -> script mapping
STRATEGY_SCRIPTS = {
    "property": "scripts/pretrain/pretrain_graph.py",
    "denoising": "scripts/pretrain/pretrain_denoising.py",
    "transformer": "scripts/pretrain/pretrain_smiles.py",
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
        "--save_dir", f"{SAVE_DIR_BASE}/{exp['id']}",
    ]

    if exp["strategy"] == "transformer":
        # Transformer uses d_model, n_layers, n_heads, batch_size
        cmd += [
            "--d_model", str(exp.get("hidden_dim", 256)),
            "--n_layers", str(exp.get("n_layers", 4)),
            "--n_heads", str(exp.get("n_heads", 8)),
            "--batch_size", "512",
        ]
    else:
        # GNN strategies use model, hidden_dim, batch_size
        cmd += [
            "--model", exp["model"],
            "--hidden_dim", str(exp["hidden_dim"]),
            "--batch_size", "256",
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
    parser.add_argument("--save_results", type=str, default=None,
                        help="Path to save results JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print(f"\n{'#':>2} {'ID':<25} {'Strategy':<12} {'Samples':>8} {'Epochs':>6} {'Model':<12} {'Dim':>4}  Notes")
        print("-" * 110)
        for i, exp in enumerate(PRIORITY_EXPERIMENTS):
            if exp["strategy"] == "transformer":
                n_layers = exp.get("n_layers", 4)
                n_heads = exp.get("n_heads", 8)
                model_str = f"trans-L{n_layers}H{n_heads}"
            else:
                model_str = exp["model"].upper()
            print(f"{i+1:>2} {exp['id']:<25} {exp['strategy']:<12} {exp['samples']:>8,} {exp['epochs']:>6} "
                  f"{model_str:<12} {exp.get('hidden_dim', 256):>4}  {exp['notes']}")
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
        print(f"  {r['exp_id']:<25} {r['status']}")
    print(f"\nTotal: {len(results)} experiments")

    # Save results
    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump({"experiments": PRIORITY_EXPERIMENTS, "results": results}, f, indent=2)
        print(f"Saved results to {args.save_results}")

    # Failed count
    failed = [r for r in results if "FAILED" in r["status"]]
    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  {r['exp_id']}: {r['status']}")


if __name__ == "__main__":
    main()
