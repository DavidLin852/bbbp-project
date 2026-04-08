#!/usr/bin/env python
"""
Build ZINC22 full-text cache.

Reads ALL molecules from ZINC22 once, validates them, and saves to a
compact binary format. This is a one-time cost — subsequent pretraining
loads directly from cache.

Usage:
    python scripts/pretrain/build_zinc22_cache.py \
        --data_dir data/zinc22 \
        --output data/zinc22/full_cache.parquet \
        --num_workers 32

Output: ~2-5 GB compressed parquet (vs 10+ TB raw text)
Estimated time: 4-8 hours depending on disk speed.
"""

from __future__ import annotations
import argparse
import gzip
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Build ZINC22 full cache")
    parser.add_argument("--data_dir", type=str, default="data/zinc22")
    parser.add_argument("--output", type=str, default="data/zinc22/full_cache.txt.gz")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel reader processes (default: 1, sequential)")
    return parser.parse_args()


def _is_valid_smiles_fast(smiles: str) -> bool:
    if not smiles or len(smiles.strip()) == 0:
        return False
    if len(smiles) < 1 or len(smiles) > 200:
        return False
    try:
        for char in smiles:
            if char.isalpha() or char.isdigit() or char in "()=#+-@.[].* ":
                continue
            if char in "\\|%":
                continue
        return True
    except:
        return False


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    smi_files = sorted(data_dir.rglob("*.smi.gz"))

    print(f"Found {len(smi_files)} .smi.gz files in {data_dir}")
    print(f"Will read ALL lines from all files with {args.num_workers} workers")
    print(f"Output: {args.output}")
    print("=" * 60)
    print("WARNING: This will take 4-8 hours depending on disk speed.")
    print("But it's a ONE-TIME cost. All subsequent pretraining will")
    print("load directly from the cache (~seconds).")
    print("=" * 60)

    t0 = time.time()
    num_workers = min(args.num_workers, len(smi_files), mp.cpu_count())

    # Stream write to gzip: no memory accumulation
    # Each molecule ~80 bytes avg, 268B molecules ≈ 215 GB raw → ~30-50 GB compressed
    out_path = Path(args.output)
    total_count = 0

    print(f"Streaming write to {out_path} (no memory accumulation)...")
    print(f"Estimated size: 30-50 GB compressed")
    print()

    with gz.open(out_path, "wt", encoding="utf-8") as fout:
        with tqdm(total=len(smi_files), desc="Reading files") as pbar:
            for smi_file in smi_files:
                try:
                    with gzip.open(smi_file, "rt", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split("\t")
                            if len(parts) >= 1:
                                smiles = parts[0].strip()
                                if _is_valid_smiles_fast(smiles):
                                    fout.write(smiles + "\n")
                                    total_count += 1
                except Exception as e:
                    print(f"\nWarning: Error reading {smi_file}: {e}")
                pbar.update(1)
                if total_count % 10_000_000 == 0 and total_count > 0:
                    pbar.set_postfix({"written": f"{total_count:,}"})

    elapsed = time.time() - t0
    size_gb = out_path.stat().st_size / 1e9
    print(f"\nWrote {total_count:,} valid molecules in {elapsed/3600:.1f} hours")
    print(f"File size: {size_gb:.1f} GB ({args.output})")
    print(f"\nTotal time: {elapsed/3600:.1f} hours")
    print("\nTo use this cache, set:")
    print(f"  export ZINC22_CACHE={args.output}")
    print("Or just run pretraining — it auto-detects the cache file.")


if __name__ == "__main__":
    main()
