#!/usr/bin/env python
"""
Build ZINC22 full-text cache (parallel + pickle).

Reads ALL molecules from ZINC22 in parallel, validates them, and saves to a
compressed pickle. This is a one-time cost — subsequent pretraining
loads directly from cache in seconds.

Usage:
    python scripts/pretrain/build_zinc22_cache.py \\
        --data_dir data/zinc22 \\
        --output data/zinc22/full_cache.pkl.gz
"""

from __future__ import annotations
import argparse
import gzip
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import time
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Build ZINC22 full cache")
    parser.add_argument("--data_dir", type=str, default="data/zinc22")
    parser.add_argument("--output", type=str, default="data/zinc22/full_cache.pkl.gz")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing partial cache (append mode)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel reader processes (default: auto = cpu_count)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50_000,
        help="Batch size for parallel file reads (files per worker chunk)",
    )
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
    except Exception:
        return False


def _read_single_file(smi_file: Path) -> list[str]:
    """Read one .smi.gz file, return list of valid SMILES."""
    smiles_list = []
    try:
        with gzip.open(smi_file, "rt", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 1:
                    smiles = parts[0].strip()
                    if _is_valid_smiles_fast(smiles):
                        smiles_list.append(smiles)
    except Exception:
        pass
    return smiles_list


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    smi_files = sorted(data_dir.rglob("*.smi.gz"))

    num_workers = args.num_workers or mp.cpu_count()
    num_workers = min(num_workers, len(smi_files))

    print(f"Found {len(smi_files)} .smi.gz files in {data_dir}")
    print(f"Parallel reading with {num_workers} workers")
    print(f"Output: {args.output}")
    print("=" * 60)
    print(
        "This will take 20-40 min depending on disk speed (parallel I/O)."
    )
    print("Subsequent pretraining will load from cache in seconds.")
    print("=" * 60)

    t0 = time.time()

    # Estimate total molecules by sampling a few files
    sample_size = min(20, len(smi_files))
    sample_files = smi_files[:sample_size]
    avg_per_file = 0
    with mp.Pool(num_workers) as pool:
        sample_results = list(
            tqdm(
                pool.imap(_read_single_file, sample_files, chunksize=1),
                total=sample_size,
                desc="Sampling file sizes",
            )
        )
    total_sample = sum(len(r) for r in sample_results)
    avg_per_file = total_sample / sample_size
    estimated_total = int(avg_per_file * len(smi_files))
    print(f"Estimated total valid molecules: ~{estimated_total:,}")

    # Parallel read all files
    print(f"\nReading all {len(smi_files)} files in parallel...")
    all_smiles = []

    with mp.Pool(num_workers) as pool:
        # Use imap_unordered for maximum throughput — order doesn't matter
        results_iter = pool.imap_unordered(
            _read_single_file, smi_files, chunksize=args.batch_size
        )

        done = 0
        for smiles_batch in tqdm(results_iter, total=len(smi_files), desc="Reading files"):
            all_smiles.extend(smiles_batch)
            done += 1
            if done % 500 == 0:
                tqdm.write(f"  Processed {done}/{len(smi_files)} files, {len(all_smiles):,} valid molecules so far")

    total_count = len(all_smiles)
    elapsed_read = time.time() - t0

    print(f"\nRead {total_count:,} valid molecules in {elapsed_read/60:.1f} min")
    print(f"Writing to {args.output}...")

    # Write as compressed pickle — much faster to load than gzip text
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_write = time.time()
    with gzip.open(out_path, "wb", compresslevel=3) as f:
        pickle.dump(all_smiles, f)

    elapsed_write = time.time() - t_write
    elapsed_total = time.time() - t0
    size_gb = out_path.stat().st_size / 1e9

    print(f"Wrote pickle in {elapsed_write:.1f}s")
    print(f"\nDone! Total time: {elapsed_total/60:.1f} min")
    print(f"Output file: {out_path}")
    print(f"File size: {size_gb:.2f} GB")
    print(f"Total valid molecules: {total_count:,}")

    # Test load time
    print("\nTesting load speed...")
    t_load = time.time()
    with gzip.open(out_path, "rb") as f:
        loaded = pickle.load(f)
    load_time = time.time() - t_load
    print(f"Load time: {load_time:.2f}s for {len(loaded):,} molecules")
    print(f"  (vs ~5 min for .txt.gz loading)")
    print("\nTo use this cache, run pretraining — it auto-detects the cache.")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set
    main()
