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
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Build ZINC22 full cache")
    parser.add_argument("--data_dir", type=str, default="data/zinc22")
    parser.add_argument("--output", type=str, default="data/zinc22/full_cache.parquet")
    parser.add_argument("--num_workers", type=int, default=32)
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


def _read_single_file(smi_file: Path) -> list:
    """Read all valid SMILES from a single file."""
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

    print(f"Found {len(smi_files)} .smi.gz files in {data_dir}")
    print(f"Will read ALL lines from all files with {args.num_workers} workers")
    print(f"Output: {args.output}")
    print("=" * 60)
    print("WARNING: This will take 4-8 hours depending on disk speed.")
    print("But it's a ONE-TIME cost. All subsequent pretraining will")
    print("load directly from the cache (~seconds).")
    print("=" * 60)

    t0 = time.time()
    all_smiles = []
    num_workers = min(args.num_workers, len(smi_files), mp.cpu_count())

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_read_single_file, f): f for f in smi_files}
        done = 0
        with tqdm(total=len(futures), desc="Reading files") as pbar:
            for future in as_completed(futures):
                smiles_list = future.result()
                all_smiles.extend(smiles_list)
                done += 1
                pbar.update(1)
                pbar.set_postfix({"total": f"{len(all_smiles):,}"})

    elapsed = time.time() - t0
    print(f"\nRead {len(all_smiles):,} valid molecules in {elapsed/3600:.1f} hours")
    print(f"Writing to {args.output}...")

    # Write as compressed parquet (requires pyarrow) or fallback to gzip pickle
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        arr = pa.array(all_smiles)
        table = pa.table({"smiles": arr})
        pq.write_table(table, args.output, compression="zstd")
        size_mb = Path(args.output).stat().st_size / 1e6
        print(f"Saved {len(all_smiles):,} molecules ({size_mb:.0f} MB) to {args.output}")

    except ImportError:
        import pickle
        import gzip as gz

        # Fallback: gzip compressed pickle
        out_path = Path(args.output).with_suffix(".pkl.gz")
        with gz.open(out_path, "wb") as f:
            pickle.dump(all_smiles, f)
        size_mb = out_path.stat().st_size / 1e6
        print(f"Saved {len(all_smiles):,} molecules ({size_mb:.0f} MB) to {out_path}")
        print("Note: .pkl.gz is slower to load than parquet. Run:")
        print(f"  pip install pyarrow")

    print(f"Total time: {elapsed/3600:.1f} hours")
    print("\nTo use this cache, set:")
    print(f"  export ZINC22_CACHE={args.output}")


if __name__ == "__main__":
    main()
