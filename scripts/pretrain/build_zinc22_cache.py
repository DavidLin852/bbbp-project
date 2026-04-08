#!/usr/bin/env python
"""
Build ZINC22 full-text cache (parallel extraction + pickle consolidation).

Two phases:
  Phase 1 — Parallel extraction: each .smi.gz is read and valid SMILES are
            written to a plain text file (one per input file). No compression
            during extraction = maximum I/O parallelism.
  Phase 2 — Consolidation: collect all extracted files and write a single
            compressed pickle for fast loading in pretraining.

Usage:
    # Phase 1: extract all SMILES to plain text (~20-40 min on fast disk)
    python scripts/pretrain/build_zinc22_cache.py \\
        --data_dir data/zinc22 \\
        --output_dir data/zinc22/extracted

    # Phase 2: consolidate into pickle (~5-10 min for 100 GB of text)
    python scripts/pretrain/build_zinc22_cache.py \\
        --data_dir data/zinc22 \\
        --output data/zinc22/full_cache.pkl.gz \\
        --input_dir data/zinc22/extracted

    # Or run full pipeline in one command
    python scripts/pretrain/build_zinc22_cache.py \\
        --data_dir data/zinc22 \\
        --output data/zinc22/full_cache.pkl.gz
"""

from __future__ import annotations
import argparse
import gzip
import multiprocessing as mp
import random
from pathlib import Path
from tqdm import tqdm
import time
import pickle
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Build ZINC22 full cache")
    parser.add_argument("--data_dir", type=str, default="data/zinc22")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/zinc22/extracted",
        help="Directory for Phase 1 extracted plain-text files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/zinc22/full_cache.pkl.gz",
        help="Output file for Phase 2 consolidated pickle",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Use existing extracted files (skip Phase 1)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count)",
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


def _extract_single_file(args) -> tuple[Path, int]:
    """Read one .smi.gz, write valid SMILES to plain text file. Returns (out_path, count)."""
    smi_file, out_dir = args
    out_file = out_dir / (smi_file.name + ".txt")

    try:
        count = 0
        with gzip.open(smi_file, "rt", encoding="utf-8") as f_in:
            with open(out_file, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    parts = line.strip().split("\t")
                    if len(parts) >= 1:
                        smiles = parts[0].strip()
                        if _is_valid_smiles_fast(smiles):
                            f_out.write(smiles + "\n")
                            count += 1
        return out_file, count
    except Exception as e:
        return out_file, 0


def _load_extracted_file(txt_file: Path) -> list[str]:
    """Load one extracted plain-text file."""
    try:
        with open(txt_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        return []


def phase1_extract(smi_files: list[Path], out_dir: Path, num_workers: int) -> int:
    """Phase 1: parallel extraction of all SMILES to plain text files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    total_count = 0
    args_list = [(f, out_dir) for f in smi_files]

    with mp.Pool(num_workers) as pool:
        for out_file, count in tqdm(
            pool.imap_unordered(_extract_single_file, args_list, chunksize=20),
            total=len(smi_files),
            desc="Extracting",
        ):
            total_count += count

    print(f"\nPhase 1 done: {total_count:,} valid SMILES written to {out_dir}/")
    return total_count


def phase2_consolidate(input_dir: Path, output_path: Path, num_workers: int) -> int:
    """Phase 2: load all extracted files and write a single pickle.gz."""
    txt_files = sorted(input_dir.rglob("*.txt"))
    print(f"Phase 2: loading {len(txt_files):,} extracted files...")

    all_smiles = []
    with mp.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_load_extracted_file, txt_files, chunksize=50),
                total=len(txt_files),
                desc="Loading",
            )
        )
    for batch in results:
        all_smiles.extend(batch)

    total_count = len(all_smiles)
    print(f"\nLoaded {total_count:,} SMILES. Writing to {output_path}...")

    t_write = time.time()
    with gzip.open(output_path, "wb", compresslevel=3) as f:
        pickle.dump(all_smiles, f)

    elapsed_write = time.time() - t_write
    size_gb = output_path.stat().st_size / 1e9
    print(f"Wrote pickle in {elapsed_write:.1f}s ({size_gb:.1f} GB)")

    return total_count


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_path = Path(args.output)

    smi_files = sorted(data_dir.rglob("*.smi.gz"))
    num_workers = min(args.num_workers or mp.cpu_count(), len(smi_files))

    print(f"Found {len(smi_files)} .smi.gz files")
    print(f"Workers: {num_workers}")
    print("=" * 60)

    t0 = time.time()

    if args.input_dir:
        # Phase 2 only
        total_count = phase2_consolidate(Path(args.input_dir), output_path, num_workers)
    else:
        # Phase 1 + Phase 2
        total_count = phase1_extract(smi_files, output_dir, num_workers)
        if output_path:
            total_count = phase2_consolidate(output_dir, output_path, num_workers)

    elapsed_total = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Total valid molecules: {total_count:,}")
    print(f"Total time: {elapsed_total/60:.1f} min")

    if output_path.exists():
        # Test load speed
        print("\nTesting load speed...")
        t_load = time.time()
        with gzip.open(output_path, "rb") as f:
            loaded = pickle.load(f)
        load_time = time.time() - t_load
        print(f"Pickle load: {load_time:.1f}s for {len(loaded):,} molecules")
        print(f"\nCache ready at: {output_path}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
