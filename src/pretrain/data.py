"""
ZINC22 Data Pipeline for Pretraining

Handles ZINC22 data in the standard ZINC22 directory structure:
- Organized by subdirectories (H04, H05, H06, etc.)
- Each subdirectory contains .smi.gz files (gzipped SMILES)
- Each line: SMILES<TAB>ZINC_ID

Supports:
- Direct reading from .smi.gz files (no manual extraction needed)
- Incremental sampling across multiple files
- Caching for efficiency
- Both graph and SMILES representations

Usage:
    dataset = ZINC22Dataset(
        data_dir="data/zinc22",
        num_samples=100000,
        representation="graph"
    )
"""

from __future__ import annotations
import gzip
import pickle
from pathlib import Path
from typing import Literal, Optional, List
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from features.graph import smiles_to_pyg_graph
from transformer.smiles_tokenizer import SMILESTokenizer, create_tokenizer_from_data


class ZINC22Dataset(Dataset):
    """
    ZINC22 dataset that reads directly from .smi.gz files.

    Handles the standard ZINC22 directory structure:
    data/zinc22/
    ├── H04/
    │   ├── H04M000.smi.gz
    │   ├── H04M100.smi.gz
    │   └── ...
    ├── H05/
    │   └── ...
    └── H06/
        └── ...
    """

    def __init__(
        self,
        data_dir: str | Path,
        representation: Literal["graph", "smiles"] = "graph",
        num_samples: int = 100000,
        cache_dir: str | Path = "data/zinc22/cache",
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize ZINC22 dataset.

        Args:
            data_dir: Path to ZINC22 directory (e.g., data/zinc22)
            representation: "graph" or "smiles"
            num_samples: Maximum number of samples to use
            cache_dir: Directory for cached processed data
            shuffle: Whether to shuffle samples
            seed: Random seed for shuffling
        """
        self.data_dir = Path(data_dir)
        self.representation = representation
        self.num_samples = num_samples
        self.cache_dir = Path(cache_dir)
        self.shuffle = shuffle
        self.seed = seed

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Find all .smi.gz files
        self.smi_files = self._find_smi_files()
        print(f"Found {len(self.smi_files)} .smi.gz files in {self.data_dir}")

        # Build index of all SMILES
        self.smiles_list = self._build_index()
        print(f"Loaded {len(self.smiles_list)} SMILES for {representation} pretraining")

    def _find_smi_files(self) -> List[Path]:
        """Find all .smi.gz files in the directory tree."""
        return sorted(self.data_dir.rglob("*.smi.gz"))

    def _build_index(self) -> List[str]:
        """
        Build index of SMILES strings.

        Priority:
        1. Full cache (all molecules from ZINC22) → fastest for subsequent runs
        2. Subsample cache (smiles_index_{N}_{seed}.pkl) → fast
        3. Fallback: read capped lines per file → slow (one-time)
        """
        import os

        # Priority 1: Full cache
        full_cache = os.environ.get("ZINC22_CACHE") or str(
            self.data_dir / "full_cache.parquet"
        )
        full_cache_path = Path(full_cache)
        if full_cache_path.exists():
            print(f"Loading full cache from {full_cache_path}...")
            all_smiles = _load_full_cache(full_cache_path)
            print(f"Loaded {len(all_smiles):,} molecules from full cache")
        else:
            # Priority 2: Subsample cache
            cache_file = self.cache_dir / f"smiles_index_{self.num_samples}_{self.seed}.pkl"
            if cache_file.exists():
                print(f"Loading cached index from {cache_file}")
                with open(cache_file, "rb") as f:
                    all_smiles = pickle.load(f)
            else:
                # Priority 3: Fallback - read capped lines per file
                k = self.num_samples
                num_files = len(self.smi_files)
                lines_per_file = max(3000, int(k * 1.6 / num_files))
                print(f"Reading {lines_per_file:,} lines per file from {num_files} files "
                      f"(target: {k:,} molecules)...")

                all_smiles = []
                for smi_file in tqdm(self.smi_files, desc="Reading files"):
                    try:
                        smiles = _read_file_fast(smi_file, max_lines=lines_per_file)
                        all_smiles.extend(smiles)
                    except Exception as e:
                        print(f"Warning: Error reading {smi_file}: {e}")

                print(f"Loaded {len(all_smiles):,} molecules")

                # Cache
                with open(cache_file, "wb") as f:
                    pickle.dump(all_smiles, f)

        # Shuffle and limit to num_samples
        rng = np.random.RandomState(self.seed)
        rng.shuffle(all_smiles)
        all_smiles = all_smiles[: self.num_samples]

        print(f"Selected {len(all_smiles):,} molecules for pretraining")
        return all_smiles

    @staticmethod
    def _is_valid_smiles(smiles: str) -> bool:
        """Check if SMILES is valid using RDKit."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None and mol.GetNumAtoms() > 0
        except:
            return False

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int):
        smiles = self.smiles_list[idx]

        if self.representation == "graph":
            # Return PyG Data object
            return smiles_to_pyg_graph(smiles)

        elif self.representation == "smiles":
            # Return SMILES string (tokenization in collate_fn)
            return smiles

        else:
            raise ValueError(f"Unknown representation: {self.representation}")


def _read_file_fast(smi_file: Path, max_lines: int) -> List[str]:
    """
    Read the first max_lines from a gzipped SMILES file.

    Much faster than reading all lines when we only need a subset.

    Args:
        smi_file: Path to .smi.gz file
        max_lines: Maximum number of lines to read

    Returns:
        List of valid SMILES strings
    """
    smiles_list = []
    try:
        with gzip.open(smi_file, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                parts = line.strip().split("\t")
                if len(parts) >= 1:
                    smiles = parts[0].strip()
                    if _is_valid_smiles_fast(smiles):
                        smiles_list.append(smiles)
    except Exception:
        pass
    return smiles_list


def _load_full_cache(cache_path: Path) -> List[str]:
    """Load full ZINC22 cache from parquet or pickle."""
    if cache_path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(cache_path)
        return table["smiles"].to_pylist()
    elif cache_path.suffix == ".gz":
        import pickle

        with gzip.open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        # Try pickle
        with open(cache_path, "rb") as f:
            return pickle.load(f)


def _is_valid_smiles_fast(smiles: str) -> bool:
    """
    Fast basic SMILES validation without RDKit.

    Only checks:
    - Length is reasonable (1-200 chars)
    - Contains valid SMILES characters
    - Not empty or whitespace only

    This is ~100x faster than RDKit validation.
    """
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


def create_zinc22_dataloader(
    data_dir: str | Path,
    representation: Literal["graph", "smiles"],
    batch_size: int = 32,
    num_samples: int = 100000,
    num_workers: int = 0,
    tokenizer: Optional[SMILESTokenizer] = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader for ZINC22 pretraining.

    Args:
        data_dir: Path to ZINC22 directory (e.g., data/zinc22)
        representation: "graph" or "smiles"
        batch_size: Batch size
        num_samples: Number of samples to load
        num_workers: Number of worker processes
        tokenizer: SMILES tokenizer (required for smiles representation)
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    dataset = ZINC22Dataset(
        data_dir=data_dir,
        representation=representation,
        num_samples=num_samples,
        shuffle=shuffle,
    )

    if representation == "graph":
        # No special collate function needed for PyG Data objects
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    elif representation == "smiles":
        if tokenizer is None:
            raise ValueError("Tokenizer required for SMILES representation")

        # Use custom collate function for SMILES
        from transformer.trainer import collate_fn

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        )

    else:
        raise ValueError(f"Unknown representation: {representation}")


def count_zinc22_molecules(data_dir: str | Path) -> int:
    """
    Count total number of molecules in ZINC22 directory using fast validation.

    Args:
        data_dir: Path to ZINC22 directory

    Returns:
        Total count of valid SMILES (fast estimate)
    """
    data_dir = Path(data_dir)
    smi_files = sorted(data_dir.rglob("*.smi.gz"))

    total = 0
    for smi_file in tqdm(smi_files, desc="Counting molecules"):
        try:
            with gzip.open(smi_file, "rt") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 1:
                        smiles = parts[0].strip()
                        if _is_valid_smiles_fast(smiles):
                            total += 1
        except:
            continue

    return total


def estimate_total_molecules(data_dir: str | Path, sample_files: int = 10) -> int:
    """
    Estimate total molecules by sampling a few files and extrapolating.

    Much faster than counting all files (~2 min vs hours).

    Args:
        data_dir: Path to ZINC22 directory
        sample_files: Number of files to sample for estimation

    Returns:
        Estimated total molecule count
    """
    import random
    data_dir = Path(data_dir)
    smi_files = sorted(data_dir.rglob("*.smi.gz"))

    if not smi_files:
        return 0

    # Sample files evenly across the directory
    step = max(1, len(smi_files) // sample_files)
    sampled_files = [smi_files[i] for i in range(0, len(smi_files), step)]

    total_lines = 0
    for smi_file in tqdm(sampled_files, desc="Sampling files"):
        try:
            with gzip.open(smi_file, "rt") as f:
                total_lines += sum(1 for _ in f)
        except:
            continue

    avg_per_file = total_lines / len(sampled_files)
    estimated_total = int(avg_per_file * len(smi_files))
    print(f"Sampled {len(sampled_files)}/{len(smi_files)} files. "
          f"Avg {avg_per_file:.0f} lines/file. "
          f"Estimated total: {estimated_total:,}")

    return estimated_total


# ==================== Backward Compatibility ====================

class ZINC22PretrainDataset(ZINC22Dataset):
    """
    Alias for backward compatibility.
    Deprecated: Use ZINC22Dataset instead.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import warnings
        warnings.warn(
            "ZINC22PretrainDataset is deprecated. Use ZINC22Dataset instead.",
            DeprecationWarning,
        )


# Keep the old create_small_zinc22_sample for compatibility
def create_small_zinc22_sample(
    output_path: str | Path = "data/zinc22/smiles_small.txt",
    num_samples: int = 10000,
    source_dir: str | Path = "data/zinc22",
) -> Path:
    """
    Create a small sample of ZINC22 for smoke testing.

    Extracts num_samples from the ZINC22 directory and saves to a single file.

    Args:
        output_path: Path for output file
        num_samples: Number of samples to extract
        source_dir: Source ZINC22 directory

    Returns:
        Path to output file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Sample file already exists: {output_path}")
        return output_path

    print(f"Creating small ZINC22 sample: {num_samples} molecules")

    # Load samples
    dataset = ZINC22Dataset(
        data_dir=source_dir,
        representation="smiles",
        num_samples=num_samples,
    )

    # Write to file
    with open(output_path, "w") as f:
        for smiles in tqdm(dataset.smiles_list, desc="Writing"):
            f.write(smiles + "\n")

    print(f"Created {output_path} with {len(dataset.smiles_list)} molecules")
    return output_path
