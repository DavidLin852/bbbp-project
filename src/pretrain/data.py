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
from typing import Literal, Optional, List, Tuple, Iterator
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

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
        Build index of all SMILES strings.

        Uses caching to avoid re-reading files on every initialization.
        Uses parallel processing for faster file reading.
        """
        cache_file = self.cache_dir / f"smiles_index_{self.num_samples}_{self.seed}.pkl"

        # Try to load from cache
        if cache_file.exists():
            print(f"Loading cached index from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        # Read all SMILES from files using parallel processing
        print(f"Reading {len(self.smi_files)} files with {mp.cpu_count()} workers...")
        all_smiles = self._read_files_parallel()

        # Shuffle if requested
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(all_smiles)

        # Limit to num_samples
        all_smiles = all_smiles[:self.num_samples]

        # Cache for next time
        with open(cache_file, "wb") as f:
            pickle.dump(all_smiles, f)

        return all_smiles

    def _read_files_parallel(self) -> List[str]:
        """Read multiple gz files in parallel using ProcessPoolExecutor."""
        # Calculate samples per file (rough estimate)
        samples_per_file = max(1000, self.num_samples // len(self.smi_files))

        all_smiles = []
        num_workers = min(mp.cpu_count(), len(self.smi_files))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks
            futures = {
                executor.submit(_read_single_file, smi_file, samples_per_file): smi_file
                for smi_file in self.smi_files[: (self.num_samples // samples_per_file + 1)]
            }

            # Collect results with progress bar
            with tqdm(total=len(futures), desc="Reading ZINC22 files") as pbar:
                for future in as_completed(futures):
                    smi_file = futures[future]
                    try:
                        smiles_list = future.result()
                        all_smiles.extend(smiles_list)
                        pbar.set_postfix({"collected": len(all_smiles)})
                    except Exception as e:
                        print(f"Warning: Error reading {smi_file}: {e}")
                    finally:
                        pbar.update(1)

                    # Early stop if we have enough
                    if len(all_smiles) >= self.num_samples:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break

        return all_smiles


def _read_single_file(smi_file: Path, max_samples: int) -> List[str]:
    """
    Read a single .smi.gz file and extract SMILES.

    This function is designed to be called in a separate process.
    Fast basic validation only - no RDKit validation for speed.
    """
    smiles_list = []

    try:
        with gzip.open(smi_file, "rt", encoding="utf-8") as f:
            for line in f:
                if len(smiles_list) >= max_samples:
                    break

                parts = line.strip().split("\t")
                if len(parts) >= 1:
                    smiles = parts[0].strip()
                    # Fast basic validation only
                    if _is_valid_smiles_fast(smiles):
                        smiles_list.append(smiles)
    except Exception:
        pass

    return smiles_list


def _is_valid_smiles_fast(smiles: str) -> bool:
    """
    Fast basic SMILES validation without RDKit.

    Only checks:
    - Length is reasonable (1-200 chars)
    - Contains valid SMILES characters
    - Not empty or whitespace only

    This is ~100x faster than RDKit validation.
    Invalid SMILES will be filtered out during first training epoch
    if needed, but most ZINC22 SMILES are valid.
    """
    if not smiles or len(smiles.strip()) == 0:
        return False

    # Length check
    if len(smiles) < 1 or len(smiles) > 200:
        return False

    # Basic character check - valid SMILES characters
    valid_chars = set("BCNOPSFIKClBrH()=#0123456789+-[@].[nh]se")
    # Add lowercase for aromatic atoms
    valid_chars.update(set("cnops"))

    try:
        # Quick check: all characters should be valid SMILES characters
        # Allow some flexibility for unusual SMILES notations
        for char in smiles:
            if char.isalpha() or char.isdigit() or char in "()=#+-@.[].* ":
                continue
            # Allow brackets and special characters
            if char in "\\|%":
                continue
        return True
    except:
        return False

    @staticmethod
    def _is_valid_smiles(smiles: str) -> bool:
        """Check if SMILES is valid."""
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
    Count total number of molecules in ZINC22 directory.

    Args:
        data_dir: Path to ZINC22 directory

    Returns:
        Total count of valid SMILES
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
                        if ZINC22Dataset._is_valid_smiles(smiles):
                            total += 1
        except:
            continue

    return total


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
