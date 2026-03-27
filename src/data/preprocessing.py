"""
B3DB data preprocessing module.

Handles loading, cleaning, and preprocessing of B3DB classification
and regression datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass
class ProcessedData:
    """Container for processed B3DB data."""
    df: pd.DataFrame
    smiles_col: str = "SMILES"
    label_col: str = "y_cls"
    logbb_col: str = "logBB"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, key):
        return self.df[key]


class B3DBPreprocessor:
    """
    Preprocessor for B3DB datasets.

    Supports:
    - Classification (BBB+/BBB-)
    - Regression (logBB)
    - Canonicalization and sanitization
    - Duplicate removal
    - Invalid molecule filtering
    """

    def __init__(
        self,
        smiles_col: str = "SMILES",
        bbb_col: str = "BBB+/BBB-",
        logbb_col: str = "logBB",
        group_col: str = "group",
        id_cols: list[str] | None = None,
    ):
        self.smiles_col = smiles_col
        self.bbb_col = bbb_col
        self.logbb_col = logbb_col
        self.group_col = group_col
        self.id_cols = id_cols or ["NO.", "CID", "compound_name"]

    def load_classification(
        self,
        filepath: Path | str,
        groups: list[str] | tuple[str, ...] = ("A", "B"),
        deduplicate: bool = True,
        canonicalize: bool = True,
    ) -> ProcessedData:
        """
        Load and preprocess B3DB classification dataset.

        Args:
            filepath: Path to B3DB_classification.tsv
            groups: Which groups to keep (default: A, B)
            deduplicate: Remove duplicate SMILES
            canonicalize: Canonicalize SMILES

        Returns:
            ProcessedData container
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        # Load TSV
        df = pd.read_csv(filepath, sep="\t")

        # Filter groups
        if groups and self.group_col in df.columns:
            df = df[df[self.group_col].astype(str).isin(groups)].copy()

        # Keep only rows with BBB labels
        df = df[df[self.bbb_col].notna()].copy()

        # Create binary label
        df["y_cls"] = (df[self.bbb_col].astype(str).str.strip() == "BBB+").astype(int)

        # Process SMILES
        df = self._process_smiles(df, canonicalize=canonicalize)

        # Remove invalid molecules
        df = df[df["mol_valid"].eq(True)].copy()

        # Deduplicate
        if deduplicate:
            df = self._deduplicate(df)

        # Add row_id if missing
        if "row_id" not in df.columns:
            df["row_id"] = range(len(df))

        return ProcessedData(df=df)

    def load_regression(
        self,
        filepath: Path | str,
        groups: list[str] | tuple[str, ...] = ("A", "B"),
        deduplicate: bool = True,
        canonicalize: bool = True,
    ) -> ProcessedData:
        """
        Load and preprocess B3DB regression dataset.

        Args:
            filepath: Path to B3DB_regression.tsv
            groups: Which groups to keep
            deduplicate: Remove duplicate SMILES
            canonicalize: Canonicalize SMILES

        Returns:
            ProcessedData container
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        # Load TSV
        df = pd.read_csv(filepath, sep="\t")

        # Filter groups
        if groups and self.group_col in df.columns:
            df = df[df[self.group_col].astype(str).isin(groups)].copy()

        # Keep only rows with logBB values
        df = df[df[self.logbb_col].notna()].copy()

        # Process SMILES
        df = self._process_smiles(df, canonicalize=canonicalize)

        # Remove invalid molecules
        df = df[df["mol_valid"].eq(True)].copy()

        # Deduplicate
        if deduplicate:
            df = self._deduplicate(df)

        return ProcessedData(df=df)

    def _process_smiles(
        self,
        df: pd.DataFrame,
        canonicalize: bool = True,
    ) -> pd.DataFrame:
        """Process SMILES: validate, canonicalize, compute scaffolds."""
        smiles_list = df[self.smiles_col].astype(str).str.strip().tolist()

        canonical_smiles = []
        mol_valid = []
        scaffolds = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                canonical_smiles.append(smi)
                mol_valid.append(False)
                scaffolds.append(None)
                continue

            mol_valid.append(True)

            if canonicalize:
                smi_canon = Chem.MolToSmiles(mol, canonical=True)
                canonical_smiles.append(smi_canon)
            else:
                canonical_smiles.append(smi)

            # Compute Murcko scaffold
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smi = Chem.MolToSmiles(scaffold, canonical=True)
                scaffolds.append(scaffold_smi)
            except:
                scaffolds.append(None)

        df["SMILES_canon"] = canonical_smiles
        df["mol_valid"] = mol_valid
        df["scaffold"] = scaffolds

        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate SMILES, keeping first occurrence."""
        # Use canonical SMILES for deduplication
        df = df.drop_duplicates(subset=["SMILES_canon"], keep="first")
        return df.reset_index(drop=True)

    def get_statistics(self, data: ProcessedData) -> dict:
        """Get dataset statistics."""
        df = data.df

        stats = {
            "n_samples": len(df),
            "n_invalid": df["mol_valid"].eq(False).sum() if "mol_valid" in df.columns else 0,
        }

        if "y_cls" in df.columns:
            stats["n_bbb_positive"] = int(df["y_cls"].sum())
            stats["bbb_positive_rate"] = float(df["y_cls"].mean())

        if self.logbb_col in df.columns:
            stats["logbb_mean"] = float(df[self.logbb_col].mean())
            stats["logbb_std"] = float(df[self.logbb_col].std())

        if "scaffold" in df.columns:
            stats["n_unique_scaffolds"] = df["scaffold"].nunique()

        if self.group_col in df.columns:
            stats["group_counts"] = df[self.group_col].value_counts().to_dict()

        return stats
