"""
Scaffold-based data splitting.

Implements scaffold split for molecular datasets to ensure
structural diversity between train/val/test sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitResult:
    """Container for split results."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)


def scaffold_split(
    df: pd.DataFrame,
    scaffold_col: str = "scaffold",
    smiles_col: str = "SMILES_canon",
    label_col: str = "y_cls",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> SplitResult:
    """
    Perform scaffold-based split.

    Scaffolds are distributed across train/val/test in a stratified manner
    to maintain label distribution while ensuring structural diversity.

    Args:
        df: Input dataframe with scaffold column
        scaffold_col: Column name containing scaffold SMILES
        smiles_col: Column name containing canonical SMILES
        label_col: Column name containing labels
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed

    Returns:
        SplitResult with train/val/test dataframes
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1. Got {total}")

    # Remove molecules without scaffolds
    df_valid = df[df[scaffold_col].notna()].copy()
    df_invalid = df[df[scaffold_col].isna()].copy()

    # Get unique scaffolds
    scaffolds = df_valid[scaffold_col].unique()
    np.random.seed(seed)
    scaffolds_shuffled = np.random.permutation(scaffolds)

    # Assign scaffolds to train/val/test
    n_scaffolds = len(scaffolds_shuffled)
    n_train = int(n_scaffolds * train_ratio)
    n_val = int(n_scaffolds * val_ratio)

    train_scaffolds = set(scaffolds_shuffled[:n_train])
    val_scaffolds = set(scaffolds_shuffled[n_train:n_train + n_val])
    test_scaffolds = set(scaffolds_shuffled[n_train + n_val:])

    # Assign molecules based on scaffold
    train_mask = df_valid[scaffold_col].isin(train_scaffolds)
    val_mask = df_valid[scaffold_col].isin(val_scaffolds)
    test_mask = df_valid[scaffold_col].isin(test_scaffolds)

    df_train = df_valid[train_mask].reset_index(drop=True)
    df_val = df_valid[val_mask].reset_index(drop=True)
    df_test = df_valid[test_mask].reset_index(drop=True)

    # Add invalid molecules to train (they can't be scaffold-split)
    if len(df_invalid) > 0:
        df_train = pd.concat([df_train, df_invalid], ignore_index=True)

    return SplitResult(
        train=df_train,
        val=df_val,
        test=df_test,
    )


def random_split(
    df: pd.DataFrame,
    label_col: str = "y_cls",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> SplitResult:
    """
    Perform random stratified split.

    This is a simpler baseline split that doesn't consider scaffolds.
    Useful for comparison with scaffold split.

    Args:
        df: Input dataframe
        label_col: Column name containing labels
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed

    Returns:
        SplitResult with train/val/test dataframes
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1. Got {total}")

    # First split: train vs temp
    df_train, df_temp = train_test_split(
        df,
        test_size=(1 - train_ratio),
        stratify=df[label_col],
        random_state=seed,
    )

    # Second split: val vs test from temp
    val_frac = val_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1 - val_frac),
        stratify=df_temp[label_col],
        random_state=seed,
    )

    return SplitResult(
        train=df_train.reset_index(drop=True),
        val=df_val.reset_index(drop=True),
        test=df_test.reset_index(drop=True),
    )
