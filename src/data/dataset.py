"""
Dataset classes for B3DB data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


class B3DBDataset:
    """
    B3DB dataset container.

    Provides convenient access to train/val/test splits
    and supports both classification and regression tasks.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        task: Literal["classification", "regression"] = "classification",
        smiles_col: str = "SMILES_canon",
        label_col: str = "y_cls",
    ):
        self.train = train_df
        self.val = val_df
        self.test = test_df
        self.task = task
        self.smiles_col = smiles_col
        self.label_col = label_col

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)

    def get_splits(self):
        """Return train/val/test dataframes."""
        return self.train, self.val, self.test

    def get_split_sizes(self):
        """Return sizes of each split."""
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }

    def get_label_distribution(self):
        """Get label distribution for each split."""
        dist = {}
        for name, df in [("train", self.train), ("val", self.val), ("test", self.test)]:
            if self.label_col in df.columns:
                dist[name] = {
                    "positive": int(df[self.label_col].sum()),
                    "negative": int(len(df) - df[self.label_col].sum()),
                    "rate": float(df[self.label_col].mean()),
                }
        return dist

    def save_splits(self, output_dir: Path):
        """Save splits to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.train.to_csv(output_dir / "train.csv", index=False)
        self.val.to_csv(output_dir / "val.csv", index=False)
        self.test.to_csv(output_dir / "test.csv", index=False)

    @classmethod
    def load_splits(cls, input_dir: Path, **kwargs):
        """Load splits from CSV files."""
        input_dir = Path(input_dir)

        train_df = pd.read_csv(input_dir / "train.csv")
        val_df = pd.read_csv(input_dir / "val.csv")
        test_df = pd.read_csv(input_dir / "test.csv")

        return cls(train_df, val_df, test_df, **kwargs)
