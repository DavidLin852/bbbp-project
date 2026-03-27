"""
Dataset classes for VAE training and molecule generation.

Provides PyTorch datasets for:
- General molecule datasets (ZINC, etc.)
- BBB-specific datasets (B3DB)
- Pre-training and fine-tuning splits
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import pickle

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import AllChem

from ..featurize.graph_pyg import smiles_to_graph, BBBGraphDataset, GraphBuildConfig
from ..config import Paths, DatasetConfig


class MoleculeDataset(InMemoryDataset):
    """
    General molecule dataset for VAE pre-training.

    Can be used with any SMILES dataset (ZINC, PubChem, etc.)
    for learning general molecular representations.
    """

    def __init__(
        self,
        root: str,
        smiles_list: List[str],
        transform=None,
        pre_transform=None,
        max_atoms: int = 50,
    ):
        self.smiles_list = smiles_list
        self.max_atoms = max_atoms
        super().__init__(root, transform, pre_transform)

        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    @property
    def raw_file_names(self) -> List[str]:
        return []

    def download(self):
        return

    def process(self):
        data_list: List[Data] = []

        for i, smi in enumerate(self.smiles_list):
            # Convert to graph
            g = smiles_to_graph(smi)
            if g is None:
                continue

            x, edge_index, edge_attr = g

            # Skip molecules that are too large
            if x.size(0) > self.max_atoms:
                continue

            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=x.size(0)
            )
            data.smiles = smi
            data.idx = i

            data_list.append(data)

        # Save
        data, slices = self.collate(data_list)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        smiles_col: str = "SMILES",
        root: Optional[str] = None,
        max_samples: Optional[int] = None,
        max_atoms: int = 50,
    ) -> "MoleculeDataset":
        """
        Create dataset from CSV file.

        Args:
            csv_path: Path to CSV file
            smiles_col: Column name for SMILES
            root: Root directory for processed data
            max_samples: Maximum number of samples to load
            max_atoms: Maximum number of atoms per molecule

        Returns:
            MoleculeDataset instance
        """
        df = pd.read_csv(csv_path)

        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found in CSV")

        smiles_list = df[smiles_col].dropna().tolist()

        if max_samples is not None:
            smiles_list = smiles_list[:max_samples]

        if root is None:
            root = str(csv_path.parent / f"processed_{csv_path.stem}")

        return cls(root=root, smiles_list=smiles_list, max_atoms=max_atoms)


class BBBDataset(BBBGraphDataset):
    """
    BBB-specific dataset for VAE fine-tuning.

    Extends BBBGraphDataset to filter for BBB+ molecules
    and add additional molecular properties.
    """

    def __init__(
        self,
        root: str,
        df: pd.DataFrame,
        cfg: GraphBuildConfig = GraphBuildConfig(),
        bbb_only: bool = True,  # Only use BBB+ molecules for fine-tuning
        min_qed: float = 0.0,
        max_sa: float = 10.0,
        transform=None,
        pre_transform=None,
    ):
        self.bbb_only = bbb_only
        self.min_qed = min_qed
        self.max_sa = max_sa

        # Filter dataset
        df_filtered = self._filter_dataframe(df, cfg)

        # Initialize parent
        super().__init__(root, df_filtered, cfg, transform, pre_transform)

    def _filter_dataframe(self, df: pd.DataFrame, cfg: GraphBuildConfig) -> pd.DataFrame:
        """Filter dataframe based on BBB status and properties."""
        df_filtered = df.copy()

        # Filter for BBB+ only if requested
        if self.bbb_only and cfg.label_col in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[cfg.label_col] == 1].reset_index(drop=True)

        return df_filtered

    @classmethod
    def from_split(
        cls,
        split_path: Path,
        root: Optional[str] = None,
        bbb_only: bool = True,
        cfg: GraphBuildConfig = GraphBuildConfig(),
    ) -> "BBBDataset":
        """
        Create dataset from train/val/test split file.

        Args:
            split_path: Path to split CSV file
            root: Root directory for processed data
            bbb_only: Only use BBB+ molecules
            cfg: Graph build configuration

        Returns:
            BBBDataset instance
        """
        df = pd.read_csv(split_path)

        if root is None:
            root = str(split_path.parent / "pyg_graphs_vae")

        return cls(root=root, df=df, cfg=cfg, bbb_only=bbb_only)

    @classmethod
    def from_b3db(
        cls,
        b3db_path: Path,
        group_filter: str = "A,B",
        root: Optional[str] = None,
        bbb_only: bool = True,
    ) -> "BBBDataset":
        """
        Create dataset directly from B3DB file.

        Args:
            b3db_path: Path to B3DB TSV file
            group_filter: Comma-separated groups to include (e.g., "A,B")
            root: Root directory for processed data
            bbb_only: Only use BBB+ molecules

        Returns:
            BBBDataset instance
        """
        from ..config import DatasetConfig

        dataset_cfg = DatasetConfig()
        groups = [g.strip() for g in group_filter.split(",")]

        df = pd.read_csv(b3db_path, sep="\t")

        # Filter by group
        if dataset_cfg.group_col in df.columns:
            df = df[df[dataset_cfg.group_col].isin(groups)].reset_index(drop=True)

        # Convert BBB+/- to 1/0
        if dataset_cfg.bbb_col in df.columns:
            df["y_cls"] = df[dataset_cfg.bbb_col].map({"BBB+": 1, "BBB-": 0})

        # Add row ID
        df["row_id"] = range(len(df))

        if root is None:
            root = str(b3db_path.parent / "pyg_graphs_vae_b3db")

        cfg = GraphBuildConfig(
            smiles_col=dataset_cfg.smiles_col,
            label_col="y_cls",
            id_col="row_id"
        )

        return cls(root=root, df=df, cfg=cfg, bbb_only=bbb_only)


class SMILESDataset(torch.utils.data.Dataset):
    """
    Simple dataset for SMILES strings.

    Useful for quick iteration without graph processing.
    """

    def __init__(
        self,
        smiles_list: List[str],
        labels: Optional[List[float]] = None,
    ):
        self.smiles_list = smiles_list
        self.labels = labels if labels is not None else [0.0] * len(smiles_list)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return {
            'smiles': self.smiles_list[idx],
            'label': self.labels[idx],
        }

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        smiles_col: str = "SMILES",
        label_col: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> "SMILESDataset":
        """Create dataset from CSV file."""
        df = pd.read_csv(csv_path)

        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found")

        smiles_list = df[smiles_col].dropna().tolist()

        if max_samples is not None:
            smiles_list = smiles_list[:max_samples]

        labels = None
        if label_col and label_col in df.columns:
            labels = df[label_col].tolist()[:len(smiles_list)]

        return cls(smiles_list=smiles_list, labels=labels)


def create_train_val_splits(
    smiles_list: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[List[str], List[str], List[str]]:
    """
    Split SMILES list into train/val/test sets.

    Args:
        smiles_list: List of SMILES strings
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed

    Returns:
        train_smiles, val_smiles, test_smiles
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(smiles_list))

    n_train = int(len(smiles_list) * train_ratio)
    n_val = int(len(smiles_list) * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_smiles = [smiles_list[i] for i in train_indices]
    val_smiles = [smiles_list[i] for i in val_indices]
    test_smiles = [smiles_list[i] for i in test_indices]

    return train_smiles, val_smiles, test_smiles
