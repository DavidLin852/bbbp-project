"""
Graph dataset utilities for GNN training.

Converts B3DB molecular data into PyTorch Geometric Data objects
for use with GNN models.

Graph featurization:
- Node features (atom-level): 22 dimensions
    - Atomic number one-hot (H,C,N,O,F,P,S,Cl,Br,I) + unknown flag  -> 11 dims
    - Degree, H-count, formal charge                                 -> 3 dims
    - Hybridization one-hot (SP/SP2/SP3/SP3D/SP3D2) + unknown flag   -> 6 dims
    - Aromaticity flag                                               -> 1 dim
    - Scaled atomic mass (mass / 100)                                -> 1 dim
    Total: 11 + 3 + 6 + 1 + 1 = 22 dims

- Edge features (bond-level): 7 dimensions
    - Bond type one-hot (SINGLE/DOUBLE/TRIPLE/AROMATIC) + unknown    -> 5 dims
    - Conjugation flag                                               -> 1 dim
    - Ring membership flag                                           -> 1 dim
    Total: 5 + 1 + 1 = 7 dims

- GCN does not use edge features (GCNConv ignores edge_attr)

For fair comparison with classical baselines:
- All splits use the same scaffold split CSVs
- Classification: BBB+/BBB- label (y_cls, binary)
- Regression: logBB label (continuous)
- No data augmentation
- No pretrained molecular representations at this stage
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondType
from torch_geometric.data import Data, Dataset


# ==================== Atom/Bond Feature Definitions ====================

# Ordered list of supported atomic numbers (most common in organic molecules)
ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H,C,N,O,F,P,S,Cl,Br,I
HYB_LIST = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
]
BOND_LIST = [
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
]

# Feature dimensions
NODE_FEATURE_DIM = len(ATOM_LIST) + 1 + 3 + (len(HYB_LIST) + 1) + 1 + 1  # 11+3+6+1+1 = 22
EDGE_FEATURE_DIM = len(BOND_LIST) + 1 + 1 + 1 + 1  # 4+1+1+1+1 = 8 (type+unknown+conj+ring+stereo)


def _one_hot(val, allowed_list):
    return [1.0 if val == v else 0.0 for v in allowed_list]


def atom_features(atom: Chem.Atom) -> list[float]:
    """Compute node features for a single atom."""
    z = atom.GetAtomicNum()
    feats = []

    # Atomic number (one-hot + unknown flag)
    feats += _one_hot(z, ATOM_LIST)
    feats.append(1.0 if z not in ATOM_LIST else 0.0)

    # Degree, H-count, formal charge (3 dims)
    feats.append(float(atom.GetDegree()))
    feats.append(float(atom.GetTotalNumHs()))
    feats.append(float(atom.GetFormalCharge()))

    # Hybridization (one-hot + unknown flag)
    hyb = atom.GetHybridization()
    feats += _one_hot(hyb, HYB_LIST)
    feats.append(1.0 if hyb not in HYB_LIST else 0.0)

    # Aromaticity and mass (2 dims)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(float(atom.GetMass() * 0.01))  # scaled mass

    return feats  # 22 dims


def bond_features(bond: Chem.Bond) -> list[float]:
    """Compute edge features for a single bond."""
    bt = bond.GetBondType()
    feats = []

    # Bond type (one-hot + unknown flag)
    feats += _one_hot(bt, BOND_LIST)
    feats.append(1.0 if bt not in BOND_LIST else 0.0)

    # Conjugation and ring (2 dims)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)

    # Stereo (1 dim)
    feats.append(1.0 if bond.GetStereo() else 0.0)

    return feats  # 8 dims


def smiles_to_data(
    smiles: str,
    label: float,
    task: Literal["classification", "regression"] = "classification",
) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Args:
        smiles: SMILES string
        label: Label value (y_cls for classification, logBB for regression)
        task: Task type

    Returns:
        PyG Data object, or None if the molecule is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Filter very small molecules (no heavy atoms)
    if mol.GetNumHeavyAtoms() == 0:
        return None

    # Node features
    x = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()],
        dtype=torch.float,
    )

    # Edge index and edge features
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        # Undirected edge: add both directions
        edge_index.append([i, j])
        edge_attr.append(bf)
        edge_index.append([j, i])
        edge_attr.append(bf)

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Label
    y = torch.tensor([label], dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )

    return data


# ==================== Dataset Class ====================

class B3DBGNNDataset(Dataset):
    """
    PyG Dataset for B3DB molecules for GNN training.

    Loads preprocessed splits from CSV files and converts
    molecules to graph representations on-the-fly.

    Usage:
        dataset = B3DBGNNDataset(
            split_dir="data/splits/seed_0/classification_scaffold",
            task="classification",
        )
        print(f"Train: {len(dataset.get_split('train'))}")
    """

    def __init__(
        self,
        split_dir: str | Path,
        task: Literal["classification", "regression"] = "classification",
        smiles_col: str = "SMILES_canon",
        label_col: str = "y_cls",
        logbb_col: str = "logBB",
        transform=None,
        pre_transform=None,
    ):
        super().__init__(transform, pre_transform)

        split_dir = Path(split_dir)
        self.split_dir = split_dir
        self.task = task
        self.smiles_col = smiles_col
        self.label_col = label_col if task == "classification" else logbb_col

        self.train_df = pd.read_csv(split_dir / "train.csv")
        self.val_df = pd.read_csv(split_dir / "val.csv")
        self.test_df = pd.read_csv(split_dir / "test.csv")

        # Pre-convert all molecules to Data objects (cached in memory)
        self._train_data: list[Data] = []
        self._val_data: list[Data] = []
        self._test_data: list[Data] = []

        self._build_split(self.train_df, self._train_data)
        self._build_split(self.val_df, self._val_data)
        self._build_split(self.test_df, self._test_data)

    def _build_split(self, df: pd.DataFrame, out_list: list[Data]):
        """Convert a dataframe split to PyG Data objects."""
        for _, row in df.iterrows():
            smiles = str(row[self.smiles_col]).strip()
            label = float(row[self.label_col])
            data = smiles_to_data(smiles, label)
            if data is not None:
                data.smiles = smiles
                out_list.append(data)

    @property
    def train_size(self) -> int:
        return len(self._train_data)

    @property
    def val_size(self) -> int:
        return len(self._val_data)

    @property
    def test_size(self) -> int:
        return len(self._test_data)

    def __len__(self) -> int:
        return len(self._train_data) + len(self._val_data) + len(self._test_data)

    def __getitem__(self, idx: int) -> Data:
        n_train = len(self._train_data)
        n_val = len(self._val_data)

        if idx < n_train:
            return self._train_data[idx]
        elif idx < n_train + n_val:
            return self._val_data[idx - n_train]
        else:
            return self._test_data[idx - n_train - n_val]

    def get_split(self, split: Literal["train", "val", "test"]) -> list[Data]:
        """Get all Data objects for a specific split."""
        if split == "train":
            return self._train_data
        elif split == "val":
            return self._val_data
        else:
            return self._test_data

    def get_input_dim(self) -> int:
        """Return the node feature dimension (22)."""
        return NODE_FEATURE_DIM

    def get_edge_dim(self) -> int:
        """Return the edge feature dimension (8)."""
        return EDGE_FEATURE_DIM


# ==================== Result Containers ====================

@dataclass
class GNNTrainingResult:
    """Container for GNN classification training results."""
    model_name: str
    seed: int
    task: str
    # Train
    train_auc: float
    train_f1: float
    train_loss: float
    # Val
    val_auc: float
    val_f1: float
    val_loss: float
    # Test
    test_auc: float
    test_f1: float
    test_loss: float
    # Training info
    best_epoch: int
    total_epochs: int

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "feature_type": "graph",
            "seed": self.seed,
            "task": self.task,
            "train_auc": self.train_auc,
            "train_f1": self.train_f1,
            "train_loss": self.train_loss,
            "val_auc": self.val_auc,
            "val_f1": self.val_f1,
            "val_loss": self.val_loss,
            "test_auc": self.test_auc,
            "test_f1": self.test_f1,
            "test_loss": self.test_loss,
            "best_epoch": self.best_epoch,
            "total_epochs": self.total_epochs,
        }


@dataclass
class GNNRegressionResult:
    """Container for GNN regression training results."""
    model_name: str
    seed: int
    task: str
    # Train
    train_r2: float
    train_rmse: float
    train_mae: float
    train_loss: float
    # Val
    val_r2: float
    val_rmse: float
    val_mae: float
    val_loss: float
    # Test
    test_r2: float
    test_rmse: float
    test_mae: float
    test_loss: float
    # Training info
    best_epoch: int
    total_epochs: int

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "feature_type": "graph",
            "seed": self.seed,
            "task": self.task,
            "train_r2": self.train_r2,
            "train_rmse": self.train_rmse,
            "train_mae": self.train_mae,
            "train_loss": self.train_loss,
            "val_r2": self.val_r2,
            "val_rmse": self.val_rmse,
            "val_mae": self.val_mae,
            "val_loss": self.val_loss,
            "test_r2": self.test_r2,
            "test_rmse": self.test_rmse,
            "test_mae": self.test_mae,
            "test_loss": self.test_loss,
            "best_epoch": self.best_epoch,
            "total_epochs": self.total_epochs,
        }
