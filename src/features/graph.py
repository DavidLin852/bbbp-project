"""
Graph representation for GNN models.

Converts SMILES to PyTorch Geometric Data objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors


# Atom and bond feature lists
ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H,C,N,O,F,P,S,Cl,Br,I
HYB_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(x, xs):
    """One-hot encoding."""
    return [1.0 if x == v else 0.0 for v in xs]


def atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    """Compute atom features."""
    z = atom.GetAtomicNum()
    feats = []

    # Atomic number
    feats += one_hot(z, ATOM_LIST) + [1.0 if z not in ATOM_LIST else 0.0]

    # Degree, H count, formal charge
    feats += [
        float(atom.GetDegree()),
        float(atom.GetTotalNumHs()),
        float(atom.GetFormalCharge()),
    ]

    # Hybridization
    feats += one_hot(atom.GetHybridization(), HYB_LIST) + [
        1.0 if atom.GetHybridization() not in HYB_LIST else 0.0
    ]

    # Aromaticity and mass
    feats += [1.0 if atom.GetIsAromatic() else 0.0]
    feats += [float(atom.GetMass() * 0.01)]  # scaled mass

    return feats


def bond_features(bond: Chem.rdchem.Bond) -> List[float]:
    """Compute bond features."""
    bt = bond.GetBondType()
    feats = []

    # Bond type
    feats += one_hot(bt, BOND_LIST) + [1.0 if bt not in BOND_LIST else 0.0]

    # Conjugation and ring
    feats += [1.0 if bond.GetIsConjugated() else 0.0]
    feats += [1.0 if bond.IsInRing() else 0.0]

    return feats


def smiles_to_graph(smiles: str) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Convert SMILES to graph tensors.

    Args:
        smiles: SMILES string

    Returns:
        Tuple of (node_features, edge_index, edge_features) or None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    x = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()], dtype=torch.float
    )

    # Edge features
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        edge_index.append([i, j])
        edge_attr.append(bf)
        edge_index.append([j, i])
        edge_attr.append(bf)

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 7), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return x, edge_index, edge_attr


@dataclass
class GraphConfig:
    """Configuration for graph dataset."""
    smiles_col: str = "SMILES_canon"
    label_col: str = "y_cls"
    id_col: str = "row_id"


class GraphGenerator:
    """
    Generate graph representations for GNN models.

    Converts SMILES to PyTorch Geometric Data objects
    with node and edge features.
    """

    def __init__(self, config: GraphConfig | None = None):
        self.config = config or GraphConfig()

    def compute(
        self,
        smiles: str,
        label: float | int,
        mol_id: str | int,
    ) -> Data | None:
        """
        Convert single SMILES to graph Data object.

        Args:
            smiles: SMILES string
            label: Label value
            mol_id: Molecule identifier

        Returns:
            PyG Data object or None if invalid
        """
        graph = smiles_to_graph(smiles)
        if graph is None:
            return None

        x, edge_index, edge_attr = graph

        # Compute auxiliary features
        mol = Chem.MolFromSmiles(smiles)
        logp = float(Crippen.MolLogP(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_cls=torch.tensor([label], dtype=torch.float),
            y_logp=torch.tensor([logp], dtype=torch.float),
            y_tpsa=torch.tensor([tpsa], dtype=torch.float),
        )
        data.mol_id = str(mol_id)
        data.smiles = smiles

        return data

    def compute_batch(
        self,
        df: pd.DataFrame,
    ) -> list[Data]:
        """
        Convert dataframe to list of Data objects.

        Args:
            df: DataFrame with SMILES and labels

        Returns:
            List of PyG Data objects
        """
        data_list = []

        for i in range(len(df)):
            smi = str(df.loc[i, self.config.smiles_col]).strip()
            label = float(df.loc[i, self.config.label_col])
            mol_id = df.loc[i, self.config.id_col]

            data = self.compute(smi, label, mol_id)
            if data is not None:
                data_list.append(data)

        return data_list
