from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors

# -------- Atom/Bond featurization (lightweight but solid) --------
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
    return [1.0 if x == v else 0.0 for v in xs]

def atom_features(a: Chem.rdchem.Atom) -> List[float]:
    z = a.GetAtomicNum()
    feats = []
    feats += one_hot(z, ATOM_LIST) + [1.0 if z not in ATOM_LIST else 0.0]
    feats += [float(a.GetDegree()), float(a.GetTotalNumHs()), float(a.GetFormalCharge())]
    feats += one_hot(a.GetHybridization(), HYB_LIST) + [1.0 if a.GetHybridization() not in HYB_LIST else 0.0]
    feats += [1.0 if a.GetIsAromatic() else 0.0]
    feats += [float(a.GetMass() * 0.01)]  # scaled mass
    return feats

def bond_features(b: Chem.rdchem.Bond) -> List[float]:
    bt = b.GetBondType()
    feats = []
    feats += one_hot(bt, BOND_LIST) + [1.0 if bt not in BOND_LIST else 0.0]
    feats += [1.0 if b.GetIsConjugated() else 0.0]
    feats += [1.0 if b.IsInRing() else 0.0]
    return feats

def smiles_to_graph(smiles: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index = []
    edge_attr = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_index.append([i, j]); edge_attr.append(bf)
        edge_index.append([j, i]); edge_attr.append(bf)

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 7), dtype=torch.float)  # bond_features length
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return x, edge_index, edge_attr

def rdkit_logp_and_tpsa(smiles: str) -> tuple[float, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan"), float("nan")
    logp = float(Crippen.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    return logp, tpsa

@dataclass(frozen=True)
class GraphBuildConfig:
    smiles_col: str = "SMILES"
    label_col: str = "y_cls"
    id_col: str = "row_id"

class BBBGraphDataset(InMemoryDataset):
    """
    y_cls: float {0,1} (for BCEWithLogits)
    y_logp: float (RDKit computed)
    y_tpsa: float (RDKit computed)
    """
    def __init__(
        self,
        root: str,
        df: pd.DataFrame,
        cfg: GraphBuildConfig = GraphBuildConfig(),
        transform=None,
        pre_transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        super().__init__(root, transform, pre_transform)

        # PyTorch 2.6+: allow non-weights objects (PyG Data)
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
        for i in range(len(self.df)):
            smi = str(self.df.loc[i, self.cfg.smiles_col]).strip()
            y_cls = int(self.df.loc[i, self.cfg.label_col])
            rid = self.df.loc[i, self.cfg.id_col]

            g = smiles_to_graph(smi)
            if g is None:
                continue
            x, edge_index, edge_attr = g

            logp, tpsa = rdkit_logp_and_tpsa(smi)
            if np.isnan(logp) or np.isnan(tpsa):
                continue

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y_cls=torch.tensor([y_cls], dtype=torch.float),
                y_logp=torch.tensor([logp], dtype=torch.float),
                y_tpsa=torch.tensor([tpsa], dtype=torch.float),
            )
            data.row_id = str(rid)
            data.smiles = smi
            data_list.append(data)

        data, slices = self.collate(data_list)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
