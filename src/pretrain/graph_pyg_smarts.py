from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from rdkit import Chem
from ..featurize.graph_pyg import smiles_to_graph  # 复用你已有的图构建
from .smarts_labels import load_smarts_list, smarts_multi_hot

@dataclass(frozen=True)
class SmartsCfg:
    smiles_col: str = "SMILES"
    label_col: str = "y_cls"
    id_col: str = "row_id"
    smarts_json: str = "assets/smarts/bbb_smarts_v1.json"

class BBBGraphSmartsDataset(InMemoryDataset):
    def __init__(self, root: str, df: pd.DataFrame, cfg: SmartsCfg, transform=None, pre_transform=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.smarts_names, self.smarts_patt = load_smarts_list(Path(cfg.smarts_json))
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    @property
    def raw_file_names(self) -> List[str]:
        return []

    def download(self): return

    def process(self):
        data_list: List[Data] = []
        for i in range(len(self.df)):
            smi = str(self.df.loc[i, self.cfg.smiles_col]).strip()
            rid = self.df.loc[i, self.cfg.id_col]
            y_cls = int(self.df.loc[i, self.cfg.label_col])

            g = smiles_to_graph(smi)
            if g is None:
                continue
            x, edge_index, edge_attr = g

            y_sm = smarts_multi_hot(smi, self.smarts_patt)
            if y_sm is None:
                continue

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y_cls=torch.tensor([y_cls], dtype=torch.float),
                y_smarts=torch.tensor(y_sm, dtype=torch.float).unsqueeze(0),
            )
            data.row_id = str(rid)
            data.smiles = smi
            data_list.append(data)

        data, slices = self.collate(data_list)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
