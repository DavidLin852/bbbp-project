from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from rdkit import Chem

def load_smarts_list(path: Path) -> Tuple[List[str], List[Chem.Mol]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    names, patt = [], []
    for it in items:
        sm = it["smarts"].replace(" ", "")
        m = Chem.MolFromSmarts(sm)
        if m is None:
            continue
        names.append(it["name"])
        patt.append(m)
    return names, patt

def smarts_multi_hot(smiles: str, patt: List[Chem.Mol]) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    y = np.zeros((len(patt),), dtype=np.float32)
    for i, p in enumerate(patt):
        if mol.HasSubstructMatch(p):
            y[i] = 1.0
    return y
