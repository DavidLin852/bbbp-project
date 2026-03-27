from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np
import torch
from rdkit import Chem

@dataclass
class SmartsContribution:
    smarts: str
    delta_prob: float
    match_atoms: list[int]

def load_smarts_list(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    # common wrappers
    for k in ["smarts", "smarts_list", "kept_smarts", "vocab"]:
        if k in obj and isinstance(obj[k], list):
            return obj[k]
    raise ValueError("Unsupported SMARTS json format")

@torch.no_grad()
def predict_prob(model, data, device):
    model.eval()
    data = data.to(device)
    logit = model(data)
    prob = torch.sigmoid(logit)[0].item()
    return float(prob)

def occlusion_smarts(model, data, smiles: str, smarts_list: list[str], device="cuda", topk=20):
    dev = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    model = model.to(dev)

    base_prob = predict_prob(model, data, dev)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles}")

    contribs = []
    x0 = data.x.detach().clone()

    for s in smarts_list:
        patt = Chem.MolFromSmarts(s)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue

        # union all matched atoms
        atom_set = sorted(set([a for m in matches for a in m]))
        if not atom_set:
            continue

        data.x = x0.clone()
        data.x[atom_set, :] = 0.0  # occlude node features

        prob2 = predict_prob(model, data, dev)
        delta = base_prob - prob2  # drop means positive contribution
        contribs.append(SmartsContribution(smarts=s, delta_prob=float(delta), match_atoms=atom_set))

    # restore
    data.x = x0

    contribs.sort(key=lambda c: abs(c.delta_prob), reverse=True)
    return base_prob, contribs[:topk]
