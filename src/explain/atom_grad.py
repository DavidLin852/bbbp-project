from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

@dataclass
class AtomAttribution:
    prob: float
    atom_score: np.ndarray  # (num_atoms,), signed contribution

def grad_x_input_atom_scores(model, data, device="cuda") -> AtomAttribution:
    """
    Signed attribution via Grad×Input on node features x.
    Returns per-atom scalar score (can be +/-).
    """
    dev = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    model = model.to(dev)
    model.eval()

    data = data.to(dev)

    # Make node features require grad
    x = data.x.detach().clone()
    x.requires_grad_(True)
    data.x = x

    logit = model(data)              # shape: [batch] but here should be 1
    prob = torch.sigmoid(logit)[0]

    # Backprop w.r.t x
    model.zero_grad(set_to_none=True)
    prob.backward(retain_graph=False)

    grad = x.grad                    # [num_nodes, feat_dim]
    scores = (grad * x).sum(dim=1)   # [num_nodes]
    scores = scores.detach().cpu().numpy()

    return AtomAttribution(prob=float(prob.detach().cpu().item()), atom_score=scores)
