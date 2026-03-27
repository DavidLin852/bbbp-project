from __future__ import annotations
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

def _normalize_for_color(values: np.ndarray):
    # symmetric normalization around 0
    vmax = float(np.max(np.abs(values))) if values.size else 1.0
    vmax = max(vmax, 1e-8)
    return values / vmax

def draw_atom_attribution(smiles: str, atom_score: np.ndarray, out_path: Path, legend: str = ""):
    """
    RDKit 2D depiction with atom highlights.
    Positive = red, negative = blue (RDKit default colormap logic via (r,g,b)).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles}")

    s = _normalize_for_color(atom_score)

    # Map each atom to color
    # red for positive, blue for negative
    atom_colors = {}
    atom_radii = {}
    for i, v in enumerate(s):
        if v >= 0:
            atom_colors[i] = (1.0, 0.2, 0.2)
        else:
            atom_colors[i] = (0.2, 0.2, 1.0)
        atom_radii[i] = 0.25 + 0.35 * float(abs(v))  # 0.25 ~ 0.6

    drawer = Draw.MolDraw2DCairo(900, 600)
    opts = drawer.drawOptions()
    opts.legendFontSize = 22

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        legend=legend
    )
    drawer.FinishDrawing()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(drawer.GetDrawingText())
