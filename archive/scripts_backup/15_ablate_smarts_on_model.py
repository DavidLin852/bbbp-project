# -*- coding: utf-8 -*-
"""
Ablation test for SMARTS motifs on a trained GAT BBB model.

- Load a finetuned BBB classifier checkpoint.
- Load split CSV (train/val/test) containing SMILES.
- For each molecule, compute baseline BBB prob p0.
- For each SMARTS motif, find matched atoms, then:
  (1) mask: zero-out node features of matched atoms
  (2) cut : remove edges internal to the matched atom set
- Re-run model, compute deltas (p_after - p0), aggregate globally.

Outputs:
- artifacts/explain/ablation_per_mol_<split>.csv
- artifacts/explain/ablation_summary_<split>.csv
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from rdkit import Chem

# project imports
from src.config import Paths, DatasetConfig
from src.featurize.graph_pyg import BBBGraphDataset
from src.finetune.train_gat_bbb_from_pretrain import GATBBB, FinetuneCfg

# PyTorch 2.6 safe loading workaround (you already ran into this)
from torch.serialization import add_safe_globals
import numpy as _np

add_safe_globals([_np.core.multiarray._reconstruct, _np.ndarray])


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


@torch.no_grad()
def predict_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        batch = batch.to(device)
        logit = model(batch)  # expected shape [B] or [B,1]
        if logit.ndim == 2 and logit.shape[1] == 1:
            logit = logit[:, 0]
        p = sigmoid(logit).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def _mask_nodes(data, atom_idx: list[int]):
    d = data.clone()
    if len(atom_idx) == 0:
        return d
    idx = torch.tensor(atom_idx, dtype=torch.long)
    d.x[idx, :] = 0.0
    return d


def _cut_internal_edges(data, atom_idx: list[int]):
    d = data.clone()
    if len(atom_idx) == 0:
        return d
    s = set(atom_idx)
    ei = d.edge_index  # [2, E]
    src = ei[0].cpu().numpy()
    dst = ei[1].cpu().numpy()
    keep = []
    for k in range(ei.shape[1]):
        a = int(src[k]); b = int(dst[k])
        # remove edges where both endpoints are in the matched atom set
        if (a in s) and (b in s):
            keep.append(False)
        else:
            keep.append(True)
    keep = torch.tensor(keep, dtype=torch.bool)
    d.edge_index = d.edge_index[:, keep]
    if hasattr(d, "edge_attr") and d.edge_attr is not None:
        d.edge_attr = d.edge_attr[keep]
    return d


def find_first_match_atoms(smiles: str, smarts: str) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return []
    matches = mol.GetSubstructMatches(patt)
    if not matches:
        return []
    # IMPORTANT: for ablation, you can:
    # - ablate only the first match (current)
    # - or union all matches (more aggressive)
    # Here we union ALL matches to reflect presence anywhere in molecule.
    atoms = sorted(set([a for m in matches for a in m]))
    return atoms


def load_smarts_json(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    # accept either {"patterns":[...]} or [...]
    if isinstance(obj, dict) and "patterns" in obj:
        obj = obj["patterns"]
    assert isinstance(obj, list), "SMARTS json must be a list or {patterns:[...]}"
    # each item: {"name":..., "smarts":...}
    return obj


def build_dataset(df: pd.DataFrame, cache_root: Path, cfg: DatasetConfig) -> BBBGraphDataset:
    # BBBGraphDataset expects certain columns. We only need smiles column; labels may be required internally.
    # If your BBBGraphDataset uses cfg.bbb_col and expects numeric labels, ensure it exists.
    # Here we add a dummy numeric label if needed.
    if cfg.bbb_col not in df.columns:
        df[cfg.bbb_col] = 0
    else:
        # if BBB+/BBB- strings exist, convert to 1/0 for dataset processing
        if df[cfg.bbb_col].dtype == object:
            df[cfg.bbb_col] = df[cfg.bbb_col].map(lambda x: 1 if str(x).strip() == "BBB+" else 0)

    # id_cols may be tuple; if dataset expects it and split csv lacks them, create placeholders
    for c in cfg.id_cols:
        if c not in df.columns:
            df[c] = -1

    ds = BBBGraphDataset(root=str(cache_root), df=df, cfg=cfg)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=64)

    # model ckpt (BBB finetune)
    ap.add_argument(
        "--ckpt",
        type=str,
        default=r"artifacts\models\gat_finetune_bbb\seed_0\pretrained_partial\best.pt",
        help="Path to finetuned BBB checkpoint",
    )

    # SMARTS list
    ap.add_argument(
        "--smarts_json",
        type=str,
        default=r"artifacts\explain\smarts_bbb_explain.json",
        help="JSON list of {name, smarts}",
    )

    ap.add_argument("--mode", type=str, default="mask", choices=["mask", "cut", "both"])
    ap.add_argument("--min_freq", type=int, default=20, help="Filter motifs by molecule frequency >= min_freq")
    ap.add_argument("--max_mols", type=int, default=-1, help="If >0, only evaluate first N molecules")

    args = ap.parse_args()

    P = Paths()
    cfg = DatasetConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    split_csv = Path(P.data_splits) / f"seed_{args.seed}" / f"{args.split}.csv"
    print(f"[INFO] split_csv={split_csv}")

    df = pd.read_csv(split_csv)
    # === 强制注入 dummy BBB 列，绕过 graph_pyg 的标签解析 ===
    if "BBB+/BBB-" not in df.columns:
        df["BBB+/BBB-"] = "BBB+"  # 随便给一个合法值，只是为了不报错

    if args.max_mols > 0:
        df = df.iloc[: args.max_mols].reset_index(drop=True)

    ckpt_path = Path(args.ckpt)
    print(f"[INFO] loading model ckpt={ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # build dataset to infer in_dim
    cache_root = Path(P.artifacts) / "cache" / f"ablation_{args.split}_seed{args.seed}"
    ds = build_dataset(df.copy(), cache_root, cfg)
    in_dim = int(ds[0].x.shape[1])
    print(f"[INFO] inferred in_dim={in_dim}")

    # model config: if ckpt stores cfg dict, use it; else fall back
    model_cfg = None
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        # ckpt["cfg"] may be a dict for FinetuneCfg
        if isinstance(ckpt["cfg"], dict):
            model_cfg = FinetuneCfg(**{k: v for k, v in ckpt["cfg"].items() if k in asdict(FinetuneCfg()).keys()})
        else:
            model_cfg = ckpt["cfg"]
    if model_cfg is None:
        model_cfg = FinetuneCfg()

    model = GATBBB(in_dim=in_dim, cfg=model_cfg).to(device)

    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    # tolerate head key style differences (head vs head.net)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print("[WARN] strict load failed, trying key remap for head.* <-> head.net.*")
        new_state = {}
        for k, v in state.items():
            if k.startswith("head.net."):
                new_state[k.replace("head.net.", "head.")] = v
            elif k.startswith("head."):
                new_state[k.replace("head.", "head.net.")] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=False)

    model.eval()

    # baseline probs
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)
    p0 = predict_probs(model, loader, device)

    # load smarts patterns
    patterns = load_smarts_json(Path(args.smarts_json))

    smiles_list = df[cfg.smiles_col].astype(str).tolist()

    # compute motif supports (freq_mols)
    motif_atoms = []
    freq = []
    for pat in patterns:
        atoms_each = [find_first_match_atoms(smi, pat["smarts"]) for smi in smiles_list]
        motif_atoms.append(atoms_each)
        freq.append(sum(1 for a in atoms_each if len(a) > 0))
    freq = np.array(freq, dtype=int)

    # filter by min_freq
    keep_idx = np.where(freq >= args.min_freq)[0]
    patterns = [patterns[i] for i in keep_idx]
    motif_atoms = [motif_atoms[i] for i in keep_idx]
    freq = freq[keep_idx]
    print(f"[INFO] kept motifs: {len(patterns)} (min_freq={args.min_freq})")

    per_mol_rows = []

    # evaluate each motif in batches by rebuilding modified dataset on the fly
    for mi, pat in enumerate(patterns):
        name = pat["name"]
        smarts = pat["smarts"]
        atoms_list = motif_atoms[mi]

        # build modified datas
        data_list_mask = []
        data_list_cut = []
        has = np.array([1 if len(a) > 0 else 0 for a in atoms_list], dtype=int)

        for i in range(len(ds)):
            data = ds[i]
            atoms = atoms_list[i]
            if args.mode in ("mask", "both"):
                data_list_mask.append(_mask_nodes(data, atoms))
            if args.mode in ("cut", "both"):
                data_list_cut.append(_cut_internal_edges(data, atoms))

        if args.mode in ("mask", "both"):
            loader_m = DataLoader(data_list_mask, batch_size=args.batch, shuffle=False)
            pm = predict_probs(model, loader_m, device)
        else:
            pm = None

        if args.mode in ("cut", "both"):
            loader_c = DataLoader(data_list_cut, batch_size=args.batch, shuffle=False)
            pc = predict_probs(model, loader_c, device)
        else:
            pc = None

        for i, smi in enumerate(smiles_list):
            row = {
                "idx": i,
                "SMILES": smi,
                "p0": float(p0[i]),
                "smarts_name": name,
                "smarts": smarts,
                "has_motif": int(has[i]),
                "n_atoms_hit": int(len(atoms_list[i])),
            }
            if pm is not None:
                row["p_mask"] = float(pm[i])
                row["delta_mask"] = float(pm[i] - p0[i])
            if pc is not None:
                row["p_cut"] = float(pc[i])
                row["delta_cut"] = float(pc[i] - p0[i])

            per_mol_rows.append(row)

        print(f"[INFO] done motif {mi+1}/{len(patterns)}: {name} (freq={freq[mi]})")

    per_mol = pd.DataFrame(per_mol_rows)
    out_dir = Path(P.artifacts) / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_mol_path = out_dir / f"ablation_per_mol_{args.split}_seed{args.seed}.csv"
    per_mol.to_csv(per_mol_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved per-mol: {per_mol_path}")

    # summary
    def summarize(delta_col: str):
        g = per_mol[per_mol["has_motif"] == 1].groupby("smarts_name", as_index=False).agg(
            support=("has_motif", "sum"),
            freq_mols=("has_motif", "sum"),
            mean_delta=(delta_col, "mean"),
            median_delta=(delta_col, "median"),
            mean_abs_delta=(delta_col, lambda x: float(np.mean(np.abs(x)))),
            std_delta=(delta_col, "std"),
        )
        # sort: positive to negative (top->bottom)
        g = g.sort_values("mean_delta", ascending=False).reset_index(drop=True)
        return g

    summaries = []
    if "delta_mask" in per_mol.columns:
        s = summarize("delta_mask")
        s["mode"] = "mask"
        summaries.append(s)
    if "delta_cut" in per_mol.columns:
        s = summarize("delta_cut")
        s["mode"] = "cut"
        summaries.append(s)

    summary = pd.concat(summaries, axis=0).reset_index(drop=True)
    summary_path = out_dir / f"ablation_summary_{args.split}_seed{args.seed}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved summary: {summary_path}")

    # =========================
    # Plot: Motif effect vs support
    # =========================
    import matplotlib.pyplot as plt

    def plot_summary(df, mode: str, out_png: Path, top_k_labels=20):
        df = df[df["mode"] == mode].copy()
        if df.empty:
            print(f"[WARN] No data for mode={mode}, skip plot.")
            return

        x = df["freq_mols"].values
        y = df["mean_delta"].values
        size = df["std_delta"].fillna(0).values

        # normalize size
        size = 50 + 300 * (size / (size.max() + 1e-6))

        colors = ["red" if v > 0 else "blue" for v in y]

        plt.figure(figsize=(10, 7))
        plt.scatter(x, y, s=size, c=colors, alpha=0.7, edgecolors="k")

        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Support (freq_mols)")
        plt.ylabel("Mean Δ probability after ablation")
        plt.title(f"SMARTS Ablation Effect ({mode})")

        # annotate top-K by |mean_delta|
        df["_abs"] = df["mean_delta"].abs()
        df_top = df.sort_values("_abs", ascending=False).head(top_k_labels)

        for _, r in df_top.iterrows():
            plt.text(
                r["freq_mols"],
                r["mean_delta"],
                r["smarts_name"],
                fontsize=9,
                ha="left",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[INFO] saved plot: {out_png}")

    fig_dir = out_dir
    if "mask" in summary["mode"].values:
        plot_summary(summary, "mask", fig_dir / f"ablation_mask_{args.split}_seed{args.seed}.png")
    if "cut" in summary["mode"].values:
        plot_summary(summary, "cut", fig_dir / f"ablation_cut_{args.split}_seed{args.seed}.png")

if __name__ == "__main__":
    main()
