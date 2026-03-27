from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from src.config import Paths, DatasetConfig
from src.featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig
from src.finetune.train_gat_bbb_from_pretrain import GATBBB, FinetuneCfg

# ---------- utils ----------

def load_explain_smarts(smarts_json: Path):
    obj = json.loads(smarts_json.read_text(encoding="utf-8"))
    assert isinstance(obj, list), "SMARTS explain file must be a list"
    out = []
    for x in obj:
        if not isinstance(x, dict):
            continue
        if "name" in x and "smarts" in x:
            out.append((x["name"], x["smarts"]))
    if len(out) == 0:
        raise ValueError("No valid {name, smarts} found in JSON")
    return out

@torch.no_grad()
def predict_prob(model, data, device):
    model.eval()
    data = data.to(device)
    logit = model(data)
    return float(torch.sigmoid(logit)[0].item())

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    ap.add_argument("--smarts_json", type=str, required=True)
    ap.add_argument("--topk_per_mol", type=int, default=20)
    ap.add_argument("--max_mols", type=int, default=0)
    ap.add_argument("--min_freq", type=int, default=20)
    args = ap.parse_args()

    P = Paths()
    D = DatasetConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data split
    split_dir = P.data_splits / f"seed_{args.seed}"
    df = pd.read_csv(split_dir / f"{args.split}.csv")
    if args.max_mols > 0:
        df = df.iloc[:args.max_mols].copy()
    if "row_id" not in df.columns:
        df["row_id"] = range(len(df))

    # load SMARTS explain vocab
    explain_smarts = load_explain_smarts(Path(args.smarts_json))

    # build graph dataset
    gcfg = GraphBuildConfig(
        smiles_col=D.smiles_col,
        label_col="y_cls",
        id_col="row_id"
    )
    cache_root = P.features / f"seed_{args.seed}" / f"explain_{args.split}"
    ds = BBBGraphDataset(root=str(cache_root), df=df, cfg=gcfg)

    # load best pretrained+partial model
    ckpt = (
        P.models
        / "gat_finetune_bbb"
        / f"seed_{args.seed}"
        / "pretrained_partial"
        / "best.pt"
    )
    assert ckpt.exists(), f"Model not found: {ckpt}"

    in_dim = ds[0].x.size(-1)
    cfg = FinetuneCfg(init="pretrained", strategy="partial", partial_k=2)
    model = GATBBB(in_dim, cfg).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    # precompile SMARTS patterns
    patterns = []
    for name, s in explain_smarts:
        mol = Chem.MolFromSmarts(s)
        if mol is not None:
            patterns.append((name, s, mol))
    print(f"Loaded {len(patterns)} valid SMARTS patterns")

    rows = []

    for i in range(len(ds)):
        smi = df.loc[i, D.smiles_col]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        data = ds[i]
        base_prob = predict_prob(model, data, device)
        x0 = data.x.detach().clone()

        local = []
        for name, s, patt in patterns:
            matches = mol.GetSubstructMatches(patt)
            if not matches:
                continue
            atom_ids = sorted(set(a for m in matches for a in m))
            if not atom_ids:
                continue

            data.x = x0.clone()
            data.x[atom_ids, :] = 0.0
            p2 = predict_prob(model, data, device)
            delta = base_prob - p2

            local.append((name, s, delta, len(atom_ids)))

        data.x = x0

        local.sort(key=lambda t: abs(t[2]), reverse=True)
        local = local[:args.topk_per_mol]

        y_true = int(df.loc[i, "y_cls"])
        for name, s, delta, n_atoms in local:
            rows.append({
                "mol_idx": i,
                "smiles": smi,
                "y_true": y_true,
                "base_prob": base_prob,
                "smarts_name": name,
                "smarts": s,
                "delta_prob": delta,
                "n_match_atoms": n_atoms
            })

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(ds)}")

    outdir = P.metrics / "global_explain"
    outdir.mkdir(parents=True, exist_ok=True)

    per_mol = pd.DataFrame(rows)
    per_mol_path = outdir / f"smarts_occlusion_{args.split}_seed{args.seed}.csv"
    per_mol.to_csv(per_mol_path, index=False)

    # global aggregation
    grp = per_mol.groupby("smarts_name")
    agg = grp["delta_prob"].agg(
        support="count",
        mean_delta="mean",
        median_delta="median",
        mean_abs_delta=lambda x: float(np.mean(np.abs(x))),
        std_delta="std"
    ).reset_index()

    freq = grp["mol_idx"].nunique().reset_index().rename(columns={"mol_idx":"freq_mols"})
    agg = agg.merge(freq, on="smarts_name", how="left")
    # =========================
    # frequency filtering
    # =========================
    agg = agg[agg["freq_mols"] >= args.min_freq]

    pos = agg.sort_values("mean_delta", ascending=False)
    neg = agg.sort_values("mean_delta", ascending=True)

    pos.to_csv(outdir / f"global_smarts_positive_{args.split}_seed{args.seed}.csv", index=False)
    neg.to_csv(outdir / f"global_smarts_negative_{args.split}_seed{args.seed}.csv", index=False)

    print("Saved:")
    print(" -", per_mol_path)
    print(" -", outdir / f"global_smarts_positive_{args.split}_seed{args.seed}.csv")
    print(" -", outdir / f"global_smarts_negative_{args.split}_seed{args.seed}.csv")

if __name__ == "__main__":
    main()
