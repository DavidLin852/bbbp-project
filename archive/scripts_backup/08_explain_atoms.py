from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch

from src.config import Paths, DatasetConfig
from src.featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig
from src.finetune.train_gat_bbb_from_pretrain import GATBBB, FinetuneCfg
from src.explain.atom_grad import grad_x_input_atom_scores
from src.explain.draw_rdkit import draw_atom_attribution

def load_model(ckpt_path: Path, in_dim: int, device: str):
    cfg = FinetuneCfg(init="pretrained", strategy="partial", partial_k=2)
    model = GATBBB(in_dim, cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--smiles", type=str, default="")
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--smiles_col", type=str, default="SMILES")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    P = Paths()
    D = DatasetConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = Path(args.ckpt) if args.ckpt else (P.models / "gat_finetune_bbb" / f"seed_{args.seed}" / "pretrained_partial" / "best.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt}")

    outdir = Path(args.outdir) if args.outdir else (P.metrics / "explain_atoms")
    outdir.mkdir(parents=True, exist_ok=True)

    gcfg = GraphBuildConfig(smiles_col=D.smiles_col, label_col="y_cls", id_col="row_id")

    # Build a tiny dataframe for single-smiles mode
    if args.smiles.strip():
        df = pd.DataFrame({D.smiles_col: [args.smiles], "y_cls": [0], "row_id": [0]})
        ds = BBBGraphDataset(root=str(P.features / f"seed_{args.seed}" / "tmp_explain_one"), df=df, cfg=gcfg)
        in_dim = ds[0].x.size(-1)
        model = load_model(ckpt, in_dim, device)

        attr = grad_x_input_atom_scores(model, ds[0], device=device)
        legend = f"prob={attr.prob:.3f}"
        draw_atom_attribution(args.smiles, attr.atom_score, outdir / "one_atom_attr.png", legend=legend)
        print("Saved:", outdir / "one_atom_attr.png")
        return

    # Batch csv mode
    if not args.csv.strip():
        raise ValueError("Provide either --smiles or --csv")

    csv_path = Path(args.csv)
    df_in = pd.read_csv(csv_path)
    if args.smiles_col not in df_in.columns:
        raise ValueError(f"CSV missing column: {args.smiles_col}")

    df = df_in.copy()
    df[D.smiles_col] = df[args.smiles_col].astype(str)
    df["y_cls"] = 0
    df["row_id"] = range(len(df))

    ds = BBBGraphDataset(root=str(P.features / f"seed_{args.seed}" / "tmp_explain_batch"), df=df, cfg=gcfg)
    in_dim = ds[0].x.size(-1)
    model = load_model(ckpt, in_dim, device)

    n = min(args.limit, len(ds))
    rows = []
    for i in range(n):
        smi = df.loc[i, D.smiles_col]
        attr = grad_x_input_atom_scores(model, ds[i], device=device)
        legend = f"i={i} prob={attr.prob:.3f}"
        out_png = outdir / f"atom_attr_{i}.png"
        draw_atom_attribution(smi, attr.atom_score, out_png, legend=legend)
        rows.append({"i": i, "smiles": smi, "prob": attr.prob, "png": str(out_png)})

    pd.DataFrame(rows).to_csv(outdir / "atom_attr_index.csv", index=False)
    print("Saved batch index:", outdir / "atom_attr_index.csv")

if __name__ == "__main__":
    main()
