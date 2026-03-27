from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import torch

from src.config import Paths, DatasetConfig
from src.featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig
from src.finetune.train_gat_bbb_from_pretrain import GATBBB, FinetuneCfg
from src.explain.smarts_occlusion import occlusion_smarts

def load_model(ckpt_path: Path, in_dim: int, device: str):
    cfg = FinetuneCfg(init="pretrained", strategy="partial", partial_k=2)
    model = GATBBB(in_dim, cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device)

def load_smarts_json(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    for k in ["smarts", "smarts_list", "kept_smarts", "vocab"]:
        if k in obj and isinstance(obj[k], list):
            return obj[k]
    raise ValueError("Unsupported SMARTS json format")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--smarts_json", type=str, required=True)
    ap.add_argument("--smiles", type=str, default="")
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--smiles_col", type=str, default="SMILES")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    P = Paths()
    D = DatasetConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = Path(args.ckpt) if args.ckpt else (P.models / "gat_finetune_bbb" / f"seed_{args.seed}" / "pretrained_partial" / "best.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt}")

    smarts_list = load_smarts_json(Path(args.smarts_json))

    gcfg = GraphBuildConfig(smiles_col=D.smiles_col, label_col="y_cls", id_col="row_id")

    outdir = P.metrics / "explain_smarts"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.smiles.strip():
        df = pd.DataFrame({D.smiles_col: [args.smiles], "y_cls": [0], "row_id": [0]})
        ds = BBBGraphDataset(root=str(P.features / f"seed_{args.seed}" / "tmp_smarts_one"), df=df, cfg=gcfg)
        in_dim = ds[0].x.size(-1)
        model = load_model(ckpt, in_dim, device)

        base_prob, top = occlusion_smarts(model, ds[0], args.smiles, smarts_list, device=device, topk=args.topk)
        out_json = outdir / "one_smarts_top.json"
        out_json.write_text(json.dumps({
            "smiles": args.smiles,
            "base_prob": base_prob,
            "topk": [c.__dict__ for c in top]
        }, indent=2), encoding="utf-8")
        print("Saved:", out_json)
        return

    if not args.csv.strip():
        raise ValueError("Provide either --smiles or --csv")

    df_in = pd.read_csv(Path(args.csv))
    if args.smiles_col not in df_in.columns:
        raise ValueError(f"CSV missing column: {args.smiles_col}")

    df = df_in.copy()
    df[D.smiles_col] = df[args.smiles_col].astype(str)
    df["y_cls"] = 0
    df["row_id"] = range(len(df))

    ds = BBBGraphDataset(root=str(P.features / f"seed_{args.seed}" / "tmp_smarts_batch"), df=df, cfg=gcfg)
    in_dim = ds[0].x.size(-1)
    model = load_model(ckpt, in_dim, device)

    n = min(args.limit, len(ds))
    rows = []
    for i in range(n):
        smi = df.loc[i, D.smiles_col]
        base_prob, top = occlusion_smarts(model, ds[i], smi, smarts_list, device=device, topk=args.topk)
        for rank, c in enumerate(top, start=1):
            rows.append({
                "i": i,
                "rank": rank,
                "smiles": smi,
                "base_prob": base_prob,
                "smarts": c.smarts,
                "delta_prob": c.delta_prob,
                "match_atoms": ",".join(map(str, c.match_atoms)),
            })

    out_csv = outdir / "smarts_contrib_topk.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()
