from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from src.config import Paths, DatasetConfig
from src.featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig
from src.phys_aux.train_gat_aux import TrainCfg, train_gat_multitask

def append_csv(row: dict, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_csv, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lambda_logp", type=float, default=0.3)
    ap.add_argument("--lambda_tpsa", type=float, default=0.3)
    args = ap.parse_args()

    P = Paths()
    D = DatasetConfig()

    split_dir = P.data_splits / f"seed_{args.seed}"
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    cfg = GraphBuildConfig(smiles_col=D.smiles_col, label_col="y_cls", id_col="row_id")

    cache_root = P.features / f"seed_{args.seed}" / "pyg_graphs"

    train_ds = BBBGraphDataset(root=str(cache_root / "train"), df=train_df, cfg=cfg)
    val_ds   = BBBGraphDataset(root=str(cache_root / "val"),   df=val_df,   cfg=cfg)
    test_ds  = BBBGraphDataset(root=str(cache_root / "test"),  df=test_df,  cfg=cfg)

    out_dir = P.models / "gat_aux" / f"seed_{args.seed}" / f"lambda_logp{args.lambda_logp}_tpsa{args.lambda_tpsa}"
    tcfg = TrainCfg(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch,
        lambda_logp=args.lambda_logp,
        lambda_tpsa=args.lambda_tpsa,
    )

    final_row, roc_path = train_gat_multitask(train_ds, val_ds, test_ds, out_dir=out_dir, cfg=tcfg)

    out_csv = P.metrics / "gat_phys_aux.csv"
    append_csv(final_row, out_csv)

    print("Saved:", out_csv)
    print("ROC:", roc_path)

if __name__ == "__main__":
    main()
