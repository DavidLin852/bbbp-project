from __future__ import annotations
import sys
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Paths, DatasetConfig
from src.featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig
from src.finetune.train_gat_bbb_from_pretrain import FinetuneCfg, finetune_bbb

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
    ap.add_argument("--partial_k", type=int, default=2)
    ap.add_argument("--pretrain_ckpt", type=str, default="")
    ap.add_argument("--only_pretrained", action="store_true", help="only run pretrained init for 3 strategies")
    args = ap.parse_args()

    P = Paths()
    D = DatasetConfig()

    split_dir = P.data_splits / f"seed_{args.seed}"
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    gcfg = GraphBuildConfig(smiles_col=D.smiles_col, label_col="y_cls", id_col="row_id")

    cache_root = P.features / f"seed_{args.seed}" / "pyg_graphs_bbb"
    train_ds = BBBGraphDataset(root=str(cache_root / "train"), df=train_df, cfg=gcfg)
    val_ds   = BBBGraphDataset(root=str(cache_root / "val"),   df=val_df,   cfg=gcfg)
    test_ds  = BBBGraphDataset(root=str(cache_root / "test"),  df=test_df,  cfg=gcfg)

    # pretrain ckpt default path
    if args.pretrain_ckpt.strip():
        pre_ckpt = Path(args.pretrain_ckpt)
    else:
        pre_ckpt = P.models / "gat_pretrain_smarts" / f"seed_{args.seed}" / "bbb_smarts_v1" / "best.pt"
    if not pre_ckpt.exists():
        raise FileNotFoundError(f"pretrain ckpt not found: {pre_ckpt}")

    inits = ["random"]
    strategies = ["full"]

    out_csv = P.metrics / "gat_finetune_bbb.csv"

    for init in inits:
        for strat in strategies:
            cfg = FinetuneCfg(
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch,
                partial_k=args.partial_k,
                init=init,
                strategy=strat,
            )
            out_dir = P.models / "gat_finetune_bbb" / f"seed_{args.seed}" / f"{init}_{strat}"
            row, roc_path = finetune_bbb(train_ds, val_ds, test_ds, out_dir, cfg, pre_ckpt)
            append_csv(row, out_csv)
            print(f"[DONE] init={init} strat={strat} -> {roc_path}")

    print("Saved metrics:", out_csv)

if __name__ == "__main__":
    main()
