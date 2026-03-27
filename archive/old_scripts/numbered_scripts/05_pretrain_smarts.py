from __future__ import annotations
import sys
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Paths, DatasetConfig
from src.pretrain.graph_pyg_smarts import BBBGraphSmartsDataset, SmartsCfg
from src.pretrain.train_gat_smarts import PretrainCfg, pretrain_smarts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min_freq", type=float, default=0.01)
    args = ap.parse_args()

    P = Paths()
    D = DatasetConfig()

    split_dir = P.data_splits / f"seed_{args.seed}"
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")

    cfg = SmartsCfg(smiles_col=D.smiles_col, label_col="y_cls", id_col="row_id",
                    smarts_json="assets/smarts/bbb_smarts_v1.json")

    cache_root = P.features / f"seed_{args.seed}" / "pyg_graphs_smarts_v1"

    train_ds = BBBGraphSmartsDataset(root=str(cache_root / "train"), df=train_df, cfg=cfg)
    val_ds   = BBBGraphSmartsDataset(root=str(cache_root / "val"),   df=val_df,   cfg=cfg)

    out_dir = P.models / "gat_pretrain_smarts" / f"seed_{args.seed}" / "bbb_smarts_v1"
    pcfg = PretrainCfg(seed=args.seed, epochs=args.epochs, batch_size=args.batch, min_freq=args.min_freq)

    ckpt_best, hist_csv = pretrain_smarts(train_ds, val_ds, out_dir, pcfg)

    print("Best ckpt:", ckpt_best)
    print("History :", hist_csv)

if __name__ == "__main__":
    main()
