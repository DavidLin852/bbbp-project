from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

from src.config import Paths, FeaturizeConfig
from src.baseline.train_rf_xgb_lgb import train_eval_models
from src.baseline.eval_baselines import append_metrics_csv
from src.utils.plotting import plot_roc_curves

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature", type=str, default="morgan", choices=["morgan", "desc"])
    args = ap.parse_args()

    P = Paths()
    F = FeaturizeConfig()

    feat_dir = P.features / f"seed_{args.seed}"
    meta = pd.read_csv(feat_dir / "meta.csv")
    y = meta["y_cls"].to_numpy().astype(int)
    split = meta["split"].to_numpy()

    if args.feature == "desc":
        X = pd.read_csv(feat_dir / "descriptors.csv").to_numpy()
    else:
        X = sparse.load_npz(feat_dir / f"morgan_{F.morgan_bits}.npz")

    def sel(s):
        m = (split == s)
        return X[m], y[m]

    X_train, y_train = sel("train")
    X_val, y_val = sel("val")
    X_test, y_test = sel("test")

    run_info = {"seed": args.seed, "feature": args.feature}
    out_model_dir = P.models / "baseline" / f"seed_{args.seed}" / args.feature
    rows, roc_preds = train_eval_models(
        X_train, y_train, X_val, y_val, X_test, y_test,
        out_model_dir=out_model_dir,
        run_info=run_info
    )

    out_metrics_csv = P.metrics / "baseline_classification.csv"
    append_metrics_csv(rows, out_metrics_csv)
    print("Appended metrics ->", out_metrics_csv)

    # ROC plot (Times New Roman enforced in plotting)
    out_fig = P.figures / f"roc_baselines_seed{args.seed}_{args.feature}.png"
    plot_roc_curves(roc_preds, out_fig, title=f"Baseline ROC (seed={args.seed}, feature={args.feature})")
    print("Saved ROC ->", out_fig)

if __name__ == "__main__":
    main()
