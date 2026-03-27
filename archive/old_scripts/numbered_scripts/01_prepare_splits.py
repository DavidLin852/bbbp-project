from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import json

from src.config import Paths, DatasetConfig, SplitConfig
from src.utils.io import read_tsv, write_csv
from src.utils.split import stratified_train_val_test
from src.utils.seed import seed_everything

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_groups", type=str, default="A,B")
    args = ap.parse_args()

    seed_everything(args.seed)

    P = Paths()
    D = DatasetConfig()
    S = SplitConfig()
    keep = tuple([x.strip() for x in args.keep_groups.split(",") if x.strip()])

    raw_path = P.data_raw / D.filename
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing dataset: {raw_path}")

    df = read_tsv(raw_path)

    # Filter groups
    df = df[df[D.group_col].astype(str).isin(keep)].copy()

    # Create binary label
    df = df[df[D.bbb_col].notna()].copy()
    df["y_cls"] = (df[D.bbb_col].astype(str).str.strip() == "BBB+").astype(int)

    # Basic cleanup
    df = df[df[D.smiles_col].notna()].copy()
    df[D.smiles_col] = df[D.smiles_col].astype(str).str.strip()

    # Keep a stable ID column
    if "NO." in df.columns:
        df["row_id"] = df["NO."]
    else:
        df["row_id"] = range(len(df))

    # Stratified split by y_cls
    split = stratified_train_val_test(
        df=df,
        label_col="y_cls",
        train_ratio=S.train_ratio,
        val_ratio=S.val_ratio,
        test_ratio=S.test_ratio,
        seed=args.seed
    )

    out_dir = P.data_splits / f"seed_{args.seed}"
    write_csv(split.train, out_dir / "train.csv")
    write_csv(split.val, out_dir / "val.csv")
    write_csv(split.test, out_dir / "test.csv")

    # small report
    def _stat(x: pd.DataFrame):
        return {
            "n": len(x),
            "pos_rate": float(x["y_cls"].mean()),
            "group_counts": x[D.group_col].value_counts(dropna=False).to_dict()
        }

    report = {"seed": args.seed, "keep_groups": keep, "train": _stat(split.train), "val": _stat(split.val), "test": _stat(split.test)}
    (out_dir / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved splits to:", out_dir)
    print(report)

if __name__ == "__main__":
    main()
