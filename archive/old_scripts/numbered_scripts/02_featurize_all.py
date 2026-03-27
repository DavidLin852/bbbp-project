from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

from src.config import Paths, DatasetConfig, FeaturizeConfig
from src.utils.io import write_csv
from src.featurize.rdkit_descriptors import compute_descriptors
from src.featurize.fingerprints import morgan_fp_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    P = Paths()
    D = DatasetConfig()
    F = FeaturizeConfig()

    split_dir = P.data_splits / f"seed_{args.seed}"
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"
    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}. Run scripts/01_prepare_splits.py first.")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    df_all = pd.concat([df_train.assign(split="train"), df_val.assign(split="val"), df_test.assign(split="test")], ignore_index=True)
    smiles = df_all[D.smiles_col].astype(str).tolist()

    feat_dir = P.features / f"seed_{args.seed}"
    feat_dir.mkdir(parents=True, exist_ok=True)

    # descriptors
    desc = compute_descriptors(smiles)
    write_csv(desc, feat_dir / "descriptors.csv")

    # fingerprints (sparse)
    X_fp = morgan_fp_matrix(smiles, radius=F.morgan_radius, n_bits=F.morgan_bits)
    sparse.save_npz(feat_dir / f"morgan_{F.morgan_bits}.npz", X_fp)

    # meta for alignment
    meta = df_all[[D.smiles_col, "y_cls", "split", "row_id"]].copy()
    write_csv(meta, feat_dir / "meta.csv")

    print("Saved features to:", feat_dir)

if __name__ == "__main__":
    main()
