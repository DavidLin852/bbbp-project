"""
Enhanced Feature Extraction Script

支持多种分子指纹和描述符的提取与组合：
- Morgan (ECFP4, 2048 bits)
- MACCS Keys (167 bits)
- Atom Pairs (1024 bits)
- FP2 (2048 bits)
- RDKit Descriptors (200+)
- Combined features (fingerprint + descriptors)

Usage:
    python scripts/02_enhanced_featurize.py --seed 0 --features morgan,maccs,atompairs,descriptors
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import pandas as pd
from scipy import sparse

from src.config import Paths, DatasetConfig, FeaturizeConfig
from src.utils.io import write_csv
from src.featurize.fingerprints import (
    morgan_fp_matrix, maccs_keys_matrix, atom_pairs_matrix, fp2_matrix,
    combine_fingerprints
)
from src.featurize.rdkit_descriptors import (
    compute_descriptors, normalize_descriptors, get_descriptor_names
)


def save_features(X: sparse.csr_matrix, name: str, feat_dir: Path):
    """Save sparse matrix to .npz file."""
    sparse.save_npz(feat_dir / f"{name}.npz", X)
    print(f"  Saved: {name} ({X.shape[0]} samples, {X.shape[1]} features)")


def main():
    ap = argparse.ArgumentParser(description="Enhanced molecular feature extraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", type=str,
                    default="morgan,maccs,atompairs,descriptors",
                    help="Comma-separated features: morgan,maccs,atompairs,fp2,descriptors")
    ap.add_argument("--combine", action="store_true", default=True,
                    help="Combine all features into single matrix")
    ap.add_argument("--descriptor_set", type=str, default="all",
                    choices=["basic", "extended", "all"],
                    help="Descriptor set to compute")
    args = ap.parse_args()

    P = Paths()
    D = DatasetConfig()
    F = FeaturizeConfig()

    # Parse features
    feature_list = [f.strip().lower() for f in args.features.split(",")]

    # Check splits exist
    split_dir = P.data_splits / f"seed_{args.seed}"
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}. Run scripts/01_prepare_splits.py first.")

    # Load data
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    df_all = pd.concat([
        df_train.assign(split="train"),
        df_val.assign(split="val"),
        df_test.assign(split="test")
    ], ignore_index=True)

    smiles = df_all[D.smiles_col].astype(str).tolist()

    # Create output directory
    feat_dir = P.features / f"seed_{args.seed}_enhanced"
    feat_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Enhanced Feature Extraction")
    print(f"{'='*60}")
    print(f"Seed: {args.seed}")
    print(f"Features: {feature_list}")
    print(f"Total samples: {len(smiles)}")

    # Storage for features
    features = {}
    feature_info = {}

    # 1. Morgan Fingerprint
    if "morgan" in feature_list:
        print("\n[1/5] Computing Morgan fingerprints...")
        X_morgan = morgan_fp_matrix(smiles, radius=F.morgan_radius, n_bits=F.morgan_bits)
        features["morgan"] = X_morgan
        feature_info["morgan"] = {"bits": F.morgan_bits, "radius": F.morgan_radius}
        save_features(X_morgan, "morgan", feat_dir)

    # 2. MACCS Keys
    if "maccs" in feature_list:
        print("\n[2/5] Computing MACCS keys...")
        X_maccs = maccs_keys_matrix(smiles, n_bits=F.maccs_bits)
        features["maccs"] = X_maccs
        feature_info["maccs"] = {"bits": F.maccs_bits}
        save_features(X_maccs, "maccs", feat_dir)

    # 3. Atom Pairs
    if "atompairs" in feature_list:
        print("\n[3/5] Computing Atom Pairs fingerprints...")
        X_atompairs = atom_pairs_matrix(
            smiles, n_bits=F.atom_pairs_bits, max_distance=F.atom_pairs_max_dist
        )
        features["atompairs"] = X_atompairs
        feature_info["atompairs"] = {"bits": F.atom_pairs_bits, "max_dist": F.atom_pairs_max_dist}
        save_features(X_atompairs, "atompairs", feat_dir)

    # 4. FP2
    if "fp2" in feature_list:
        print("\n[4/5] Computing FP2 fingerprints...")
        X_fp2 = fp2_matrix(smiles, n_bits=F.fp2_bits)
        features["fp2"] = X_fp2
        feature_info["fp2"] = {"bits": F.fp2_bits}
        save_features(X_fp2, "fp2", feat_dir)

    # 5. RDKit Descriptors
    if "descriptors" in feature_list:
        print("\n[5/5] Computing RDKit descriptors...")
        desc_df = compute_descriptors(smiles, descriptor_set=args.descriptor_set)
        desc_names = list(desc_df.columns)

        # Normalize
        desc_norm, scaler = normalize_descriptors(desc_df)

        # Save
        desc_norm.to_csv(feat_dir / "descriptors.csv", index=False)
        print(f"  Saved: descriptors ({desc_norm.shape[0]} samples, {desc_norm.shape[1]} features)")

        # Save as dense matrix too
        desc_matrix = sparse.csr_matrix(desc_norm.values)
        save_features(desc_matrix, "descriptors", feat_dir)

        feature_info["descriptors"] = {
            "count": len(desc_names),
            "set": args.descriptor_set,
            "names": desc_names[:20]  # First 20 names for reference
        }

    # 6. Combine features
    if args.combine and len(features) > 1:
        print("\n[Combining] Creating combined feature matrix...")

        # Order: morgan + maccs + atompairs + fp2 + descriptors
        combined = None
        for fp_type in ["morgan", "maccs", "atompairs", "fp2"]:
            if fp_type in features:
                combined = features[fp_type] if combined is None else sparse.hstack([combined, features[fp_type]])

        if "descriptors" in features:
            # Convert descriptors to sparse if needed
            desc_sparse = sparse.csr_matrix(features["descriptors"])
            combined = sparse.hstack([combined, desc_sparse])

        total_dims = combined.shape[1]
        save_features(combined, "combined_all", feat_dir)

        # Feature count summary
        print(f"\n{'='*60}")
        print("Feature Summary:")
        print(f"{'='*60}")
        total = 0
        for name, mat in features.items():
            print(f"  {name:15s}: {mat.shape[1]:5d} dims")
            total += mat.shape[1]
        print(f"  {'combined':15s}: {total_dims:5d} dims")
        print(f"  {'total':15s}: {total:5d} dims")

        feature_info["combined"] = {
            "total_dims": total_dims,
            "feature_order": list(features.keys())
        }

    # Save feature info
    with open(feat_dir / "feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)
    print(f"\nFeature info saved to: {feat_dir / 'feature_info.json'}")

    # Save metadata
    meta = df_all[[D.smiles_col, "y_cls", "split", "row_id"]].copy()
    write_csv(meta, feat_dir / "meta.csv")

    print(f"\n{'='*60}")
    print(f"Enhanced features saved to: {feat_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
