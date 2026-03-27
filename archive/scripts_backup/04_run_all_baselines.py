"""
Comprehensive Ablation Study: All Features × All Models

系统性地测试所有特征类型与模型组合，找出最优配置。

Features:
  - morgan:     Morgan fingerprint (2048 bits)
  - maccs:      MACCS keys (167 bits)
  - atompairs:  Atom pairs (1024 bits)
  - fp2:        FP2 fingerprint (2048 bits)
  - desc13:     Basic descriptors (13)
  - desc208:    Extended descriptors (200+)
  - all_fp:     All fingerprints combined (5579)
  - combined:   All features (5778)

Models:
  - RF:         Random Forest
  - ETC:        Extra Trees
  - XGB:        XGBoost
  - LGBM:       LightGBM
  - GB:         Gradient Boosting
  - ADA:        AdaBoost
  - SVM_RBF:    SVM with RBF kernel
  - SVM_LINEAR: SVM with linear kernel
  - KNN3/5/7:   K-Nearest Neighbors
  - NB_Gaussian: Gaussian Naive Bayes
  - NB_Bernoulli: Bernoulli Naive Bayes
  - LR:         Logistic Regression
  - MLP:        Multi-Layer Perceptron
  - MLP_Small:  Smaller MLP

Total combinations: 8 features × 17 models = 136 experiments

Usage:
    # Full ablation study
    python scripts/04_run_all_baselines.py --seed 0 --full

    # Specific feature
    python scripts/04_run_all_baselines.py --seed 0 --feature morgan --models rf,xgb,lgbm
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy import sparse

from src.config import Paths, DatasetConfig
from src.featurize.rdkit_descriptors import get_descriptor_names
from src.baseline.train_baselines import (
    train_eval_all_models,
    MODEL_CONFIGS,
    ALL_MODELS,
    TRADITIONAL_MODELS,
    QUICK_MODELS,
    SVM_MODELS,
    KNN_MODELS,
    NB_MODELS,
    NN_MODELS,
    ENSEMBLE_MODELS
)
from src.utils.io import write_csv


# =============================================================================
# Feature Definitions
# =============================================================================

FEATURE_CONFIG = {
    "morgan": {
        "file": "morgan_2048.npz",
        "dims": 2048,
        "type": "fingerprint",
        "sparse": True
    },
    "maccs": {
        "file": "maccs.npz",
        "dims": 167,
        "type": "fingerprint",
        "sparse": True
    },
    "atompairs": {
        "file": "atompairs.npz",
        "dims": 1024,
        "type": "fingerprint",
        "sparse": True
    },
    "fp2": {
        "file": "fp2.npz",
        "dims": 2048,
        "type": "fingerprint",
        "sparse": True
    },
    "desc13": {
        "file": "descriptors.csv",
        "dims": 13,
        "type": "descriptor",
        "sparse": False,
        "from": "enhanced",  # look in seed_X_enhanced
        "version": "basic"
    },
    "desc208": {
        "file": "descriptors.csv",
        "dims": 208,
        "type": "descriptor",
        "sparse": False,
        "from": "enhanced",
        "version": "all"
    },
    "all_fp": {
        "file": "all_fingerprints.npz",
        "dims": 5579,  # morgan(2048) + maccs(167) + atompairs(1024) + fp2(2048)
        "type": "combined",
        "sparse": True
    },
    "combined": {
        "file": "combined_all.npz",
        "dims": 5287,  # morgan(2048) + maccs(167) + atompairs(1024) + fp2(2048) + desc(98)
        "type": "combined",
        "sparse": True
    }
}


def load_features(seed: int, feature_name: str, feat_dir: Path = None):
    """Load features based on name."""
    if feat_dir is None:
        P = Paths()
        feat_dir = P.features / f"seed_{seed}_enhanced"

    config = FEATURE_CONFIG[feature_name]
    file = config["file"]

    # Try enhanced directory first, then regular
    paths_to_try = [
        feat_dir / file,
        P.features / f"seed_{seed}" / file
    ]

    X = None
    for p in paths_to_try:
        if p.exists():
            if config["sparse"]:
                X = sparse.load_npz(p)
            else:
                X = pd.read_csv(p)
            break

    if X is None:
        raise FileNotFoundError(
            f"Feature file not found for {feature_name}. "
            f"Tried: {[str(p) for p in paths_to_try]}"
        )

    return X


def create_feature_combination(seed: int, feature_names: list[str]):
    """Create combined feature matrix from multiple features."""
    P = Paths()
    feat_dir = P.features / f"seed_{seed}_enhanced"

    matrices = []
    for name in feature_names:
        config = FEATURE_CONFIG[name]
        file = config["file"]
        p = feat_dir / file

        if config["sparse"]:
            mat = sparse.load_npz(p)
        else:
            mat = sparse.csr_matrix(pd.read_csv(p).values)
        matrices.append(mat)

    return sparse.hstack(matrices)


def load_labels(seed: int):
    """Load labels from enhanced feature meta file."""
    P = Paths()
    feat_dir = P.features / f"seed_{seed}_enhanced"
    meta = pd.read_csv(feat_dir / "meta.csv")
    return meta["y_cls"].values


def run_ablation_experiment(
    seed: int,
    feature_name: str,
    model_list: list[str],
    output_dir: Path
):
    """Run ablation experiment for a single feature-model combination."""
    print(f"\n{'='*60}")
    print(f"Experiment: Feature={feature_name}, Models={len(model_list)}")
    print(f"{'='*60}")

    start_time = time.time()

    # Load features
    print(f"Loading features: {feature_name}...")
    if "+" in feature_name:
        # Handle feature combinations
        features = feature_name.split("+")
        X = create_feature_combination(seed, features)
    else:
        X = load_features(seed, feature_name)

    # Load meta to get proper row indices
    P = Paths()
    feat_dir = P.features / f"seed_{seed}_enhanced"
    meta = pd.read_csv(feat_dir / "meta.csv")

    # Create row_id to new_index mapping
    row_id_to_idx = {row_id: idx for idx, row_id in enumerate(meta["row_id"].values)}

    # Get split dataframes
    split_dir = P.data_splits / f"seed_{seed}"
    df_train = pd.read_csv(split_dir / "train.csv")
    df_val = pd.read_csv(split_dir / "val.csv")
    df_test = pd.read_csv(split_dir / "test.csv")

    # Map row_id to sequential indices
    train_idx = df_train["row_id"].map(row_id_to_idx).values
    val_idx = df_val["row_id"].map(row_id_to_idx).values
    test_idx = df_test["row_id"].map(row_id_to_idx).values

    # Filter out any NaN values (samples not found in features)
    train_idx = train_idx[~np.isnan(train_idx)].astype(int)
    val_idx = val_idx[~np.isnan(val_idx)].astype(int)
    test_idx = test_idx[~np.isnan(test_idx)].astype(int)

    # Get labels from meta
    y = meta["y_cls"].values

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # Convert sparse to dense if needed for certain models
    needs_dense = any(
        MODEL_CONFIGS[m.upper()].get("needs_scaling", False) and FEATURE_CONFIG.get(feature_name, {"sparse": True}).get("sparse", True)
        for m in model_list
    )
    if needs_dense and hasattr(X_train, 'toarray'):
        X_train = X_train.toarray().astype(np.float32)
        X_val = X_val.toarray().astype(np.float32)
        X_test = X_test.toarray().astype(np.float32)

    print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # Create model directory
    model_dir = output_dir / "models" / f"seed_{seed}" / feature_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Run info
    run_info = {
        "seed": seed,
        "feature": feature_name,
        "feature_dims": X_train.shape[1],
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0]
    }

    # Train all models
    rows, preds_for_roc = train_eval_all_models(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        model_dir,
        run_info,
        model_list
    )

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds")

    return rows, preds_for_roc


def create_ablation_summary(all_results: list[dict], output_dir: Path):
    """Create summary table of all ablation results."""
    df = pd.DataFrame(all_results)

    # Sort by AUC descending
    df = df.sort_values("auc", ascending=False)

    # Save full results
    df.to_csv(output_dir / "ablation_all_results.csv", index=False)

    # Create summary by feature
    feature_summary = df.groupby("feature").agg({
        "auc": ["mean", "std", "max"],
        "f1": ["mean", "max"],
        "mcc": ["mean", "max"]
    }).round(4)
    feature_summary.columns = ["_".join(col) for col in feature_summary.columns]
    feature_summary = feature_summary.reset_index()
    feature_summary.to_csv(output_dir / "ablation_feature_summary.csv", index=False)

    # Create summary by model
    model_summary = df.groupby("model").agg({
        "auc": ["mean", "std", "max"],
        "f1": ["mean", "max"],
        "mcc": ["mean", "max"]
    }).round(4)
    model_summary.columns = ["_".join(col) for col in model_summary.columns]
    model_summary = model_summary.reset_index()
    model_summary.to_csv(output_dir / "ablation_model_summary.csv", index=False)

    # Create feature-model heatmap data
    pivot_auc = df.pivot_table(values="auc", index="model", columns="feature", aggfunc="max")
    pivot_auc.to_csv(output_dir / "ablation_heatmap_auc.csv")

    # Find best combinations
    best_combos = df.groupby("feature").apply(
        lambda x: x.loc[x["auc"].idxmax()]
    ).reset_index(drop=True)
    best_combos = best_combos.sort_values("auc", ascending=False)

    print("\n" + "="*60)
    print("TOP 10 FEATURE-MODEL COMBINATIONS")
    print("="*60)
    for i, row in best_combos.head(10).iterrows():
        print(f"{row['feature']:12s} + {row['model']:10s}: AUC={row['auc']:.4f}, F1={row['f1']:.4f}, MCC={row['mcc']:.4f}")

    best_combos.to_csv(output_dir / "ablation_best_combinations.csv", index=False)

    return df


def main():
    ap = argparse.ArgumentParser(description="Comprehensive Ablation Study")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature", type=str, default="all",
                    help="Feature: morgan, maccs, atompairs, fp2, desc13, desc208, all_fp, combined, all")
    ap.add_argument("--models", type=str, default="all",
                    help="Models: all, traditional, quick, or comma-separated list")
    ap.add_argument("--output", type=str, default=None,
                    help="Output directory suffix")
    ap.add_argument("--quick", action="store_true",
                    help="Quick test with subset of models")
    ap.add_argument("--full", action="store_true",
                    help="Run full ablation (all features × all models)")

    args = ap.parse_args()

    P = Paths()
    D = DatasetConfig()

    # Output directory
    if args.output:
        output_dir = P.artifacts / "ablation" / args.output
    else:
        output_dir = P.artifacts / "ablation" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("COMPREHENSIVE ABLATION STUDY")
    print(f"{'='*60}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")

    # Parse models
    if args.models == "all":
        model_list = ALL_MODELS
    elif args.models == "traditional":
        model_list = TRADITIONAL_MODELS
    elif args.models == "quick":
        model_list = QUICK_MODELS
    else:
        model_list = [m.strip() for m in args.models.split(",")]

    print(f"Models ({len(model_list)}): {model_list}")

    # Parse features
    if args.feature == "all":
        feature_list = list(FEATURE_CONFIG.keys())
    else:
        feature_list = [args.feature]

    # Add combined features if --full flag is used
    if args.full:
        extra_features = ["morgan+desc208", "maccs+desc208", "all_fp", "all_fp+desc208"]
        for extra in extra_features:
            if extra not in feature_list:
                feature_list.append(extra)

    # Update dims for combined features
    FEATURE_CONFIG["morgan+desc208"] = {"dims": 2048+98, "type": "combined", "sparse": True}
    FEATURE_CONFIG["maccs+desc208"] = {"dims": 167+98, "type": "combined", "sparse": True}
    FEATURE_CONFIG["all_fp"] = {"dims": 5287, "type": "combined", "sparse": True, "file": "combined_all.npz"}
    FEATURE_CONFIG["all_fp+desc208"] = {"dims": 5287, "type": "combined", "sparse": True, "file": "combined_all.npz"}

    print(f"Features ({len(feature_list)}): {feature_list}")

    all_results = []
    total_experiments = len(feature_list) * len(model_list)
    completed = 0

    start_time = time.time()

    for feature_name in feature_list:
        if feature_name not in FEATURE_CONFIG and "+" not in feature_name:
            print(f"\nSkipping unknown feature: {feature_name}")
            continue

        # Filter models that work with this feature
        # SVM and KNN need scaling, which requires dense input
        # For very high-dim sparse features, use subset of models
        feature_config = FEATURE_CONFIG.get(feature_name, {"dims": 0, "type": "fingerprint"})
        n_dims = feature_config.get("dims", 2048)

        if n_dims > 5000:
            # For high-dim fingerprints, skip SVM/KNN (slow)
            filtered_models = [m for m in model_list if m.upper() not in SVM_MODELS + KNN_MODELS]
        else:
            filtered_models = model_list

        rows, _ = run_ablation_experiment(
            seed=args.seed,
            feature_name=feature_name,
            model_list=filtered_models,
            output_dir=output_dir
        )
        all_results.extend(rows)
        completed += len(filtered_models)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"{'='*60}")
    print(f"Total experiments: {completed}")
    print(f"Total time: {elapsed/60:.1f} minutes")

    # Create summary
    if all_results:
        summary_df = create_ablation_summary(all_results, output_dir)

        # Save run info
        run_info = {
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "features_tested": feature_list,
            "models_tested": model_list,
            "total_experiments": completed,
            "total_time_minutes": elapsed / 60,
            "best_model": summary_df.iloc[0]["model"] if len(summary_df) > 0 else None,
            "best_feature": summary_df.iloc[0]["feature"] if len(summary_df) > 0 else None,
            "best_auc": float(summary_df.iloc[0]["auc"]) if len(summary_df) > 0 else None
        }

        with open(output_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
