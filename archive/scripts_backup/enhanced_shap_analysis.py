"""
Enhanced SHAP Analysis Script

对训练好的模型进行SHAP可解释性分析：
- 计算SHAP值
- 生成特征重要性排名
- 识别关键分子子结构
- 可视化分析结果

Usage:
    # XGBoost模型
    python scripts/enhanced_shap_analysis.py --seed 0 --model xgb --features combined

    # 多模型对比
    python scripts/enhanced_shap_analysis.py --seed 0 --model all --features morgan

    # 毒性基团分析
    python scripts/enhanced_shap_analysis.py --seed 0 --model xgb --features combined --toxicophores
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from src.config import Paths, DatasetConfig
from src.featurize.rdkit_descriptors import get_descriptor_names


def load_features(seed: int, feature_type: str, feat_dir: Path = None):
    """Load features based on type."""
    P = Paths()
    if feat_dir is None:
        feat_dir = P.features / f"seed_{seed}_enhanced"

    if feature_type == "combined":
        X = sparse.load_npz(feat_dir / "combined_all.npz")
    elif feature_type == "morgan":
        X = sparse.load_npz(feat_dir / "morgan_2048.npz")
    elif feature_type == "descriptors":
        X = pd.read_csv(feat_dir / "descriptors.csv")
        X = sparse.csr_matrix(X.values)
    else:
        fp_file = feat_dir / f"{feature_type}.npz"
        X = sparse.load_npz(fp_file)

    return X


def load_labels(seed: int):
    """Load labels from split files."""
    P = Paths()
    split_dir = P.data_splits / f"seed_{seed}"

    df = pd.concat([
        pd.read_csv(split_dir / "train.csv"),
        pd.read_csv(split_dir / "val.csv"),
        pd.read_csv(split_dir / "test.csv")
    ], ignore_index=True)

    return df["y_cls"].values


def load_model(seed: int, model_name: str, feat_type: str = "combined"):
    """Load trained model."""
    P = Paths()
    model_dir = P.models / f"seed_{seed}_enhanced" / feat_type
    model_path = model_dir / f"{model_name.upper()}_seed{seed}.joblib"
    return joblib.load(model_path)


def run_shap_analysis(args):
    """Run SHAP analysis for specified model(s)."""
    P = Paths()
    output_dir = P.artifacts / "shap" / f"seed_{args.seed}" / args.features
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    feat_dir = P.features / f"seed_{args.seed}_enhanced"

    if args.features == "combined":
        X_full = sparse.load_npz(feat_dir / "combined_all.npz")
        # Create feature names
        feature_names = []
        feature_names.extend([f"morgan_{i}" for i in range(2048)])
        feature_names.extend([f"maccs_{i}" for i in range(167)])
        feature_names.extend([f"atompairs_{i}" for i in range(1024)])
        feature_names.extend([f"fp2_{i}" for i in range(2048)])
        feature_names.extend(get_descriptor_names("all"))
    elif args.features == "morgan":
        X_full = sparse.load_npz(feat_dir / "morgan_2048.npz")
        feature_names = [f"morgan_{i}" for i in range(2048)]
    elif args.features == "descriptors":
        X_full = pd.read_csv(feat_dir / "descriptors.csv")
        feature_names = list(X_full.columns)
    else:
        X_full = sparse.load_npz(feat_dir / f"{args.features}.npz")
        n_features = X_full.shape[1]
        feature_names = [f"{args.features}_{i}" for i in range(n_features)]

    # Subsample for SHAP analysis
    n_samples = min(args.n_samples, X_full.shape[0])
    X_sample = X_full[:n_samples]
    if hasattr(X_sample, 'toarray'):
        X_sample = X_sample.toarray()

    # Load labels
    y = load_labels(args.seed)
    y_sample = y[:n_samples]

    # Determine models to analyze
    if args.model == "all":
        models_to_analyze = ["rf", "xgb", "lgbm"]
    else:
        models_to_analyze = [args.model]

    print(f"\n{'='*60}")
    print("SHAP Analysis")
    print(f"{'='*60}")
    print(f"Seed: {args.seed}")
    print(f"Features: {args.features}")
    print(f"Samples: {n_samples}")
    print(f"Features per sample: {X_sample.shape[1]}")
    print(f"Models: {models_to_analyze}")
    print(f"Output dir: {output_dir}")

    all_results = []

    for model_name in models_to_analyze:
        print(f"\n{'='*40}")
        print(f"Analyzing {model_name.upper()}...")
        print(f"{'='*40}")

        try:
            model_path = P.models / f"seed_{args.seed}_enhanced" / args.features / f"{model_name.upper()}_seed{args.seed}.joblib"
            model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"  Model not found: {model_path}")
            continue

        # Determine model type
        if model_name == "rf":
            model_type = "rf"
        elif model_name == "xgb":
            model_type = "xgb"
        elif model_name == "lgbm":
            model_type = "lgbm"
        else:
            model_type = "rf"

        # Run SHAP analysis
        from src.explain.shap_analysis import SHAPExplainer, ModelType, explain_model

        explainer, importance = explain_model(
            model, getattr(ModelType, model_type.upper()),
            X_sample, feature_names
        )

        # Save importance
        importance_path = output_dir / f"{model_name}_importance.csv"
        importance.to_csv(importance_path, index=False)
        print(f"  Importance saved: {importance_path}")

        # Plot summary
        if args.plot:
            from src.explain.shap_analysis import SHAPConfig

            config = SHAPConfig()
            explainer.plot_summary(
                X_sample,
                output_dir / f"{model_name}_shap_summary.png",
                max_features=args.max_features
            )
            print(f"  Summary plot saved")

            explainer.plot_feature_importance(
                X_sample,
                output_dir / f"{model_name}_feature_importance.png",
                top_n=args.max_features
            )
            print(f"  Importance plot saved")

        # Save top features
        top_n = min(30, len(importance))
        top_features = importance.head(top_n).to_dict("records")
        all_results.append({
            "model": model_name,
            "top_features": top_features
        })

    # Save combined results
    results_path = output_dir / "shap_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Toxicophore analysis
    if args.toxicophores:
        print(f"\n{'='*40}")
        print("Toxicophore Analysis")
        print(f"{'='*40}")

        from src.explain.shap_analysis import identify_toxicophores_from_smarts, COMMON_TOXICOPHORES

        toxicophores = identify_toxicophores_from_smarts(
            explainer.shap_values,
            COMMON_TOXICOPHORES,
            feature_names
        )

        tox_path = output_dir / "toxicophores.csv"
        toxicophores.to_csv(tox_path, index=False)
        print(f"Toxicophores saved: {tox_path}")

        print("\nTop Toxicophores:")
        for i, row in toxicophores.head(10).iterrows():
            print(f"  {row['smarts_pattern']:20s}: {row['mean_shap']:+.4f} ({row['typical_impact']})")

    print(f"\n{'='*60}")
    print("SHAP Analysis Complete!")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser(description="SHAP Explainability Analysis")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", type=str, default="xgb",
                    choices=["rf", "xgb", "lgbm", "all"],
                    help="Model to analyze")
    ap.add_argument("--features", type=str, default="combined",
                    choices=["morgan", "maccs", "atompairs", "fp2", "descriptors", "combined"],
                    help="Feature type")
    ap.add_argument("--n_samples", type=int, default=500,
                    help="Number of samples for SHAP calculation")
    ap.add_argument("--max_features", type=int, default=20,
                    help="Max features to display in plots")
    ap.add_argument("--plot", action="store_true", default=True,
                    help="Generate plots")
    ap.add_argument("--toxicophores", action="store_true",
                    help="Run toxicophore analysis")

    args = ap.parse_args()

    from scipy import sparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    run_shap_analysis(args)


if __name__ == "__main__":
    main()
