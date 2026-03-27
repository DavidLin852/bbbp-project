"""
Run Missing Experiments for Simplified Model Set
- SVM_RBF: morgan, atompairs, fp2
- KNN5: morgan, atompairs
- NB_Bernoulli: morgan, atompairs, fp2, rdkit_desc
"""

import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse

# UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline.train_baselines import MODEL_CONFIGS, train_single_model


def load_features(feature_name):
    """Load features"""
    feat_dir = PROJECT_ROOT / "artifacts" / "features" / "seed_0_enhanced"

    # Feature file mapping
    feat_files = {
        "morgan": "morgan.npz",
        "maccs": "maccs.npz",
        "atompairs": "atompairs.npz",
        "fp2": "fp2.npz",
        "rdkit_desc": "descriptors.npz"
    }

    feat_file = feat_dir / feat_files[feature_name]
    data = np.load(feat_file)

    if 'format' in data.files and 'data' in data.files:
        X = sparse.csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
    else:
        X = data['X'] if 'X' in data.files else data[data.files[0]]

    # Load meta mapping
    meta = pd.read_csv(feat_dir / "meta.csv")
    row_id_to_idx = {row_id: idx for idx, row_id in enumerate(meta["row_id"].values)}

    return X, row_id_to_idx


def run_missing_experiments():
    """Run all missing experiments"""

    results_dir = PROJECT_ROOT / "artifacts" / "ablation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data splits
    split_dir = PROJECT_ROOT / "data" / "splits" / "seed_0"
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    # Missing experiments
    missing_experiments = [
        ("SVM_RBF", "morgan"),
        ("SVM_RBF", "atompairs"),
        ("SVM_RBF", "fp2"),
        ("KNN5", "morgan"),
        ("KNN5", "atompairs"),
        ("NB_Bernoulli", "morgan"),
        ("NB_Bernoulli", "atompairs"),
        ("NB_Bernoulli", "fp2"),
        ("NB_Bernoulli", "rdkit_desc"),
    ]

    all_results = []

    print("=" * 80)
    print("Running Missing Experiments")
    print("=" * 80)
    print(f"Total experiments: {len(missing_experiments)}")
    print("=" * 80)

    for model_name, feature_name in missing_experiments:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}, Feature: {feature_name}")
        print(f"{'='*60}")

        try:
            # Load features
            X_all, row_id_to_idx = load_features(feature_name)

            # Get indices
            train_idx = train_df["row_id"].map(row_id_to_idx).values
            val_idx = val_df["row_id"].map(row_id_to_idx).values
            test_idx = test_df["row_id"].map(row_id_to_idx).values

            X_train = X_all[train_idx]
            X_val = X_all[val_idx]
            X_test = X_all[test_idx]

            y_train = train_df["y_cls"].values
            y_val = val_df["y_cls"].values
            y_test = test_df["y_cls"].values

            print(f"X_train shape: {X_train.shape}")

            # Train model
            result = train_single_model(
                model_name=model_name,
                model_config=MODEL_CONFIGS[model_name],
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                out_model_dir=results_dir / f"{model_name}_{feature_name}",
                seed=0
            )

            if result is not None:
                result['feature'] = feature_name
                all_results.append(result)
                print(f"  [OK] AUC={result['auc']:.4f}, F1={result['f1']:.4f}, MCC={result['mcc']:.4f}")
            else:
                print(f"  [FAIL] Training failed")

        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        output_file = results_dir / "MISSING_EXPERIMENTS.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved: {output_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("Missing Experiments Summary")
        print("=" * 80)
        for _, row in df.iterrows():
            print(f"  {row['model']:15} + {row['feature']:12} → AUC={row['auc']:.4f}")

        return df
    else:
        print("\n❌ No results collected")
        return None


def update_complete_matrix():
    """Update COMPLETE_MATRIX_RESULTS.csv to include missing experiments"""

    results_dir = PROJECT_ROOT / "artifacts" / "ablation"

    # Load existing results
    df_existing = pd.read_csv(results_dir / "COMPLETE_MATRIX_RESULTS.csv")

    # Load missing results
    df_missing = pd.read_csv(results_dir / "MISSING_EXPERIMENTS.csv")

    # Combine
    df_combined = pd.concat([df_existing, df_missing], ignore_index=True)

    # Remove duplicates (keep last)
    df_combined = df_combined.drop_duplicates(
        subset=['model', 'feature', 'split'],
        keep='last'
    )

    # Sort by AUC
    df_combined = df_combined.sort_values('auc', ascending=False).reset_index(drop=True)

    # Save
    output_file = results_dir / "COMPLETE_MATRIX_RESULTS.csv"
    df_combined.to_csv(output_file, index=False)

    print(f"\n✅ Updated COMPLETE_MATRIX_RESULTS.csv")
    print(f"   Total records: {len(df_combined)}")

    return df_combined


if __name__ == "__main__":
    # Run missing experiments
    run_missing_experiments()

    # Update combined results
    update_complete_matrix()

    print("\n" + "=" * 80)
    print("All missing experiments completed!")
    print("=" * 80)
