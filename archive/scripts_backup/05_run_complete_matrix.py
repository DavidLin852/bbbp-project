"""
Complete Model-Feature Matrix Experiment
Run all models on all features for comprehensive comparison
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from joblib import dump, load
from scipy import sparse

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline.train_baselines import MODEL_CONFIGS, train_single_model
from src.utils.metrics import ClsMetrics

# MCC calculation function
def calculate_mcc(tp, tn, fp, fn):
    """Calculate Matthews Correlation Coefficient"""
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return numerator / denominator

# =============================================================================
# Configuration
# =============================================================================

FEATURE_CONFIG = {
    "morgan": {
        "file": "morgan.npz",
        "dims": 2048,
        "sparse": True,
        "display": "Morgan (2048D)"
    },
    "maccs": {
        "file": "maccs.npz",
        "dims": 167,
        "sparse": True,
        "display": "MACCS (167D)"
    },
    "atompairs": {
        "file": "atompairs.npz",
        "dims": 1024,
        "sparse": True,
        "display": "AtomPairs (1024D)"
    },
    "fp2": {
        "file": "fp2.npz",
        "dims": 2048,
        "sparse": True,
        "display": "FP2 (2048D)"
    },
    "rdkit_desc": {
        "file": "descriptors.npz",
        "dims": 98,
        "sparse": False,
        "display": "RDKitDescriptors (98D)"
    },
    "combined": {
        "file": "combined_all.npz",
        "dims": 5287,
        "sparse": True,
        "display": "Combined (5287D)"
    }
}

# Exclude models that don't work well with certain features
EXCLUDE_MATRIX = {
    # High-dimensional features (>1000D) - exclude SVM and KNN
    "morgan": ["SVM_RBF", "SVM_LINEAR", "SVM_POLY", "KNN3", "KNN5", "KNN7", "NB_Bernoulli"],
    "atompairs": ["SVM_RBF", "SVM_LINEAR", "SVM_POLY", "KNN3", "KNN5", "KNN7", "NB_Bernoulli"],
    "fp2": ["SVM_RBF", "SVM_LINEAR", "SVM_POLY", "NB_Bernoulli"],
    "combined": ["SVM_RBF", "SVM_LINEAR", "SVM_POLY", "KNN3", "KNN5", "KNN7", "NB_Bernoulli"],
    # Binary features - exclude Gaussian NB
    "maccs": ["NB_Gaussian"],
    "rdkit_desc": ["NB_Bernoulli"],  # Continuous descriptors
}

# =============================================================================
# Data Loading
# =============================================================================

def load_data(seed=0):
    """Load train/val/test splits"""
    split_dir = PROJECT_ROOT / "data" / "splits" / f"seed_{seed}"
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    return train_df, val_df, test_df


def load_features(feature_name, seed=0):
    """Load feature matrix"""
    feat_dir = PROJECT_ROOT / "artifacts" / "features" / f"seed_{seed}_enhanced"
    feat_config = FEATURE_CONFIG[feature_name]
    feat_file = feat_dir / feat_config["file"]

    data = np.load(feat_file)

    # Check if it's sparse matrix format (CSR)
    if 'format' in data.files and 'data' in data.files:
        # Reconstruct sparse matrix
        X = sparse.csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
    else:
        # Regular dense array
        X = data['X'] if 'X' in data.files else data[list(data.keys())[0]]

    # Load meta for row_id mapping
    meta = pd.read_csv(feat_dir / "meta.csv")
    row_id_to_idx = {row_id: idx for idx, row_id in enumerate(meta["row_id"].values)}

    return X, row_id_to_idx


def get_feature_indices(train_df, val_df, test_df, row_id_to_idx):
    """Map row_ids to feature indices"""
    train_idx = train_df["row_id"].map(row_id_to_idx).values
    val_idx = val_df["row_id"].map(row_id_to_idx).values
    test_idx = test_df["row_id"].map(row_id_to_idx).values

    return train_idx, val_idx, test_idx


# =============================================================================
# Experiment Running
# =============================================================================

def run_single_experiment(seed, feature_name, model_name, output_dir):
    """Run one model-feature combination"""

    try:
        print(f"\n{'='*60}")
        print(f"Running: {model_name} on {feature_name}")
        print(f"{'='*60}")

        # Load data
        train_df, val_df, test_df = load_data(seed)
        X_all, row_id_to_idx = load_features(feature_name, seed)

        # Get indices
        train_idx, val_idx, test_idx = get_feature_indices(
            train_df, val_df, test_df, row_id_to_idx
        )

        # Extract features and labels
        X_train = X_all[train_idx]
        X_val = X_all[val_idx]
        X_test = X_all[test_idx]

        y_train = train_df["y_cls"].values
        y_val = val_df["y_cls"].values
        y_test = test_df["y_cls"].values

        print(f"X_train shape: {X_train.shape}")
        print(f"y_train distribution: {np.bincount(y_train)}")

        # Check if model exists in config
        if model_name not in MODEL_CONFIGS:
            print(f"⚠️  Model {model_name} not in config, skipping")
            return None

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
            out_model_dir=output_dir,
            seed=seed
        )

        if result is None:
            print(f"❌ Failed to train {model_name} on {feature_name}")
            return None

        # Add metadata
        result['seed'] = seed
        result['feature'] = feature_name
        result['feature_dims'] = FEATURE_CONFIG[feature_name]['dims']
        result['n_train'] = len(y_train)
        result['n_test'] = len(y_test)

        print(f"✅ Completed: AUC={result['auc']:.4f}, F1={result['f1']:.4f}")

        return result

    except Exception as e:
        print(f"❌ Error running {model_name} on {feature_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_complete_matrix(seed=0, output_dir=None):
    """Run all model-feature combinations"""

    if output_dir is None:
        output_dir = PROJECT_ROOT / "artifacts" / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    summary_file = output_dir / "COMPLETE_MATRIX_RESULTS.csv"

    print("\n" + "="*80)
    print("COMPLETE MODEL-FEATURE MATRIX EXPERIMENT")
    print("="*80)
    print(f"Seed: {seed}")
    print(f"Models: {len(MODEL_CONFIGS)}")
    print(f"Features: {len(FEATURE_CONFIG)}")
    print(f"Total experiments: {len(MODEL_CONFIGS) * len(FEATURE_CONFIG)}")
    print("="*80)

    # Load existing results if any
    existing_results = []
    if summary_file.exists():
        existing_df = pd.read_csv(summary_file)
        existing_results = [
            (row['model'], row['feature'])
            for _, row in existing_df.iterrows()
        ]
        print(f"\n📊 Found {len(existing_results)} existing results")

    completed = 0
    skipped = 0
    failed = 0

    # Run all combinations
    for feature_name in FEATURE_CONFIG.keys():
        for model_name in MODEL_CONFIGS.keys():

            # Check if should be excluded
            if feature_name in EXCLUDE_MATRIX:
                if model_name in EXCLUDE_MATRIX[feature_name]:
                    print(f"⊘  Skipping {model_name} on {feature_name} (excluded)")
                    skipped += 1
                    continue

            # Check if already run
            if (model_name, feature_name) in existing_results:
                print(f"✓  Already completed: {model_name} on {feature_name}")
                completed += 1
                continue

            # Run experiment
            result = run_single_experiment(
                seed=seed,
                feature_name=feature_name,
                model_name=model_name,
                output_dir=output_dir
            )

            if result is not None:
                results.append(result)
                # Save results incrementally
                df_new = pd.DataFrame(results)
                if summary_file.exists():
                    df_old = pd.read_csv(summary_file)
                    df_all = pd.concat([df_old, df_new], ignore_index=True)
                else:
                    df_all = df_new

                df_all.to_csv(summary_file, index=False)
                results = []  # Clear to save memory
                completed += 1
            else:
                failed += 1

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"✅ Completed: {completed}")
    print(f"⊘  Skipped:   {skipped}")
    print(f"❌ Failed:    {failed}")
    print(f"📊 Total:     {completed + skipped + failed}")
    print("="*80)

    # Load final results
    if summary_file.exists():
        df_final = pd.read_csv(summary_file)
        print(f"\n📁 Results saved to: {summary_file}")
        print(f"📊 Total results: {len(df_final)}")

        # Print top models
        print("\n🏆 Top 20 Models:")
        top20 = df_final.nlargest(20, 'auc')
        for idx, row in top20.iterrows():
            print(f"  {row['model']:15} + {row['feature']:15} → AUC={row['auc']:.4f}, F1={row['f1']:.4f}")

    return summary_file


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run complete model-feature matrix")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    # Run complete matrix
    result_file = run_complete_matrix(seed=args.seed, output_dir=args.output)

    print(f"\n🎉 Complete! Results saved to: {result_file}")
