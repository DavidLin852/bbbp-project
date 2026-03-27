"""
Train mechanism prediction models using Cornelissen et al. 2022 data.

This script trains separate models for each transport mechanism:
- BBB (Blood-Brain Barrier permeability)
- Influx (Active transport into brain)
- Efflux (Active transport out of brain / P-gp substrates)
- PAMPA (Parallel Artificial Membrane Permeability Assay)
- CNS (Central Nervous System activity)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Paths
from xgboost import XGBClassifier


def load_feature_info(data_dir):
    """Load feature information."""
    info_file = data_dir / "feature_info.json"
    with open(info_file, 'r') as f:
        return json.load(f)


def load_data(data_dir):
    """Load processed data."""
    data_file = data_dir / "cornelissen_2022_processed.csv"
    return pd.read_csv(data_file)


def get_feature_columns(df, feature_info):
    """Get feature column names."""
    physicochemical = feature_info['physicochemical']
    maccs = feature_info['maccs']
    morgan = feature_info['morgan']

    # Check which columns exist in dataframe
    physicochemical = [c for c in physicochemical if c in df.columns]
    maccs = [c for c in maccs if c in df.columns]
    morgan = [c for c in morgan if c in df.columns]

    return {
        'physicochemical': physicochemical,
        'maccs': maccs,
        'morgan': morgan,
    }


def prepare_data_for_mechanism(df, mechanism, feature_cols, split_col):
    """Prepare data for a specific mechanism."""
    label_col = f'label_{mechanism}'

    # Get samples with labels
    mask = df[label_col].notna()
    df_labeled = df[mask].copy()

    # Apply train/test split if available
    if split_col in df.columns:
        train_mask = df_labeled[split_col] == 'Train'
        test_mask = df_labeled[split_col] == 'Test'

        # If split exists, use it
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            df_train = df_labeled[train_mask]
            df_test = df_labeled[test_mask]
        else:
            # Fallback to random split
            df_train, df_test = train_test_split(df_labeled, test_size=0.2, random_state=42, stratify=df_labeled[label_col])
    else:
        # Random split
        df_train, df_test = train_test_split(df_labeled, test_size=0.2, random_state=42, stratify=df_labeled[label_col])

    # Extract features
    X_train = df_train[feature_cols].values
    y_train = df_train[label_col].values.astype(int)

    X_test = df_test[feature_cols].values
    y_test = df_test[label_col].values.astype(int)

    return X_train, X_test, y_train, y_test, df_train, df_test


def train_model(X_train, y_train, model_type='xgboost'):
    """Train a model."""
    if model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        'auc': roc_auc_score(y_test, y_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return metrics, cm, y_proba


def get_feature_importance(model, feature_names, top_n=20):
    """Get top N important features."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return []

    # Get top features
    indices = np.argsort(importances)[::-1][:top_n]

    return [(feature_names[i], importances[i]) for i in indices]


def main():
    """Main training function."""
    print("="*80)
    print("Training Mechanism Prediction Models")
    print("Using Cornelissen et al. 2022 Dataset")
    print("="*80)

    # Setup paths
    data_dir = Paths.root / "data" / "transport_mechanisms" / "cornelissen_2022"
    output_dir = Paths.artifacts / "models" / "cornelissen_2022"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n1. Loading data...")
    df = load_data(data_dir)
    feature_info = load_feature_info(data_dir)
    feature_cols = get_feature_columns(df, feature_info)

    # Combine all features
    all_features = (feature_cols['physicochemical'] +
                   feature_cols['maccs'] +
                   feature_cols['morgan'])

    print(f"   Total samples: {len(df)}")
    print(f"   Feature dimensions: {len(all_features)}")
    print(f"   - Physicochemical: {len(feature_cols['physicochemical'])}")
    print(f"   - MACCS: {len(feature_cols['maccs'])}")
    print(f"   - Morgan: {len(feature_cols['morgan'])}")

    # Mechanisms to train
    mechanisms = ['Influx', 'Efflux', 'PAMPA', 'BBB', 'CNS']

    # Store results
    all_results = {}

    # Train models for each mechanism
    for mechanism in mechanisms:
        print(f"\n{'='*80}")
        print(f"Training {mechanism} Model")
        print(f"{'='*80}")

        # Prepare data
        split_col = f'split_{mechanism}'
        try:
            X_train, X_test, y_train, y_test, df_train, df_test = prepare_data_for_mechanism(
                df, mechanism, all_features, split_col
            )
        except Exception as e:
            print(f"   Error preparing data for {mechanism}: {e}")
            continue

        print(f"   Train samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Positive rate (train): {y_train.mean()*100:.1f}%")
        print(f"   Positive rate (test): {y_test.mean()*100:.1f}%")

        # Skip if too few samples
        if len(X_train) < 50 or len(X_test) < 20:
            print(f"   Skipping {mechanism}: not enough samples")
            continue

        # Train model
        print(f"\n   Training XGBoost model...")
        model = train_model(X_train, y_train, model_type='xgboost')

        # Evaluate
        print(f"   Evaluating...")
        metrics, cm, y_proba = evaluate_model(model, X_test, y_test)

        print(f"\n   Results:")
        print(f"   AUC:      {metrics['auc']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1-Score: {metrics['f1']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:   {metrics['recall']:.4f}")

        # Feature importance
        print(f"\n   Top 10 Important Features:")
        top_features = get_feature_importance(model, all_features, top_n=10)
        for i, (feat, imp) in enumerate(top_features, 1):
            print(f"   {i:2d}. {feat:30s} {imp:.4f}")

        # Save model
        model_file = output_dir / f"{mechanism.lower()}_model.json"
        model.save_model(model_file)
        print(f"\n   Model saved to: {model_file}")

        # Store results
        all_results[mechanism] = {
            'metrics': {k: float(v) for k, v in metrics.items()},
            'confusion_matrix': cm.tolist(),
            'feature_importance': [(f, float(i)) for f, i in top_features],
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'positive_rate_train': float(y_train.mean()),
            'positive_rate_test': float(y_test.mean()),
        }

    # Save all results
    results_file = output_dir / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n   Results saved to: {results_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("Summary of All Models")
    print(f"{'='*80}")
    print(f"{'Mechanism':<12} {'Samples':>10} {'AUC':>8} {'Accuracy':>8} {'F1':>8}")
    print(f"{'-'*50}")
    for mech, res in all_results.items():
        n_samples = res['train_samples'] + res['test_samples']
        auc = res['metrics']['auc']
        acc = res['metrics']['accuracy']
        f1 = res['metrics']['f1']
        print(f"{mech:<12} {n_samples:>10} {auc:>8.4f} {acc:>8.4f} {f1:>8.4f}")

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
