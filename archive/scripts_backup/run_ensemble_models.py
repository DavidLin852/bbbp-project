"""
Ensemble Models: Stacking and Soft Voting
- Stacking: Base learners + Meta-learner (RF or XGBoost)
- Soft Voting: Weighted average of predicted probabilities
"""

import sys
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)
from sklearn.model_selection import cross_val_predict
import joblib

# UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent


def load_data(seed=0):
    """Load data splits"""
    split_dir = PROJECT_ROOT / "data" / "splits" / f"seed_{seed}"
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    return train_df, val_df, test_df


def load_features(feature_name, seed=0):
    """Load features"""
    feat_dir = PROJECT_ROOT / "artifacts" / "features" / f"seed_{seed}_enhanced"

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

    meta = pd.read_csv(feat_dir / "meta.csv")
    row_id_to_idx = {row_id: idx for idx, row_id in enumerate(meta["row_id"].values)}

    return X, row_id_to_idx


def get_feature_indices(train_df, val_df, test_df, row_id_to_idx):
    """Get feature indices"""
    train_idx = train_df["row_id"].map(row_id_to_idx).values
    val_idx = val_df["row_id"].map(row_id_to_idx).values
    test_idx = test_df["row_id"].map(row_id_to_idx).values
    return train_idx, val_idx, test_idx


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all performance metrics"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    se = tp / (tp + fn)  # Sensitivity / Recall
    sp = tn / (tn + fp)  # Specificity
    ba = (se + sp) / 2   # Balanced Accuracy

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': se,
        'sensitivity': se,
        'specificity': sp,
        'f1': f1_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'ba': ba,
        'auc': roc_auc_score(y_true, y_prob)
    }


def get_base_estimators():
    """Define base estimators for ensemble"""
    return [
        ('rf', RandomForestClassifier(
            n_estimators=800,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
        ('xgb', GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )),
        ('lgbm', GradientBoostingClassifier(  # Use GB as LGBM substitute
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )),
        ('svm', SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )),
        ('knn', KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )),
        ('lr', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=800,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
        ('ada', AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.5,
            random_state=42
        )),
        ('nb', BernoulliNB(alpha=1.0))
    ]


def run_stacking_ensemble(X_train, y_train, X_val, y_val, X_test, y_test, meta_learner='rf'):
    """Run Stacking Ensemble"""

    print(f"\n{'='*60}")
    print(f"Stacking Ensemble (Meta-Learner: {meta_learner.upper()})")
    print(f"{'='*60}")

    # Define meta-learner
    if meta_learner == 'rf':
        final_estimator = RandomForestClassifier(
            n_estimators=500,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif meta_learner == 'xgb':
        final_estimator = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
    else:
        final_estimator = RandomForestClassifier(
            n_estimators=500,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

    # Create stacking classifier
    stacking = StackingClassifier(
        estimators=get_base_estimators()[:6],  # Use top 6 base learners
        final_estimator=final_estimator,
        cv=5,
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=-1
    )

    # Train
    print("Training Stacking Ensemble...")
    stacking.fit(X_train, y_train)

    # Predict
    y_pred = stacking.predict(X_test)
    y_prob = stacking.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    metrics['model'] = f'Stacking_{meta_learner}'
    metrics['feature'] = 'combined'

    print(f"\nStacking Results ({meta_learner.upper()}):")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  MCC: {metrics['mcc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  BA: {metrics['ba']:.4f}")

    return stacking, metrics


def run_soft_voting_ensemble(X_train, y_train, X_val, y_val, X_test, y_test, weights=None):
    """Run Soft Voting Ensemble"""

    print(f"\n{'='*60}")
    print("Soft Voting Ensemble")
    print(f"{'='*60}")

    # Calculate weights based on performance if not provided
    if weights is None:
        # Default weights favoring better performers
        weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.15]  # RF, XGB, LGBM, SVM, KNN, LR

    # Create voting classifier
    voting = VotingClassifier(
        estimators=get_base_estimators()[:6],  # Use top 6 base learners
        voting='soft',
        weights=weights,
        n_jobs=-1
    )

    # Train
    print("Training Soft Voting Ensemble...")
    voting.fit(X_train, y_train)

    # Predict
    y_pred = voting.predict(X_test)
    y_prob = voting.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    metrics['model'] = 'SoftVoting'
    metrics['feature'] = 'combined'

    print(f"\nSoft Voting Results:")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  MCC: {metrics['mcc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  BA: {metrics['ba']:.4f}")

    return voting, metrics


def run_ensemble_experiments():
    """Run all ensemble experiments"""

    results_dir = PROJECT_ROOT / "artifacts" / "ablation"
    model_dir = PROJECT_ROOT / "artifacts" / "models" / "ensemble"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, val_df, test_df = load_data(seed=0)

    # Load combined features
    X_all, row_id_to_idx = load_features('morgan', seed=0)
    train_idx, val_idx, test_idx = get_feature_indices(train_df, val_df, test_df, row_id_to_idx)

    # Combine Morgan + MACCS + AtomPairs + FP2
    for feat_name in ['maccs', 'atompairs', 'fp2']:
        X_feat, _ = load_features(feat_name, seed=0)
        X_all = sparse.hstack([X_all, X_feat])

    X_train = X_all[train_idx]
    X_val = X_all[val_idx]
    X_test = X_all[test_idx]

    y_train = train_df["y_cls"].values
    y_val = val_df["y_cls"].values
    y_test = test_df["y_cls"].values

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")

    all_results = []

    # Run Stacking with RF meta-learner
    stacking_rf, metrics_rf = run_stacking_ensemble(
        X_train, y_train, X_val, y_val, X_test, y_test, meta_learner='rf'
    )
    all_results.append(metrics_rf)

    # Save model
    joblib.dump(stacking_rf, model_dir / "stacking_rf.joblib")
    print(f"Model saved: {model_dir / 'stacking_rf.joblib'}")

    # Run Stacking with XGB meta-learner
    stacking_xgb, metrics_xgb = run_stacking_ensemble(
        X_train, y_train, X_val, y_val, X_test, y_test, meta_learner='xgb'
    )
    all_results.append(metrics_xgb)

    # Save model
    joblib.dump(stacking_xgb, model_dir / "stacking_xgb.joblib")
    print(f"Model saved: {model_dir / 'stacking_xgb.joblib'}")

    # Run Soft Voting
    voting, metrics_voting = run_soft_voting_ensemble(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    all_results.append(metrics_voting)

    # Save model
    joblib.dump(voting, model_dir / "soft_voting.joblib")
    print(f"Model saved: {model_dir / 'soft_voting.joblib'}")

    # Save results
    df_results = pd.DataFrame(all_results)
    output_file = results_dir / "ENSEMBLE_RESULTS.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n[OK] Results saved: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("Ensemble Summary")
    print("=" * 60)
    for result in all_results:
        print(f"  {result['model']:20} AUC={result['auc']:.4f} F1={result['f1']:.4f} MCC={result['mcc']:.4f}")

    return df_results


def main():
    """Main function"""

    print("=" * 80)
    print("Ensemble Learning: Stacking and Soft Voting")
    print("=" * 80)

    results = run_ensemble_experiments()

    print("\n" + "=" * 80)
    print("Ensemble Experiments Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
