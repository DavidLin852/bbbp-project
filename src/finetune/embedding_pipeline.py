"""
LightGBM training on pretrained embeddings (Groups B and C).

Group B: Train LightGBM on pretrained embeddings alone.
Group C: Concatenate pretrained embeddings with classical fingerprints/descriptors, then train LightGBM.

Reuses existing LightGBM hyperparameters from the baseline pipeline.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, f1_score, r2_score, mean_squared_error, mean_absolute_error

# Add project root
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.finetune.finetune_config import get_pretrain_config


# ============================================================
# Metrics
# ============================================================

def compute_cls_metrics(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float, float]:
    """Compute AUC, F1, Accuracy for classification."""
    if len(np.unique(labels)) < 2:
        return float("nan"), 0.0, 0.0
    auc = roc_auc_score(labels, preds)
    binary_preds = (preds >= 0.5).astype(int)
    f1 = f1_score(labels, binary_preds, zero_division=0)
    acc = float((binary_preds == labels).mean())
    return auc, f1, acc


def compute_reg_metrics(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float, float]:
    """Compute R², RMSE, MAE for regression."""
    r2 = r2_score(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae = mean_absolute_error(labels, preds)
    return r2, rmse, mae


# ============================================================
# LightGBM Training
# ============================================================

def train_lgbm_on_embeddings(
    pretrain_id: str,
    seed: int,
    task: Literal["classification", "regression"],
    feature_type: str | None,
    output_dir: str | Path,
    embedding_dir: str | Path = "artifacts/embeddings",
    verbose: bool = True,
) -> dict | None:
    """
    Train LightGBM on pretrained embeddings (with optional feature concatenation).

    Args:
        pretrain_id: e.g. "P_E10_GIN_1M"
        seed: Random seed (0-4)
        task: "classification" or "regression"
        feature_type: Classical feature to concatenate (morgan/maccs/fp2/descriptors_basic).
                      None for Group B (embedding only).
        output_dir: Where to save model and results
        embedding_dir: Where embeddings are cached
        verbose: Print progress

    Returns:
        Result dict on success, None on failure
    """
    embedding_dir = Path(embedding_dir)
    output_dir = Path(output_dir)

    # --- Output subdir ---
    if feature_type:
        exp_dir = output_dir / f"{pretrain_id}+{feature_type}"
    else:
        exp_dir = output_dir / pretrain_id

    model_dir = exp_dir / f"seed_{seed}" / task
    model_dir.mkdir(parents=True, exist_ok=True)

    result_file = model_dir / "result.json"
    if result_file.exists():
        if verbose:
            print(f"  SKIP: {model_dir} already has result.json")
        return json.loads(result_file.read_text())

    if verbose:
        feat_str = f"+{feature_type}" if feature_type else " (embedding only)"
        print(f"  LGBM {pretrain_id}{feat_str}, seed={seed}, {task}")

    # --- Load embeddings ---
    emb_subdir = embedding_dir / pretrain_id / f"seed_{seed}" / task

    try:
        X_train = np.load(emb_subdir / "X_train.npy").astype(np.float32)
        X_val = np.load(emb_subdir / "X_val.npy").astype(np.float32)
        X_test = np.load(emb_subdir / "X_test.npy").astype(np.float32)
        y_train = np.load(emb_subdir / "y_train.npy")
        y_val = np.load(emb_subdir / "y_val.npy")
        y_test = np.load(emb_subdir / "y_test.npy")
    except FileNotFoundError as e:
        if verbose:
            print(f"  [ERROR] Missing embedding file: {e}")
        return None

    # --- Load and concatenate classical features (Group C) ---
    if feature_type:
        feat_dir = project_root / "artifacts" / "features" / f"seed_{seed}" / "scaffold" / task / feature_type
        try:
            F_train = np.load(feat_dir / "X_train.npy").astype(np.float32)
            F_val = np.load(feat_dir / "X_val.npy").astype(np.float32)
            F_test = np.load(feat_dir / "X_test.npy").astype(np.float32)

            if hasattr(F_train, "toarray"):
                F_train = F_train.toarray()
                F_val = F_val.toarray()
                F_test = F_test.toarray()

            X_train = np.concatenate([X_train, F_train], axis=1)
            X_val = np.concatenate([X_val, F_val], axis=1)
            X_test = np.concatenate([X_test, F_test], axis=1)
        except FileNotFoundError as e:
            if verbose:
                print(f"  [ERROR] Missing feature file: {e}")
            return None

    if verbose:
        print(f"    Train: X={X_train.shape}, y range=[{y_train.min():.2f}, {y_train.max():.2f}]")
        print(f"    Val: X={X_val.shape}")
        print(f"    Test: X={X_test.shape}")

    # --- Train LightGBM ---
    # Use EXACT same hyperparameters as baseline from ModelConfig
    if task == "classification":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.01,
            num_leaves=64,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lambda env: (
                    True  # disable logging
                    if verbose and env.iteration % 500 == 0
                    else None
                )
            ],
        )

        # Predict
        train_preds = model.predict_proba(X_train)[:, 1]
        val_preds = model.predict_proba(X_val)[:, 1]
        test_preds = model.predict_proba(X_test)[:, 1]

        # Metrics
        train_auc, train_f1, train_acc = compute_cls_metrics(y_train, train_preds)
        val_auc, val_f1, val_acc = compute_cls_metrics(y_val, val_preds)
        test_auc, test_f1, test_acc = compute_cls_metrics(y_test, test_preds)

        result = {
            "exp_id": f"{pretrain_id}+{feature_type}" if feature_type else pretrain_id,
            "group": "C" if feature_type else "B",
            "pretrain_id": pretrain_id,
            "model_type": get_pretrain_config(pretrain_id)["model_type"],
            "strategy": get_pretrain_config(pretrain_id)["strategy"],
            "task": task,
            "feature_type": feature_type,
            "seed": seed,
            "train_auc": float(train_auc),
            "train_f1": float(train_f1),
            "train_acc": float(train_acc),
            "val_auc": float(val_auc),
            "val_f1": float(val_f1),
            "val_acc": float(val_acc),
            "test_auc": float(test_auc),
            "test_f1": float(test_f1),
            "test_acc": float(test_acc),
            "input_dim": int(X_train.shape[1]),
        }

        if verbose:
            print(f"    Test AUC: {test_auc:.4f}, F1: {test_f1:.4f}, Acc: {test_acc:.4f}")

    else:  # regression
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            num_leaves=64,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        # Predict
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

        # Metrics
        train_r2, train_rmse, train_mae = compute_reg_metrics(y_train, train_preds)
        val_r2, val_rmse, val_mae = compute_reg_metrics(y_val, val_preds)
        test_r2, test_rmse, test_mae = compute_reg_metrics(y_test, test_preds)

        result = {
            "exp_id": f"{pretrain_id}+{feature_type}" if feature_type else pretrain_id,
            "group": "C" if feature_type else "B",
            "pretrain_id": pretrain_id,
            "model_type": get_pretrain_config(pretrain_id)["model_type"],
            "strategy": get_pretrain_config(pretrain_id)["strategy"],
            "task": task,
            "feature_type": feature_type,
            "seed": seed,
            "train_r2": float(train_r2),
            "train_rmse": float(train_rmse),
            "train_mae": float(train_mae),
            "val_r2": float(val_r2),
            "val_rmse": float(val_rmse),
            "val_mae": float(val_mae),
            "test_r2": float(test_r2),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "input_dim": int(X_train.shape[1]),
        }

        if verbose:
            print(f"    Test R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

    # --- Save ---
    with open(model_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    joblib.dump(model, model_dir / "model.joblib")

    if verbose:
        print(f"    Saved: {model_dir}")

    return result
