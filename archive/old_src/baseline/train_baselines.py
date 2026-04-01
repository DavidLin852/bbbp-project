"""
Comprehensive Baseline Model Training

Supports multiple ML models:
- RF: Random Forest
- XGB: XGBoost
- LGBM: LightGBM
- SVM: Support Vector Machine
- KNN: K-Nearest Neighbors
- NB: Naive Bayes (Gaussian, Bernoulli)
- LR: Logistic Regression
- MLP: Multi-Layer Perceptron
- ETC: Extra Trees Classifier
- ADA: AdaBoost
- GB: Gradient Boosting

Usage:
    python scripts/04_run_all_baselines.py --seed 0 --feature morgan
    python scripts/04_run_all_baselines.py --seed 0 --feature combined --models rf,xgb,svm,knn
"""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ..utils.metrics import classification_metrics


# =============================================================================
# Model Configurations
# =============================================================================

MODEL_CONFIGS = {
    # Random Forest
    "RF": {
        "model": RandomForestClassifier,
        "params": {
            "n_estimators": 800,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1,
            "random_state": 42,
            "class_weight": "balanced_subsample"
        }
    },

    # Extra Trees
    "ETC": {
        "model": ExtraTreesClassifier,
        "params": {
            "n_estimators": 800,
            "max_depth": None,
            "min_samples_split": 2,
            "n_jobs": -1,
            "random_state": 42,
            "class_weight": "balanced_subsample"
        }
    },

    # XGBoost
    "XGB": {
        "model": XGBClassifier,
        "params": {
            "n_estimators": 1200,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_jobs": -1,
            "random_state": 42,
        }
    },

    # LightGBM
    "LGBM": {
        "model": LGBMClassifier,
        "params": {
            "n_estimators": 2000,
            "max_depth": -1,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42,
            "class_weight": "balanced",
            "n_jobs": -1,
            "verbose": -1
        }
    },

    # Gradient Boosting
    "GB": {
        "model": GradientBoostingClassifier,
        "params": {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "random_state": 42
        }
    },

    # AdaBoost
    "ADA": {
        "model": AdaBoostClassifier,
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.5,
            "random_state": 42
        }
    },

    # SVM - RBF Kernel
    "SVM_RBF": {
        "model": SVC,
        "params": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "random_state": 42,
            "class_weight": "balanced"
        },
        "needs_scaling": True
    },

    # SVM - Linear Kernel
    "SVM_LINEAR": {
        "model": SVC,
        "params": {
            "C": 1.0,
            "kernel": "linear",
            "probability": True,
            "random_state": 42,
            "class_weight": "balanced"
        },
        "needs_scaling": True
    },

    # SVM - Polynomial Kernel
    "SVM_POLY": {
        "model": SVC,
        "params": {
            "C": 1.0,
            "kernel": "poly",
            "degree": 3,
            "gamma": "scale",
            "probability": True,
            "random_state": 42,
            "class_weight": "balanced"
        },
        "needs_scaling": True
    },

    # KNN
    "KNN3": {
        "model": KNeighborsClassifier,
        "params": {
            "n_neighbors": 3,
            "weights": "distance",
            "n_jobs": -1
        },
        "needs_scaling": True
    },

    "KNN5": {
        "model": KNeighborsClassifier,
        "params": {
            "n_neighbors": 5,
            "weights": "distance",
            "n_jobs": -1
        },
        "needs_scaling": True
    },

    "KNN7": {
        "model": KNeighborsClassifier,
        "params": {
            "n_neighbors": 7,
            "weights": "distance",
            "n_jobs": -1
        },
        "needs_scaling": True
    },

    # Naive Bayes - Gaussian
    "NB_Gaussian": {
        "model": GaussianNB,
        "params": {
            "var_smoothing": 1e-9
        },
        "needs_scaling": False  # NB works better without scaling sometimes
    },

    # Naive Bayes - Bernoulli
    "NB_Bernoulli": {
        "model": BernoulliNB,
        "params": {
            "alpha": 1.0
        },
        "needs_scaling": False
    },

    # Logistic Regression
    "LR": {
        "model": LogisticRegression,
        "params": {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "n_jobs": -1
        },
        "needs_scaling": True
    },

    # MLP Neural Network
    "MLP": {
        "model": MLPClassifier,
        "params": {
            "hidden_layer_sizes": (256, 128),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.001,
            "batch_size": 64,
            "learning_rate": "adaptive",
            "max_iter": 500,
            "random_state": 42,
            "early_stopping": True,
            "validation_fraction": 0.1
        },
        "needs_scaling": True
    },

    "MLP_Small": {
        "model": MLPClassifier,
        "params": {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.001,
            "batch_size": 64,
            "learning_rate": "adaptive",
            "max_iter": 500,
            "random_state": 42,
            "early_stopping": True,
            "validation_fraction": 0.1
        },
        "needs_scaling": True
    }
}


def _save_model(model, out_dir: Path, name: str):
    """Save model to joblib file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / f"{name}.joblib")


def train_single_model(
    model_name: str,
    model_config: dict,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    out_model_dir: Path,
    seed: int
):
    """Train and evaluate a single model."""
    # Create model instance
    model_cls = model_config["model"]
    params = model_config["params"].copy()
    if "random_state" in params:
        params["random_state"] = seed

    needs_scaling = model_config.get("needs_scaling", False)

    # Handle SVM probability parameter warning
    if "probability" in params:
        params["probability"] = True

    model = model_cls(**params)

    # Convert data dtype for LightGBM (needs float32/float64)
    if model_cls.__name__ == 'LGBMClassifier':
        if hasattr(X_train, 'astype'):
            X_train = X_train.astype(np.float32)
        if hasattr(X_val, 'astype'):
            X_val = X_val.astype(np.float32)
        if hasattr(X_test, 'astype'):
            X_test = X_test.astype(np.float32)

    # Create pipeline with scaling if needed
    if needs_scaling:
        # Use StandardScaler with_mean=False for sparse matrices
        scaler = StandardScaler(with_mean=False)
        pipe = Pipeline([
            ("scaler", scaler),
            ("model", model)
        ])
    else:
        pipe = Pipeline([
            ("model", model)
        ])

    # Fit
    pipe.fit(X_train, y_train)

    # Predict on test
    prob_test = pipe.predict_proba(X_test)[:, 1]
    m = classification_metrics(y_test, prob_test, threshold=0.5)

    # Compute specificity and MCC manually
    y_pred = (prob_test >= 0.5).astype(int)
    specificity = m.tn / (m.tn + m.fp) if (m.tn + m.fp) > 0 else 0
    mcc = (m.tp * m.tn - m.fp * m.fn) / np.sqrt((m.tp + m.fp) * (m.tp + m.fn) * (m.tn + m.fp) * (m.tn + m.fn)) if (m.tp + m.fp) * (m.tp + m.fn) * (m.tn + m.fp) * (m.tn + m.fn) > 0 else 0

    # Save model
    _save_model(pipe, out_model_dir, f"{model_name}_seed{seed}")

    return {
        "model": model_name,
        "split": "test",
        "auc": m.auc,
        "auprc": m.auprc,
        "accuracy": m.accuracy,
        "precision": m.precision_pos,
        "recall": m.recall_pos,
        "f1": m.f1_pos,
        "specificity": specificity,
        "mcc": mcc,
        "y_true": y_test,
        "y_prob": prob_test
    }


def train_eval_all_models(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    out_model_dir: Path,
    run_info: dict,
    model_list: list[str] = None
):
    """Train and evaluate all specified models.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (for early stopping if needed)
        X_test, y_test: Test data
        out_model_dir: Output directory for models
        run_info: Dictionary with run metadata
        model_list: List of model names to train (None = all)

    Returns:
        List of result dictionaries and predictions for ROC
    """
    if model_list is None:
        model_list = list(MODEL_CONFIGS.keys())

    rows = []
    preds_for_roc = []

    for model_name in model_list:
        model_name_upper = model_name.upper()
        if model_name_upper not in MODEL_CONFIGS:
            print(f"  Warning: Model {model_name} not found in MODEL_CONFIGS, skipping...")
            continue

        print(f"  Training {model_name_upper}...")
        config = MODEL_CONFIGS[model_name_upper]

        result = train_single_model(
            model_name_upper,
            config,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            out_model_dir,
            run_info["seed"]
        )

        # Add run info
        row = dict(run_info)
        row.update({
            "model": result["model"],
            "split": "test",
            "auc": result["auc"],
            "auprc": result["auprc"],
            "accuracy": result["accuracy"],
            "precision": result.get("precision", result.get("precision_pos", 0)),
            "recall": result.get("recall", result.get("recall_pos", 0)),
            "f1": result.get("f1", result.get("f1_pos", 0)),
            "specificity": result["specificity"],
            "mcc": result["mcc"]
        })
        rows.append(row)

        preds_for_roc.append({
            "name": f"{run_info['feature']}_{result['model']}",
            "y_true": result["y_true"],
            "y_prob": result["y_prob"]
        })

        f1_val = result.get("f1", result.get("f1_pos", 0))
        print(f"    AUC: {result['auc']:.4f}, F1: {f1_val:.4f}")

    return rows, preds_for_roc


def get_available_models() -> list[str]:
    """Get list of available model names."""
    return list(MODEL_CONFIGS.keys())


def get_model_info(model_name: str) -> dict:
    """Get information about a specific model."""
    if model_name not in MODEL_CONFIGS:
        return None
    return {
        "name": model_name,
        "class": MODEL_CONFIGS[model_name]["model"].__name__,
        "params": MODEL_CONFIGS[model_name]["params"],
        "needs_scaling": MODEL_CONFIGS[model_name].get("needs_scaling", False)
    }


# =============================================================================
# Model Aliases for Convenience
# =============================================================================

# All models
ALL_MODELS = list(MODEL_CONFIGS.keys())

# Tree-based ensemble
ENSEMBLE_MODELS = ["RF", "ETC", "XGB", "LGBM", "GB", "ADA"]

# SVM variants
SVM_MODELS = ["SVM_RBF", "SVM_LINEAR", "SVM_POLY"]

# KNN variants
KNN_MODELS = ["KNN3", "KNN5", "KNN7"]

# Naive Bayes
NB_MODELS = ["NB_Gaussian", "NB_Bernoulli"]

# Neural Network
NN_MODELS = ["MLP", "MLP_Small"]

# Traditional ML (excluding deep learning)
TRADITIONAL_MODELS = ["LR", "SVM_RBF", "KNN5", "NB_Gaussian"]

# Quick test set
QUICK_MODELS = ["RF", "XGB", "LGBM", "SVM_RBF", "LR"]
