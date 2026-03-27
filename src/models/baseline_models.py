"""
Baseline classical ML models.

Supports:
- Random Forest
- XGBoost
- LightGBM
- SVM
- KNN
- Logistic Regression
- Naive Bayes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import joblib
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


@dataclass
class ModelConfig:
    """Configuration for baseline models."""

    # Random Forest
    rf_n_estimators: int = 800
    rf_max_depth: int | None = None
    rf_min_samples_split: int = 2
    rf_class_weight: str = "balanced_subsample"

    # XGBoost
    xgb_n_estimators: int = 1200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.03
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.8

    # LightGBM
    lgbm_n_estimators: int = 2000
    lgbm_max_depth: int = -1
    lgbm_learning_rate: float = 0.01
    lgbm_num_leaves: int = 64

    # SVM
    svm_c: float = 1.0
    svm_kernel: str = "rbf"
    svm_probability: bool = True

    # KNN
    knn_n_neighbors: int = 5

    # Logistic Regression
    lr_c: float = 1.0
    lr_max_iter: int = 1000

    # Random seed
    random_state: int = 42
    n_jobs: int = -1


class BaselineModel:
    """
    Wrapper for baseline ML models.

    Provides unified interface for training and prediction
    across multiple model types.
    """

    def __init__(
        self,
        model_type: Literal[
            "rf", "xgb", "lgbm", "svm", "knn", "lr", "nb", "gb", "ada", "etc"
        ],
        config: ModelConfig | None = None,
    ):
        """
        Initialize baseline model.

        Args:
            model_type: Type of model to create
            config: Model configuration
        """
        self.model_type = model_type
        self.config = config or ModelConfig()
        self.model = self._create_model()

    def _create_model(self):
        """Create model instance based on type."""
        cfg = self.config

        if self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=cfg.rf_n_estimators,
                max_depth=cfg.rf_max_depth,
                min_samples_split=cfg.rf_min_samples_split,
                class_weight=cfg.rf_class_weight,
                n_jobs=cfg.n_jobs,
                random_state=cfg.random_state,
            )
        elif self.model_type == "xgb":
            return XGBClassifier(
                n_estimators=cfg.xgb_n_estimators,
                max_depth=cfg.xgb_max_depth,
                learning_rate=cfg.xgb_learning_rate,
                subsample=cfg.xgb_subsample,
                colsample_bytree=cfg.xgb_colsample_bytree,
                objective="binary:logistic",
                eval_metric="auc",
                n_jobs=cfg.n_jobs,
                random_state=cfg.random_state,
                use_label_encoder=False,
            )
        elif self.model_type == "lgbm":
            return LGBMClassifier(
                n_estimators=cfg.lgbm_n_estimators,
                max_depth=cfg.lgbm_max_depth,
                learning_rate=cfg.lgbm_learning_rate,
                num_leaves=cfg.lgbm_num_leaves,
                objective="binary",
                n_jobs=cfg.n_jobs,
                random_state=cfg.random_state,
                verbose=-1,
                feature_name=[],
            )
        elif self.model_type == "svm":
            return SVC(
                C=cfg.svm_c,
                kernel=cfg.svm_kernel,
                probability=cfg.svm_probability,
                random_state=cfg.random_state,
            )
        elif self.model_type == "knn":
            return KNeighborsClassifier(
                n_neighbors=cfg.knn_n_neighbors,
                n_jobs=cfg.n_jobs,
            )
        elif self.model_type == "lr":
            return LogisticRegression(
                C=cfg.lr_c,
                max_iter=cfg.lr_max_iter,
                random_state=cfg.random_state,
                n_jobs=cfg.n_jobs,
            )
        elif self.model_type == "nb":
            return GaussianNB()
        elif self.model_type == "gb":
            return GradientBoostingClassifier(
                random_state=cfg.random_state,
            )
        elif self.model_type == "ada":
            return AdaBoostClassifier(
                random_state=cfg.random_state,
            )
        elif self.model_type == "etc":
            return ExtraTreesClassifier(
                n_estimators=cfg.rf_n_estimators,
                class_weight=cfg.rf_class_weight,
                n_jobs=cfg.n_jobs,
                random_state=cfg.random_state,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X, y):
        """
        Train the model.

        Args:
            X: Feature matrix
            y: Labels
        """
        # Convert sparse to dense for some models
        if hasattr(X, "toarray"):
            if self.model_type in ["svm", "knn", "lr", "nb"]:
                X = X.toarray()

        # Handle LGBM dtype
        if self.model_type == "lgbm":
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = X.astype(np.float32)

        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities (n_samples, 2)
        """
        # Convert sparse to dense for some models
        if hasattr(X, "toarray"):
            if self.model_type in ["svm", "knn", "lr", "nb"]:
                X = X.toarray()

        # Handle LGBM dtype
        if self.model_type == "lgbm":
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = X.astype(np.float32)

        return self.model.predict_proba(X)

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions (n_samples,)
        """
        # Convert sparse to dense for some models
        if hasattr(X, "toarray"):
            if self.model_type in ["svm", "knn", "lr", "nb"]:
                X = X.toarray()

        # Handle LGBM dtype
        if self.model_type == "lgbm":
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = X.astype(np.float32)

        return self.model.predict(X)

    def save(self, path: str):
        """Save model to file."""
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load model from file."""
        self.model = joblib.load(path)
        return self
