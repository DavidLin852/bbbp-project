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
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


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

    # Random Forest Regressor
    rf_reg_n_estimators: int = 500

    # XGBoost Regressor
    xgb_reg_n_estimators: int = 1000
    xgb_reg_max_depth: int = 6
    xgb_reg_learning_rate: float = 0.05

    # LightGBM Regressor
    lgbm_reg_n_estimators: int = 1500
    lgbm_reg_max_depth: int = -1
    lgbm_reg_learning_rate: float = 0.03
    lgbm_reg_num_leaves: int = 64

    # KNN Regressor
    knn_reg_n_neighbors: int = 5

    # Ridge
    ridge_alpha: float = 1.0

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
            "rf", "xgb", "lgbm", "svm", "knn", "lr", "nb", "gb", "ada", "etc",
            "rf_reg", "xgb_reg", "lgbm_reg", "svm_reg", "ridge", "knn_reg",
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
        # For LightGBM: sklearn warns if feature-name presence differs
        # between fit and predict.  We store the column names used at fit
        # time so predict can produce a matching DataFrame.
        self._lgbm_columns: list[str] | None = None

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
        elif self.model_type == "rf_reg":
            return RandomForestRegressor(
                n_estimators=cfg.rf_reg_n_estimators,
                max_depth=cfg.rf_max_depth,
                min_samples_split=cfg.rf_min_samples_split,
                n_jobs=cfg.n_jobs,
                random_state=cfg.random_state,
            )
        elif self.model_type == "xgb_reg":
            return XGBRegressor(
                n_estimators=cfg.xgb_reg_n_estimators,
                max_depth=cfg.xgb_reg_max_depth,
                learning_rate=cfg.xgb_reg_learning_rate,
                objective="reg:squarederror",
                n_jobs=cfg.n_jobs,
                random_state=cfg.random_state,
            )
        elif self.model_type == "lgbm_reg":
            return LGBMRegressor(
                n_estimators=cfg.lgbm_reg_n_estimators,
                max_depth=cfg.lgbm_reg_max_depth,
                learning_rate=cfg.lgbm_reg_learning_rate,
                num_leaves=cfg.lgbm_reg_num_leaves,
                objective="regression",
                n_jobs=cfg.n_jobs,
                random_state=cfg.random_state,
                verbose=-1,
            )
        elif self.model_type == "svm_reg":
            return SVR(
                kernel="rbf",
                C=cfg.svm_c,
            )
        elif self.model_type == "ridge":
            return Ridge(
                alpha=cfg.ridge_alpha,
                random_state=cfg.random_state,
            )
        elif self.model_type == "knn_reg":
            return KNeighborsRegressor(
                n_neighbors=cfg.knn_reg_n_neighbors,
                n_jobs=cfg.n_jobs,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    @staticmethod
    def _to_numpy(X):
        """Convert sparse or array-like to dense numpy."""
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X)

    def _lgbm_wrap(self, X: np.ndarray) -> pd.DataFrame:
        """Wrap numpy array in a DataFrame matching LightGBM's auto-generated column names."""
        cols = self._lgbm_columns
        if cols is None:
            cols = [f"Column_{i}" for i in range(X.shape[1])]
            self._lgbm_columns = cols
        return pd.DataFrame(X, columns=cols)

    def fit(self, X, y):
        """
        Train the model.

        Args:
            X: Feature matrix (numpy array or sparse matrix)
            y: Labels
        """
        X = self._to_numpy(X)

        # LightGBM requires float32; wrap in DataFrame so sklearn's
        # feature-name check stays consistent with predict.
        if self.model_type.startswith("lgbm"):
            X = self._lgbm_wrap(X.astype(np.float32))

        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Feature matrix (numpy array or sparse matrix)

        Returns:
            Array of probabilities (n_samples, 2)
        """
        X = self._to_numpy(X)

        if self.model_type.startswith("lgbm"):
            X = self._lgbm_wrap(X.astype(np.float32))

        return self.model.predict_proba(X)

    def predict(self, X):
        """
        Predict class labels or continuous values.

        Args:
            X: Feature matrix (numpy array or sparse matrix)

        Returns:
            Array of predictions (n_samples,)
        """
        X = self._to_numpy(X)

        if self.model_type.startswith("lgbm"):
            X = self._lgbm_wrap(X.astype(np.float32))

        return self.model.predict(X)

    def save(self, path: str):
        """Save model to file."""
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load model from file."""
        self.model = joblib.load(path)
        return self
