"""
Model training and evaluation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..models import BaselineModel
from ..utils.metrics import classification_metrics, ClsMetrics


@dataclass
class TrainingConfig:
    """Configuration for training."""
    output_dir: Path | str = "artifacts/models"
    save_model: bool = True
    save_predictions: bool = True
    threshold: float = 0.5


@dataclass
class TrainingResult:
    """Container for training results."""
    model_name: str
    feature_type: str
    train_metrics: ClsMetrics
    val_metrics: ClsMetrics
    test_metrics: ClsMetrics
    model: Any | None = None

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "feature_type": self.feature_type,
            "train_auc": self.train_metrics.auc,
            "train_accuracy": self.train_metrics.accuracy,
            "train_f1": self.train_metrics.f1_pos,
            "val_auc": self.val_metrics.auc,
            "val_accuracy": self.val_metrics.accuracy,
            "val_f1": self.val_metrics.f1_pos,
            "test_auc": self.test_metrics.auc,
            "test_accuracy": self.test_metrics.accuracy,
            "test_f1": self.test_metrics.f1_pos,
        }


class Trainer:
    """
    Train and evaluate baseline models.

    Provides unified interface for training models,
    computing metrics, and saving results.
    """

    def __init__(
        self,
        model: BaselineModel,
        config: TrainingConfig | None = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Training configuration
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) -> TrainingResult:
        """
        Train model and evaluate on all splits.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels

        Returns:
            TrainingResult with metrics
        """
        # Train model
        self.model.fit(X_train, y_train)

        # Predictions
        train_probs = self.model.predict_proba(X_train)[:, 1]
        val_probs = self.model.predict_proba(X_val)[:, 1]
        test_probs = self.model.predict_proba(X_test)[:, 1]

        # Compute metrics
        train_metrics = classification_metrics(
            y_train, train_probs, self.config.threshold
        )
        val_metrics = classification_metrics(
            y_val, val_probs, self.config.threshold
        )
        test_metrics = classification_metrics(
            y_test, test_probs, self.config.threshold
        )

        # Create result
        result = TrainingResult(
            model_name=self.model.model_type,
            feature_type="unknown",  # Set by caller
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            model=self.model if self.config.save_model else None,
        )

        # Save model
        if self.config.save_model:
            self._save_model(result)

        # Save predictions
        if self.config.save_predictions:
            self._save_predictions(
                X_train, y_train, train_probs, "train"
            )
            self._save_predictions(
                X_val, y_val, val_probs, "val"
            )
            self._save_predictions(
                X_test, y_test, test_probs, "test"
            )

        return result

    def _save_model(self, result: TrainingResult):
        """Save trained model."""
        model_dir = self.output_dir / result.model_name
        model_dir.mkdir(exist_ok=True)

        model_path = model_dir / f"{result.model_name}_model.joblib"
        joblib.dump(self.model.model, model_path)

    def _save_predictions(
        self,
        X,
        y_true,
        y_prob,
        split_name: str,
    ):
        """Save predictions to CSV."""
        model_dir = self.output_dir / self.model.model_type
        model_dir.mkdir(exist_ok=True)

        y_pred = (y_prob >= self.config.threshold).astype(int)

        df = pd.DataFrame({
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
        })

        output_path = model_dir / f"{split_name}_predictions.csv"
        df.to_csv(output_path, index=False)


def train_multiple_models(
    models: list[BaselineModel],
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_type: str,
    output_dir: Path | str = "artifacts/models",
) -> list[TrainingResult]:
    """
    Train multiple models and collect results.

    Args:
        models: List of models to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        feature_type: Name of feature type
        output_dir: Output directory

    Returns:
        List of TrainingResult objects
    """
    results = []

    for model in models:
        config = TrainingConfig(
            output_dir=output_dir,
            save_model=True,
            save_predictions=True,
        )

        trainer = Trainer(model, config)
        result = trainer.train(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
        )

        # Update feature type
        result.feature_type = feature_type
        results.append(result)

    return results
