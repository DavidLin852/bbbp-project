"""
Model comparison utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import pandas as pd

from ..train.trainer import TrainingResult, RegTrainingResult


@dataclass
class ModelComparison:
    """
    Container for model comparison results.

    Stores results from multiple model runs
    and provides comparison methods.
    """

    results: list[TrainingResult]

    def __len__(self):
        return len(self.results)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Returns:
            DataFrame with one row per model
        """
        data = [r.to_dict() for r in self.results]
        return pd.DataFrame(data)

    def sort_by_test_auc(self, ascending: bool = False) -> pd.DataFrame:
        """
        Sort models by test AUC.

        Args:
            ascending: Sort order

        Returns:
            Sorted DataFrame
        """
        df = self.to_dataframe()
        return df.sort_values("test_auc", ascending=ascending)

    def get_best_model(self) -> TrainingResult:
        """
        Get model with best test AUC.

        Returns:
            TrainingResult for best model
        """
        return max(self.results, key=lambda r: r.test_metrics.auc)

    def summary(self) -> dict:
        """
        Get summary statistics.

        Returns:
            Dictionary with summary statistics
        """
        df = self.to_dataframe()

        return {
            "n_models": len(self.results),
            "best_test_auc": float(df["test_auc"].max()),
            "best_model_name": df.loc[df["test_auc"].idxmax(), "model_name"],
            "mean_test_auc": float(df["test_auc"].mean()),
            "std_test_auc": float(df["test_auc"].std()),
        }

    def save(self, output_path: Path | str):
        """
        Save comparison results to JSON.

        Args:
            output_path: Path to save JSON
        """
        output_path = Path(output_path)

        data = {
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary(),
        }

        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, input_path: Path | str) -> "ModelComparison":
        """
        Load comparison results from JSON.

        Args:
            input_path: Path to load JSON from

        Returns:
            ModelComparison instance
        """
        input_path = Path(input_path)
        data = json.loads(input_path.read_text(encoding="utf-8"))

        # Recreate TrainingResult objects
        from ..train.trainer import TrainingResult
        from ..utils.metrics import ClsMetrics

        results = []
        for r_dict in data["results"]:
            result = TrainingResult(
                model_name=r_dict["model_name"],
                feature_type=r_dict["feature_type"],
                train_metrics=ClsMetrics(**{
                    k.replace("train_", ""): v
                    for k, v in r_dict.items()
                    if k.startswith("train_") and k != "train_metrics"
                }),
                val_metrics=ClsMetrics(**{
                    k.replace("val_", ""): v
                    for k, v in r_dict.items()
                    if k.startswith("val_") and k != "val_metrics"
                }),
                test_metrics=ClsMetrics(**{
                    k.replace("test_", ""): v
                    for k, v in r_dict.items()
                    if k.startswith("test_") and k != "test_metrics"
                }),
            )
            results.append(result)

        return cls(results=results)


def compare_models(results: list[TrainingResult]) -> ModelComparison:
    """
    Create model comparison from results.

    Args:
        results: List of training results

    Returns:
        ModelComparison instance
    """
    return ModelComparison(results=results)


@dataclass
class RegModelComparison:
    """
    Container for regression model comparison results.
    """

    results: list[RegTrainingResult]

    def __len__(self):
        return len(self.results)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        data = [r.to_dict() for r in self.results]
        return pd.DataFrame(data)

    def sort_by_test_r2(self, ascending: bool = False) -> pd.DataFrame:
        """Sort models by test R2 (higher is better)."""
        df = self.to_dataframe()
        return df.sort_values("test_r2", ascending=ascending)

    def get_best_model(self) -> RegTrainingResult:
        """Get model with best test R2."""
        return max(self.results, key=lambda r: r.test_metrics.r2)

    def summary(self) -> dict:
        """Get summary statistics."""
        df = self.to_dataframe()
        return {
            "n_models": len(self.results),
            "best_test_r2": float(df["test_r2"].max()),
            "best_model_name": df.loc[df["test_r2"].idxmax(), "model_name"],
            "mean_test_r2": float(df["test_r2"].mean()),
            "std_test_r2": float(df["test_r2"].std()),
        }

    def save(self, output_path: Path | str):
        """Save comparison results to JSON."""
        output_path = Path(output_path)
        data = {
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary(),
        }
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def compare_regression_models(results: list[RegTrainingResult]) -> RegModelComparison:
    """Create regression model comparison from results."""
    return RegModelComparison(results=results)
