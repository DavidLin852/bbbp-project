"""
Factory for creating model instances.
"""

from typing import Literal

from .baseline_models import BaselineModel, ModelConfig


class ModelFactory:
    """
    Factory for creating model instances.

    Provides convenient interface for instantiating
    models with consistent configuration.
    """

    def __init__(self, config: ModelConfig | None = None):
        """
        Initialize factory.

        Args:
            config: Default model configuration
        """
        self.config = config or ModelConfig()

    def create_model(
        self,
        model_type: Literal[
            "rf", "xgb", "lgbm", "svm", "knn", "lr", "nb", "gb", "ada", "etc"
        ],
        config: ModelConfig | None = None,
    ) -> BaselineModel:
        """
        Create a model instance.

        Args:
            model_type: Type of model to create
            config: Optional model-specific config

        Returns:
            BaselineModel instance
        """
        cfg = config or self.config
        return BaselineModel(model_type, cfg)

    def create_rf(self, config: ModelConfig | None = None) -> BaselineModel:
        """Create Random Forest model."""
        return self.create_model("rf", config)

    def create_xgb(self, config: ModelConfig | None = None) -> BaselineModel:
        """Create XGBoost model."""
        return self.create_model("xgb", config)

    def create_lgbm(self, config: ModelConfig | None = None) -> BaselineModel:
        """Create LightGBM model."""
        return self.create_model("lgbm", config)

    def create_svm(self, config: ModelConfig | None = None) -> BaselineModel:
        """Create SVM model."""
        return self.create_model("svm", config)

    def create_lr(self, config: ModelConfig | None = None) -> BaselineModel:
        """Create Logistic Regression model."""
        return self.create_model("lr", config)

    @staticmethod
    def get_available_models() -> list[str]:
        """Get list of available model types."""
        return ["rf", "xgb", "lgbm", "svm", "lr", "knn", "nb", "gb", "ada", "etc"]
