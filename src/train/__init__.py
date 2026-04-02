"""
Training module.

This module handles:
- Training baseline models
- Model evaluation
- Result logging
"""

from .trainer import Trainer, TrainingConfig, train_multiple_models, train_multiple_regression_models

__all__ = ["Trainer", "TrainingConfig", "train_multiple_models", "train_multiple_regression_models"]
