"""
Model definitions module.

This module contains:
- Classical ML models (RF, XGBoost, LightGBM, etc.)
- GNN models (GAT, GCN, etc.)
- Model configurations
"""

from .baseline_models import BaselineModel, ModelConfig
from .model_factory import ModelFactory

__all__ = ["BaselineModel", "ModelConfig", "ModelFactory"]
