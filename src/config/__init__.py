"""
Configuration module for BBB permeability prediction project.

This module provides a unified configuration interface for the entire project,
while internally separating baseline pipeline configuration from research module
configuration.

Usage:
    # For baseline pipeline (RECOMMENDED for most use cases)
    from src.config import Paths, DatasetConfig, SplitConfig

    # For research modules (advanced use)
    from src.config import get_config
    config = get_config(module="baseline")  # or "research"
    paths = config.paths
    dataset = config.dataset
"""

from src.config.paths import Paths
from src.config.baseline import DatasetConfig, SplitConfig, FeatureConfig
from src.config.research import (
    TransformerConfig,
    VAEConfig,
    VAETrainConfig,
    GANConfig,
    GANTrainConfig,
    GenerationConfig,
    StackingConfig,
    SHAPConfig,
)

# For backward compatibility, export all baseline configs at module level
__all__ = [
    # Paths (shared)
    "Paths",
    # Baseline pipeline configs
    "DatasetConfig",
    "SplitConfig",
    "FeatureConfig",
    # Research configs (future use)
    "TransformerConfig",
    "VAEConfig",
    "VAETrainConfig",
    "GANConfig",
    "GANTrainConfig",
    "GenerationConfig",
    "StackingConfig",
    "SHAPConfig",
]
