"""
Backward compatibility module for configuration.

This module maintains backward compatibility with existing code that imports
from src.config. It re-exports all configuration classes from the new
modular structure (src.config.paths, src.config.baseline, src.config.research).

DEPRECATED: For new code, prefer importing from specific modules:
    - from src.config.paths import Paths
    - from src.config.baseline import DatasetConfig, SplitConfig, FeatureConfig
    - from src.config.research import VAEConfig, GANConfig, etc.

For backward compatibility, all imports continue to work:
    from src.config import Paths, DatasetConfig, SplitConfig  # Still works ✅
"""

# Re-export everything from the new modular structure
from src.config.paths import Paths
from src.config.baseline import (
    DatasetConfig,
    SplitConfig,
    FingerprintConfig,
    DescriptorConfig,
    FeatureConfig,
)
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

# For backward compatibility, also export old names that were in the original config
__all__ = [
    # Paths
    "Paths",
    # Baseline configs
    "DatasetConfig",
    "SplitConfig",
    "FeaturizeConfig",  # Deprecated: Use FeatureConfig instead
    "FingerprintConfig",
    "DescriptorConfig",
    "FeatureConfig",
    # Research configs (future work)
    "TransformerConfig",
    "VAEConfig",
    "VAETrainConfig",
    "GANConfig",
    "GANTrainConfig",
    "GenerationConfig",
    "StackingConfig",
    "SHAPConfig",
]


# ==================== Deprecated Aliases ====================

# FeaturizeConfig was renamed to FeatureConfig for clarity
# Keep this alias for backward compatibility
FeaturizeConfig = FeatureConfig
