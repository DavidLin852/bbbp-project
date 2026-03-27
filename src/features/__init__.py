"""
Feature extraction module.

This module handles:
- Molecular fingerprint generation (ECFP, MACCS, etc.)
- Physicochemical descriptor computation
- Graph representation for GNN models
- Feature combination and normalization
"""

from .fingerprints import FingerprintGenerator
from .descriptors import DescriptorGenerator
from .graph import GraphGenerator

__all__ = ["FingerprintGenerator", "DescriptorGenerator", "GraphGenerator"]
