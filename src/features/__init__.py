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


def __getattr__(name):
    """Lazy import for heavy dependencies (torch, PyG)."""
    if name == "GraphGenerator":
        from .graph import GraphGenerator
        return GraphGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FingerprintGenerator", "DescriptorGenerator", "GraphGenerator"]
