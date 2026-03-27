"""
Evaluation module.

This module handles:
- Computing and aggregating metrics
- Comparing models
- Generating evaluation reports
"""

from .comparison import ModelComparison, compare_models
from .report import generate_report

__all__ = ["ModelComparison", "compare_models", "generate_report"]
