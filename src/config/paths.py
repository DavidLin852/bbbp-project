"""
Path configuration for BBB permeability prediction project.

This module defines all file system paths used throughout the project.
Paths are organized by purpose (data, artifacts, outputs) and are computed
relative to the project root.

This is a shared module used by both baseline and research code.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """
    File system paths for the BBB prediction project.

    All paths are computed relative to the project root directory.
    The root is automatically detected as the parent directory of 'src/'.

    Attributes:
        root: Project root directory
        data_raw: Raw data directory (contains B3DB datasets)
        data_splits: Generated data splits (train/val/test)
        data_external: External datasets (not used by baseline)
        artifacts: General artifacts directory
        features: Computed features cache
        models: Trained models directory
        metrics: Old metrics directory (deprecated, use reports/)
        figures: Generated figures directory
        logs: Training and evaluation logs
        reports: Benchmark and analysis reports
    """

    root: Path = Path(__file__).resolve().parents[2]

    # Data paths
    data_raw: Path = root / "data" / "raw"
    data_splits: Path = root / "data" / "splits"
    data_external: Path = root / "data" / "external"

    # Artifacts paths
    artifacts: Path = root / "artifacts"
    features: Path = artifacts / "features"
    models: Path = artifacts / "models"

    # Output paths (organized by type)
    metrics: Path = artifacts / "metrics"  # Deprecated: Use reports/
    figures: Path = artifacts / "figures"
    logs: Path = artifacts / "logs"
    reports: Path = artifacts / "reports"  # Primary output for benchmark results
