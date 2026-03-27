"""
Baseline pipeline configuration for BBB permeability prediction.

This module contains all configuration classes for the working baseline pipeline:
- Dataset configuration (B3DB loading and filtering)
- Split configuration (train/val/test ratios)
- Feature configuration (fingerprints and descriptors)

These configurations are used by:
- scripts/baseline/*.py (core pipeline scripts)
- src/data/*.py (data loading and preprocessing)
- src/features/*.py (feature extraction)
- src/models/*.py (baseline models)
- src/train/*.py (training utilities)
- src/evaluate/*.py (evaluation utilities)

Usage:
    from src.config.baseline import DatasetConfig, SplitConfig, FeatureConfig

    # Use defaults
    dataset_config = DatasetConfig()
    split_config = SplitConfig()
    feature_config = FeatureConfig()

    # Or customize
    dataset_config = DatasetConfig(group_keep=("A", "B", "C"))
    split_config = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
"""

from dataclasses import dataclass, field


# ==================== Dataset Configuration ====================

@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuration for B3DB dataset loading and filtering.

    Attributes:
        filename: Name of B3DB file (in data/raw/)
        group_keep: Which groups to keep ("A", "B", "C", "D")
            - "A": High precision, 87.7% BBB+ (n=846)
            - "A,B": Best balance, 76.5% BBB+ (n=3,743) ⭐ DEFAULT
            - "A,B,C": Large scale, 66.7% BBB+ (n=6,203)
            - "A,B,C,D": Maximum coverage, 63.5% BBB+ (n=6,244)
        smiles_col: Column name for SMILES strings
        group_col: Column name for group labels (A/B/C/D)
        bbb_col: Column name for BBB permeability label
        logbb_col: Column name for logBB values (regression)
        id_cols: Columns that identify molecules (not features)
    """

    filename: str = "B3DB_classification.tsv"
    group_keep: tuple[str, ...] = ("A", "B")
    smiles_col: str = "SMILES"
    group_col: str = "group"
    bbb_col: str = "BBB+/BBB-"
    logbb_col: str = "logBB"
    id_cols: tuple[str, ...] = ("NO.", "CID", "compound_name")


# ==================== Split Configuration ====================

@dataclass(frozen=True)
class SplitConfig:
    """
    Configuration for train/val/test split ratios.

    The split is performed using scaffold-based splitting to ensure
    structural diversity between train, validation, and test sets.

    Attributes:
        train_ratio: Proportion of data for training (default: 0.8)
        val_ratio: Proportion of data for validation (default: 0.1)
        test_ratio: Proportion of data for testing (default: 0.1)

    Note:
        Ratios should sum to 1.0. If they don't, they will be normalized.
    """

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


# ==================== Feature Configuration ====================

@dataclass(frozen=True)
class FingerprintConfig:
    """
    Configuration for molecular fingerprint extraction.

    Attributes:
        morgan_bits: Number of bits for Morgan/ECFP4 fingerprints
        morgan_radius: Radius for Morgan fingerprints (2 = ECFP4)
        maccs_bits: Number of bits for MACCS keys (fixed at 167)
        atom_pairs_bits: Number of bits for atom pair fingerprints
        atom_pairs_max_dist: Maximum distance for atom pairs
        fp2_bits: Number of bits for FP2/Daylight fingerprints
    """

    # Morgan fingerprint (ECFP4-like)
    morgan_bits: int = 2048
    morgan_radius: int = 2

    # MACCS keys
    maccs_bits: int = 167

    # Atom pairs
    atom_pairs_bits: int = 1024
    atom_pairs_max_dist: int = 3

    # FP2 (Daylight-like)
    fp2_bits: int = 2048


@dataclass(frozen=True)
class DescriptorConfig:
    """
    Configuration for physicochemical descriptor extraction.

    Attributes:
        set: Which descriptor set to use
            - "basic": 13 fundamental properties (MW, LogP, TPSA, etc.)
            - "extended": ~30 properties (includes electronic properties)
            - "all": ~200 properties (all RDKit descriptors)
        normalize: Whether to normalize descriptors to zero mean and unit variance
    """

    set: str = "all"  # "basic", "extended", "all"
    normalize: bool = True


@dataclass(frozen=True)
class FeatureConfig:
    """
    Combined feature configuration for the baseline pipeline.

    This class combines fingerprint and descriptor configurations, with
    flags to enable/disable specific feature types.

    Attributes:
        fingerprint: Fingerprint configuration
        descriptor: Descriptor configuration
        use_morgan: Include Morgan fingerprints (ECFP4)
        use_maccs: Include MACCS keys
        use_atom_pairs: Include atom pair fingerprints
        use_fp2: Include FP2 fingerprints
        use_descriptors: Include physicochemical descriptors

    Note:
        Morgan fingerprints are recommended for best performance (0.9401 AUC).
    """

    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    descriptor: DescriptorConfig = field(default_factory=DescriptorConfig)

    # Feature flags (which features to compute)
    use_morgan: bool = True
    use_maccs: bool = True
    use_atom_pairs: bool = True
    use_fp2: bool = True
    use_descriptors: bool = True
