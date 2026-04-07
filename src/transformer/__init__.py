"""Transformer module for molecular property prediction.

Supports both:
- Fingerprint-based classification (legacy, transformer_model.py)
- SMILES sequence-based classification/regression (new, smiles_transformer.py)
"""

from .transformer_model import (
    TransformerClassifier,
    TransformerEncoder,
    TransformerEncoderLayer,
    PositionalEncoding,
    FingerprintEmbedding,
    TransformerTrainingConfig,
    FocalLoss,
    train_transformer,
    evaluate_transformer,
    get_criterion
)

from .smiles_tokenizer import (
    SMILESTokenizer,
    create_tokenizer_from_data,
    collate_smiles_batch
)

from .smiles_transformer import (
    SMILESTransformerEncoder,
    SMILESTransformerClassifier,
    SMILESTransformerRegressor,
    TransformerConfig,
    get_model
)

from .trainer import (
    SMILESDataset,
    Trainer,
    evaluate_model
)

__all__ = [
    # Legacy fingerprint-based
    'TransformerClassifier',
    'TransformerEncoder',
    'TransformerEncoderLayer',
    'PositionalEncoding',
    'FingerprintEmbedding',
    'TransformerTrainingConfig',
    'FocalLoss',
    'train_transformer',
    'evaluate_transformer',
    'get_criterion',

    # New SMILES-based
    'SMILESTokenizer',
    'create_tokenizer_from_data',
    'collate_smiles_batch',
    'SMILESTransformerEncoder',
    'SMILESTransformerClassifier',
    'SMILESTransformerRegressor',
    'TransformerConfig',
    'get_model',
    'SMILESDataset',
    'Trainer',
    'evaluate_model'
]
