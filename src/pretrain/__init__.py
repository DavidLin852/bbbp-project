"""
Pretraining Module for ZINC22

Implements pretraining infrastructure for graph and sequence models.

New Implementation (Checkpoint 1):
- data.py: ZINC22 data pipeline
- graph.py: Graph pretraining (GIN focus, extensible to GAT)
- smiles.py: SMILES Transformer pretraining (MLM)

Legacy (Preserved):
- zinc20_loader.py: Original ZINC20 loader
- zinc20_pretrain.py: Original pretraining code
- Other files: Preserved for reference

Usage:
    # Graph pretraining
    python scripts/pretrain/pretrain_graph.py --num_samples 10000 --epochs 5

    # SMILES pretraining
    python scripts/pretrain/pretrain_smiles.py --num_samples 10000 --epochs 5

    # Fine-tuning on B3DB
    python scripts/pretrain/finetune_graph.py --pretrained_path <path>
"""

from .data import (
    ZINC22Dataset,
    ZINC22PretrainDataset,
    create_zinc22_dataloader,
    create_small_zinc22_sample,
    count_zinc22_molecules,
)

from .graph import (
    GraphPretrainer,
    pretrain_graph_model,
)

from .smiles import (
    SMILESPretrainer,
    SMILESMasking,
    pretrain_smiles_model,
)

__all__ = [
    # Data
    "ZINC22Dataset",
    "ZINC22PretrainDataset",
    "create_zinc22_dataloader",
    "create_small_zinc22_sample",
    "count_zinc22_molecules",
    # Graph
    "GraphPretrainer",
    "pretrain_graph_model",
    # SMILES
    "SMILESPretrainer",
    "SMILESMasking",
    "pretrain_smiles_model",
]
