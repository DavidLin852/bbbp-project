"""
Fine-tuning experiment configuration and matrix definition (Stage 4).

Defines:
- 17 pretrain experiments from ZINC22
- Experiment matrix across 4 groups (A: GNN fine-tune, B: Embedding+LGBM, C: Embed+Feat+LGBM, D: Transformer fine-tune)
- Architecture parameter resolution

Usage:
    from src.finetune.finetune_config import build_experiment_matrix, get_pretrain_config

    # Phase 1: Groups A + B + D
    phase1 = build_experiment_matrix(phase=1)

    # Phase 2: Group C
    phase2 = build_experiment_matrix(phase=2)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

# ============================================================
# Pretrain Experiment Registry
# ============================================================
# Maps pretrain experiment ID to architecture + strategy info.
# Loaded from artifacts/models/pretrain/exp_matrix/{id}/

PRETRAIN_EXPERIMENTS: list[dict] = [
    # --- Property Prediction (GIN) ---
    {
        "id": "P_E10_GIN_100K",
        "strategy": "property",
        "model_type": "gin",
        "hidden_dim": 128,
        "num_layers": 3,
        "heads": 4,
        "samples": 100_000,
        "epochs": 10,
    },
    {
        "id": "P_E10_GIN_500K",
        "strategy": "property",
        "model_type": "gin",
        "hidden_dim": 128,
        "num_layers": 3,
        "heads": 4,
        "samples": 500_000,
        "epochs": 10,
    },
    {
        "id": "P_E10_GIN_1M",
        "strategy": "property",
        "model_type": "gin",
        "hidden_dim": 128,
        "num_layers": 3,
        "heads": 4,
        "samples": 1_000_000,
        "epochs": 10,
    },
    {
        "id": "P_E10_GIN_2M",
        "strategy": "property",
        "model_type": "gin",
        "hidden_dim": 128,
        "num_layers": 3,
        "heads": 4,
        "samples": 2_000_000,
        "epochs": 10,
    },
    {
        "id": "P_E10_GIN_5M",
        "strategy": "property",
        "model_type": "gin",
        "hidden_dim": 128,
        "num_layers": 3,
        "heads": 4,
        "samples": 5_000_000,
        "epochs": 10,
    },
    # --- Property Prediction (GAT) ---
    {
        "id": "P_E10_GAT_1M",
        "strategy": "property",
        "model_type": "gat",
        "hidden_dim": 128,
        "num_layers": 3,
        "heads": 4,
        "samples": 1_000_000,
        "epochs": 10,
    },
    # --- Property Prediction (GIN, flagship) ---
    {
        "id": "P_E20_GIN_256_5M",
        "strategy": "property",
        "model_type": "gin",
        "hidden_dim": 256,
        "num_layers": 3,
        "heads": 4,
        "samples": 5_000_000,
        "epochs": 20,
    },
    # --- Denoising (GIN) ---
    {
        "id": "D_E10_GIN_100K",
        "strategy": "denoising",
        "model_type": "gin",
        "hidden_dim": 128,
        "num_layers": 3,
        "heads": 4,
        "samples": 100_000,
        "epochs": 10,
    },
    {
        "id": "D_E10_GIN_1M",
        "strategy": "denoising",
        "model_type": "gin",
        "hidden_dim": 128,
        "num_layers": 3,
        "heads": 4,
        "samples": 1_000_000,
        "epochs": 10,
    },
    {
        "id": "D_E10_GIN_5M",
        "strategy": "denoising",
        "model_type": "gin",
        "hidden_dim": 128,
        "num_layers": 3,
        "heads": 4,
        "samples": 5_000_000,
        "epochs": 10,
    },
    # --- Denoising (GAT) ---
    {
        "id": "D_E10_GAT_1M",
        "strategy": "denoising",
        "model_type": "gat",
        "hidden_dim": 128,
        "num_layers": 3,
        "heads": 4,
        "samples": 1_000_000,
        "epochs": 10,
    },
    # --- Denoising (GIN, flagship) ---
    {
        "id": "D_E20_GIN_256_5M",
        "strategy": "denoising",
        "model_type": "gin",
        "hidden_dim": 256,
        "num_layers": 3,
        "heads": 4,
        "samples": 5_000_000,
        "epochs": 20,
    },
    # --- Transformer MLM ---
    {
        "id": "T_E10_TRANS_100K",
        "strategy": "transformer_mlm",
        "model_type": "transformer",
        "hidden_dim": 256,
        "num_layers": 4,
        "heads": 8,
        "samples": 100_000,
        "epochs": 10,
    },
    {
        "id": "T_E10_TRANS_1M",
        "strategy": "transformer_mlm",
        "model_type": "transformer",
        "hidden_dim": 256,
        "num_layers": 4,
        "heads": 8,
        "samples": 1_000_000,
        "epochs": 10,
    },
    {
        "id": "T_E10_TRANS_5M",
        "strategy": "transformer_mlm",
        "model_type": "transformer",
        "hidden_dim": 256,
        "num_layers": 4,
        "heads": 8,
        "samples": 5_000_000,
        "epochs": 10,
    },
    {
        "id": "T_E10_TRANS_1M_L6",
        "strategy": "transformer_mlm",
        "model_type": "transformer",
        "hidden_dim": 256,
        "num_layers": 6,
        "heads": 8,
        "samples": 1_000_000,
        "epochs": 10,
    },
    {
        "id": "T_E20_TRANS_512_5M",
        "strategy": "transformer_mlm",
        "model_type": "transformer",
        "hidden_dim": 512,
        "num_layers": 8,
        "heads": 12,
        "samples": 5_000_000,
        "epochs": 20,
    },
]

# Classical feature sets for Group C
CLASSICAL_FEATURES: list[str] = [
    "morgan",
    "maccs",
    "fp2",
    "descriptors_basic",
]

# Pretrain ID → index mapping
PRETRAIN_ID_TO_INDEX = {e["id"]: i for i, e in enumerate(PRETRAIN_EXPERIMENTS)}


def get_pretrain_config(pretrain_id: str) -> dict:
    """Get pretrain experiment config by ID."""
    for exp in PRETRAIN_EXPERIMENTS:
        if exp["id"] == pretrain_id:
            return exp
    raise ValueError(f"Unknown pretrain ID: {pretrain_id}")


def get_backbone_path(pretrain_id: str, model_type: str) -> str:
    """Get the path to pretrained backbone for a given pretrain ID and model type.

    Args:
        pretrain_id: e.g. "P_E10_GIN_1M"
        model_type: "gin", "gat", or "transformer"

    Returns:
        Path string relative to project root
    """
    base = f"artifacts/models/pretrain/exp_matrix/{pretrain_id}"
    if model_type == "transformer":
        return f"{base}/transformer_pretrained_encoder.pt"
    return f"{base}/{model_type}_pretrained_backbone.pt"


def get_tokenizer_path(pretrain_id: str) -> str:
    """Get the path to the tokenizer for a Transformer pretrain experiment."""
    return f"artifacts/models/pretrain/exp_matrix/{pretrain_id}/tokenizer.pkl"


@dataclass
class FinetuneConfig:
    """Fine-tuning hyperparameters."""
    finetune_lr: float = 1e-4
    finetune_epochs: int = 200
    finetune_patience: int = 25
    finetune_weight_decay: float = 1e-5
    finetune_batch_size: int = 64
    lgbm_n_estimators: int = 2000
    lgbm_lr: float = 0.01
    lgbm_num_leaves: int = 64
    lgbm_max_depth: int = -1


# ============================================================
# Experiment Matrix Builder
# ============================================================

def build_experiment_matrix(
    phase: int = 1,
    pretrain_ids: Optional[list[str]] = None,
    groups: Optional[list[str]] = None,
    tasks: Optional[list[str]] = None,
) -> list[dict]:
    """
    Build the fine-tuning experiment matrix.

    Args:
        phase: 1 (Groups A+B+D) or 2 (Group C)
        pretrain_ids: Filter to specific pretrain IDs (default: all)
        groups: Filter to specific groups (default: all for the phase)
        tasks: Filter to specific tasks (default: ["classification", "regression"])

    Returns:
        List of experiment dicts
    """
    if tasks is None:
        tasks = ["classification", "regression"]

    matrix: list[dict] = []

    for exp in PRETRAIN_EXPERIMENTS:
        pid = exp["id"]
        model_type = exp["model_type"]

        if pretrain_ids and pid not in pretrain_ids:
            continue

        # --- Group A: GNN Fine-tuning (pretrained GNN backbone → head) ---
        # Applicable only to GNN backbones (GIN, GAT)
        if (groups is None or "A" in groups) and model_type in ("gin", "gat") and phase == 1:
            backbone_path = get_backbone_path(pid, model_type)
            for task in tasks:
                exp_id = f"A_{pid}_{task}"
                matrix.append({
                    "exp_id": exp_id,
                    "group": "A",
                    "pretrain_id": pid,
                    "model_type": model_type,
                    "strategy": exp["strategy"],
                    "hidden_dim": exp["hidden_dim"],
                    "num_layers": exp["num_layers"],
                    "heads": exp["heads"],
                    "task": task,
                    "feature_type": None,
                    "backbone_path": backbone_path,
                    "phase": 1,
                })

        # --- Group B: Embedding → LightGBM (pretrained backbone → LGBM) ---
        # Applicable to ALL pretrained models (GNN + Transformer)
        if (groups is None or "B" in groups) and phase == 1:
            if model_type in ("gin", "gat"):
                backbone_path = get_backbone_path(pid, model_type)
            else:
                backbone_path = get_backbone_path(pid, "transformer")
            for task in tasks:
                exp_id = f"B_{pid}_{task}"
                matrix.append({
                    "exp_id": exp_id,
                    "group": "B",
                    "pretrain_id": pid,
                    "model_type": model_type,
                    "strategy": exp["strategy"],
                    "hidden_dim": exp["hidden_dim"],
                    "num_layers": exp["num_layers"],
                    "heads": exp["heads"],
                    "task": task,
                    "feature_type": None,
                    "backbone_path": backbone_path,
                    "phase": 1,
                })

        # --- Group C: Embedding + Feature → LightGBM ---
        # Concatenate pretrained embeddings with classical fingerprints/descriptors
        if (groups is None or "C" in groups) and phase == 2:
            if model_type in ("gin", "gat"):
                backbone_path = get_backbone_path(pid, model_type)
            else:
                backbone_path = get_backbone_path(pid, "transformer")
            for task in tasks:
                for feature in CLASSICAL_FEATURES:
                    exp_id = f"C_{pid}_{feature}_{task}"
                    matrix.append({
                        "exp_id": exp_id,
                        "group": "C",
                        "pretrain_id": pid,
                        "model_type": model_type,
                        "strategy": exp["strategy"],
                        "hidden_dim": exp["hidden_dim"],
                        "num_layers": exp["num_layers"],
                        "heads": exp["heads"],
                        "task": task,
                        "feature_type": feature,
                        "backbone_path": backbone_path,
                        "phase": 2,
                    })

        # --- Group D: Transformer Fine-tuning (pretrained encoder → head) ---
        # Applicable only to Transformer backbones
        if (groups is None or "D" in groups) and model_type == "transformer" and phase == 1:
            backbone_path = get_backbone_path(pid, "transformer")
            tokenizer_path = get_tokenizer_path(pid)
            for task in tasks:
                exp_id = f"D_{pid}_{task}"
                matrix.append({
                    "exp_id": exp_id,
                    "group": "D",
                    "pretrain_id": pid,
                    "model_type": model_type,
                    "strategy": exp["strategy"],
                    "hidden_dim": exp["hidden_dim"],
                    "num_layers": exp["num_layers"],
                    "heads": exp["heads"],
                    "task": task,
                    "feature_type": None,
                    "backbone_path": backbone_path,
                    "tokenizer_path": tokenizer_path,
                    "phase": 1,
                })

    return matrix


def count_experiments(phase: int = 1, groups: Optional[list[str]] = None) -> dict:
    """Count experiments per group for a phase."""
    matrix = build_experiment_matrix(phase=phase, groups=groups)
    counts = {"total": len(matrix)}
    for exp in matrix:
        g = exp["group"]
        counts[g] = counts.get(g, 0) + 1
    return counts
