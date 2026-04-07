#!/usr/bin/env python
"""
Fine-tune Pretrained Graph Models on B3DB

This script loads a pretrained GIN/GAT backbone and fine-tunes it on B3DB.

Usage:
    # Fine-tune pretrained GIN on B3DB classification
    python scripts/pretrain/finetune_graph.py \\
        --pretrained_path artifacts/models/pretrain/graph/gin_pretrained_backbone.pt \\
        --model_type gin \\
        --task classification \\
        --seeds 0,1,2,3,4
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gnn.models import GNNClassifier, GNNRegressor
from src.gnn.run_gnn_benchmark import (
    load_data_splits,
    get_device,
    GraphDataset,
    collate_graphs,
)


def load_pretrained_backbone(
    model_type: str,
    pretrained_path: str,
    node_dim: int = 22,
    hidden_dim: int = 128,
    num_layers: int = 3,
    device: torch.device = torch.device("cpu"),
):
    """
    Load pretrained backbone and wrap in fine-tuning model.

    Args:
        model_type: Model type ("gin" or "gat")
        pretrained_path: Path to pretrained backbone
        node_dim: Node feature dimension
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        device: Device to load model on

    Returns:
        Model with pretrained backbone
    """
    from src.gnn.models import GIN, GAT

    # Create backbone architecture
    if model_type == "gin":
        backbone = GIN(
            in_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.1,
        )
    elif model_type == "gat":
        backbone = GAT(
            in_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load pretrained weights
    state_dict = torch.load(pretrained_path, map_location=device)
    backbone.load_state_dict(state_dict)

    print(f"Loaded pretrained {model_type.upper()} backbone from {pretrained_path}")

    return backbone


def finetune_on_b3db(
    pretrained_path: str,
    model_type: str,
    task: str,
    seeds: list,
    data_dir: str,
    save_dir: str,
    device: str = "auto",
):
    """
    Fine-tune pretrained model on B3DB.

    Args:
        pretrained_path: Path to pretrained backbone
        model_type: Model type
        task: "classification" or "regression"
        seeds: List of seeds
        data_dir: Data directory
        save_dir: Save directory
        device: Device to use
    """
    device = get_device(device)

    print("=" * 60)
    print(f"Fine-tuning Pretrained {model_type.upper()} on B3DB")
    print("=" * 60)
    print(f"Pretrained: {pretrained_path}")
    print(f"Task: {task}")
    print(f"Seeds: {seeds}")
    print(f"Device: {device}")
    print("=" * 60)

    # Results storage
    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        # Load data
        train_df, val_df, test_df = load_data_splits(
            Path(data_dir), "scaffold", seed, task
        )

        # Create datasets
        train_dataset = GraphDataset(train_df)
        val_dataset = GraphDataset(val_df)
        test_dataset = GraphDataset(test_df)

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_graphs,
            num_workers=0,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=collate_graphs,
            num_workers=0,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=collate_graphs,
            num_workers=0,
        )

        # Load pretrained backbone
        backbone = load_pretrained_backbone(
            model_type=model_type,
            pretrained_path=pretrained_path,
            device=device,
        )

        # Create task-specific head
        if task == "classification":
            model = GNNClassifier(
                backbone=backbone,
                hidden_dim=128,
                dropout=0.1,
            ).to(device)
        else:
            model = GNNRegressor(
                backbone=backbone,
                hidden_dim=128,
                dropout=0.1,
            ).to(device)

        # Fine-tuning settings
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4,  # Lower LR for fine-tuning
            weight_decay=1e-5,
        )

        # Loss
        if task == "classification":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()

        # Training loop (simplified - adapt from run_gnn_benchmark.py)
        # TODO: Copy full training loop from run_gnn_benchmark.py
        print("\nFine-tuning training loop to be implemented...")
        print("Adapt from scripts/gnn/run_gnn_benchmark.py")

        # For now, just evaluate pretrained backbone
        print("\nEvaluating pretrained backbone (without fine-tuning):")
        # TODO: Add evaluation

        all_results.append({"seed": seed, "task": task, "status": "pending"})

    # Save results
    import pandas as pd
    results_df = pd.DataFrame(all_results)
    results_path = Path(save_dir) / f"{model_type}_pretrained_finetuned_{task}.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nResults saved to: {results_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained graph models on B3DB"
    )

    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to pretrained backbone checkpoint",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gin",
        choices=["gin", "gat"],
        help="Model type",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "regression"],
        help="Task to fine-tune on",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Seeds to evaluate",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/splits",
        help="Data directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="artifacts/models/pretrain/finetuned",
        help="Save directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    finetune_on_b3db(
        pretrained_path=args.pretrained_path,
        model_type=args.model_type,
        task=args.task,
        seeds=seeds,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
