"""
Graph Pretraining Module for GIN

Implements graph-based pretraining on ZINC22.
Target: GIN as primary model (extensible to GAT later).

Pretraining Tasks:
1. Graph property prediction (regression)
2. Graph masking (node feature reconstruction)

Usage:
    # Pretrain GIN on ZINC22
    python scripts/pretrain/pretrain_graph.py \\
        --representation graph \\
        --model gin \\
        --task property_prediction \\
        --num_samples 10000 \\
        --epochs 10
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn.models import GIN, GAT
from pretrain.data import ZINC22Dataset, create_zinc22_dataloader


# ==================== Pretraining Models ====================

class GraphPretrainer(nn.Module):
    """
    Wrapper for GNN models during pretraining.

    Supports different pretraining tasks:
    - Property prediction (regression)
    - Graph masking (reconstruction)
    """

    def __init__(
        self,
        model_type: Literal["gin", "gat"] = "gin",
        node_dim: int = 22,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        pretraining_task: Literal["property_prediction", "masking"] = "property_prediction",
        num_properties: int = 4,
    ):
        super().__init__()

        self.model_type = model_type
        self.pretraining_task = pretraining_task
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # Create backbone GNN
        if model_type == "gin":
            self.backbone = GIN(
                in_dim=node_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif model_type == "gat":
            self.backbone = GAT(
                in_dim=node_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Pretraining head
        if pretraining_task == "property_prediction":
            # Predict molecular properties
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_properties),
            )
        elif pretraining_task == "masking":
            # Reconstruct masked node features
            self.head = nn.Linear(hidden_dim, node_dim)
        else:
            raise ValueError(f"Unknown pretraining task: {pretraining_task}")

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: PyG batch object

        Returns:
            Predictions
        """
        # Get graph embeddings
        graph_emb = self.backbone(batch)  # (batch_size, hidden_dim)

        if self.pretraining_task == "property_prediction":
            # Predict properties from graph embedding
            return self.head(graph_emb)

        elif self.pretraining_task == "masking":
            # Reconstruct node features
            # For now, use graph embedding to predict all node features (simplified)
            batch_size = batch.num_graphs
            num_nodes = batch.x.shape[0]

            # Expand graph embedding to match number of nodes
            graph_emb_expanded = graph_emb[batch.batch]  # (num_nodes, hidden_dim)
            reconstructed = self.head(graph_emb_expanded)  # (num_nodes, node_dim)

            return reconstructed

        else:
            raise ValueError(f"Unknown pretraining task: {self.pretraining_task}")


# ==================== Pretraining Functions ====================

def compute_zinc_properties(smiles: str) -> Optional[torch.Tensor]:
    """
    Compute molecular properties for pretraining targets.

    Returns:
        Tensor of shape (4,) with [logP, TPSA, MW, num_rotatable_bonds]
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, rdMolDescriptors, Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        mw = Descriptors.ExactMolWt(mol)
        num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)

        return torch.tensor([logp, tpsa, mw, num_rotatable], dtype=torch.float32)

    except:
        return None


class PropertyPredictionDataset(torch.utils.data.Dataset):
    """
    Dataset for graph property prediction pretraining.
    """

    def __init__(self, smiles_list: list):
        self.smiles_list = smiles_list
        self.properties = []

        # Precompute properties
        print("Computing molecular properties...")
        for smiles in tqdm(smiles_list):
            props = compute_zinc_properties(smiles)
            if props is not None:
                self.properties.append(props)
            else:
                # Use zero padding for invalid molecules
                self.properties.append(torch.zeros(4))

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        from features.graph import smiles_to_pyg_graph

        smiles = self.smiles_list[idx]
        graph = smiles_to_pyg_graph(smiles)
        props = self.properties[idx]

        return graph, props


def pretrain_graph_model(
    data_dir: str | Path,
    model_type: Literal["gin", "gat"] = "gin",
    pretraining_task: Literal["property_prediction"] = "property_prediction",
    num_samples: int = 10000,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 3,
    save_dir: str | Path = "artifacts/models/pretrain",
    device: str = "auto",
) -> dict:
    """
    Pretrain a graph model on ZINC22.

    Args:
        data_path: Path to ZINC22 SMILES file
        model_type: Model type ("gin" or "gat")
        pretraining_task: Pretraining task
        num_samples: Number of samples to use
        batch_size: Batch size
        epochs: Number of training epochs
        lr: Learning rate
        hidden_dim: Hidden dimension
        num_layers: Number of GNN layers
        save_dir: Directory to save checkpoints
        device: Device to use

    Returns:
        Training history
    """
    # Device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading {num_samples} samples from {data_dir}")
    dataset = ZINC22Dataset(
        data_dir=data_dir,
        representation="graph",
        num_samples=num_samples,
    )

    # Wrap for property prediction
    if pretraining_task == "property_prediction":
        dataset = PropertyPredictionDataset(dataset.smiles_list)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: PyGBatch(batch),
    )

    # Create model
    model = GraphPretrainer(
        model_type=model_type,
        node_dim=22,  # Fixed node feature dimension
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
        pretraining_task=pretraining_task,
        num_properties=4,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Loss function
    if pretraining_task == "property_prediction":
        criterion = nn.MSELoss()
    else:
        criterion = nn.MSELoss()

    # Training loop
    history = {"train_loss": []}

    print(f"Starting pretraining: {epochs} epochs")
    print(f"Model: {model_type.upper()}, Task: {pretraining_task}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            if pretraining_task == "property_prediction":
                batch = batch.to(device)
                graphs = batch.batch
                targets = batch.targets

                # Forward
                predictions = model(graphs)
                loss = criterion(predictions, targets)

            else:
                raise NotImplementedError(f"Task not implemented: {pretraining_task}")

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / num_batches
        history["train_loss"].append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": {
                "model_type": model_type,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "pretraining_task": pretraining_task,
            },
        }

        torch.save(checkpoint, save_dir / f"{model_type}_pretrain_epoch_{epoch}.pt")

    # Save final model
    torch.save(model.backbone.state_dict(), save_dir / f"{model_type}_pretrained_backbone.pt")

    print(f"\nPretraining complete!")
    print(f"Saved pretrained backbone to: {save_dir / f'{model_type}_pretrained_backbone.pt'}")

    return history


# ==================== Helper Classes ====================

class PyGBatch:
    """
    Custom batch collation for PyG Data objects with property targets.
    """

    def __init__(self, batch_list):
        from torch_geometric.data import Batch

        # Separate graphs and targets
        graphs = [item[0] for item in batch_list]
        targets = torch.stack([item[1] for item in batch_list])

        # Batch graphs
        self.batch = Batch.from_data_list(graphs)
        self.targets = targets

    def to(self, device):
        self.batch = self.batch.to(device)
        self.targets = self.targets.to(device)
        return self

    def __len__(self):
        return len(self.targets)
