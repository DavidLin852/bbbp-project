"""
Strategy 5: Context Prediction (Subgraph-Context Matching)

Based on Hu et al. 2019 "Strategies for Pre-training Graph Neural Networks".

Key idea: For each molecule, extract a k-hop subgraph centered on a node,
and the surrounding context (nodes at distance k+1 to k+r).
Train the GNN to predict whether a subgraph and context pair are from the same molecule.

Simplified version: predict whether two nodes belong to the same subgraph context.
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch_geometric.data import Batch
from tqdm import tqdm
import numpy as np
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn.models import GIN, GAT, GCN
from pretrain.data import ZINC22Dataset
from features.graph import smiles_to_pyg_graph


# ==================== Context Extraction ====================

def extract_context_pairs(data, k: int = 2, num_negatives: int = 5):
    """
    Extract positive and negative (node, context) pairs from a molecule.

    Simplified approach: for each node v, the "context" is the k-hop neighborhood.
    Two nodes from the same neighborhood form a positive pair.
    Nodes from different graphs form negative pairs.

    Returns:
        positive_pairs: list of (node_emb_1, node_emb_2) indices
        negative_pairs: list of (node_emb_1, node_emb_2) indices
    """
    # For simplicity, we'll use a batch-level approach:
    # Each molecule in the batch is a "graph"
    # Nodes from the same molecule = positive pairs
    # Nodes from different molecules = negative pairs
    pass


# ==================== Context Prediction Head ====================

class ContextPredictor(nn.Module):
    """
    Predicts whether two node embeddings belong to the same context (molecule).

    MLP: Linear(2 * hidden_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h1: (batch_size,) or (N,) node embeddings
            h2: (batch_size,) or (N,) node embeddings

        Returns:
            logits for matching (positive=1, negative=0)
        """
        combined = torch.cat([h1, h2], dim=-1)
        return self.predictor(combined)


# ==================== Dataset ====================

class ContextPredictionDataset(torch.utils.data.Dataset):
    """
    Dataset that generates (anchor, positive, negative) node pairs per molecule.
    """

    def __init__(
        self,
        smiles_list: list,
        cache_file: Optional[Path] = None,
        seed: int = 42,
        k: int = 2,  # k-hop neighborhood size
        num_pairs_per_graph: int = 8,
    ):
        self.smiles_list = smiles_list
        self.data = []
        self.pairs = []  # (graph_idx, node_i, node_j) positive pairs
        self._length = 0
        self.rng = np.random.RandomState(seed)
        self.k = k
        self.num_pairs = num_pairs_per_graph

        if cache_file and cache_file.exists():
            print(f"Loading cached graphs from {cache_file}")
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
                self.data = cached["data"]
                self.pairs = cached["pairs"]
                self._length = len(self.data)
            print(f"Loaded {self._length:,} cached graphs")
            return

        print(f"Building {len(smiles_list):,} graphs...")
        for i, smiles in enumerate(tqdm(smiles_list, desc="Building graphs")):
            try:
                graph = smiles_to_pyg_graph(smiles)
                self.data.append(graph)

                # Generate positive pairs from this graph
                num_nodes = graph.x.shape[0]
                for _ in range(num_pairs_per_graph):
                    ni = self.rng.randint(0, num_nodes)
                    nj = self.rng.randint(0, num_nodes)
                    self.pairs.append((i, ni, nj))
            except Exception:
                continue

        self._length = len(self.data)
        print(f"Built {self._length:,} graphs with {len(self.pairs):,} positive pairs")

        if cache_file:
            with open(cache_file, "wb") as f:
                pickle.dump({"data": self.data, "pairs": self.pairs}, f)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return self.data[idx]


def collate_context_batch(batch_list):
    return Batch.from_data_list([b for b in batch_list])


# ==================== Training ====================

def pretrain_context_prediction(
    data_dir: str | Path,
    num_samples: int = 100000,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    model_type: Literal["gin", "gat", "gcn"] = "gin",
    hidden_dim: int = 128,
    num_layers: int = 3,
    heads: int = 4,
    k: int = 2,
    num_negatives: int = 5,
    num_workers: int = 4,
    save_dir: str | Path = "artifacts/models/pretrain/context_prediction",
    device: str = "auto",
    gradient_accumulation: int = 1,
    log_interval: int = 100,
) -> dict:
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    print(f"\nLoading {num_samples:,} SMILES from {data_dir}")
    zinc_dataset = ZINC22Dataset(data_dir=data_dir, representation="smiles",
                                num_samples=num_samples, shuffle=True)
    smiles_list = zinc_dataset.smiles_list

    cache_file = save_dir / f"graph_cache_{num_samples}.pkl"
    graph_dataset = ContextPredictionDataset(smiles_list=smiles_list, cache_file=cache_file, k=k)
    all_pairs = graph_dataset.pairs

    dataloader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True, drop_last=True)

    # Model
    if model_type == "gin":
        backbone = GIN(in_dim=22, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.1)
    elif model_type == "gat":
        backbone = GAT(in_dim=22, hidden_dim=hidden_dim, num_layers=num_layers, heads=heads, dropout=0.1)
    elif model_type == "gcn":
        backbone = GCN(in_dim=22, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.1)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    predictor = ContextPredictor(hidden_dim=hidden_dim)
    criterion = nn.BCEWithLogitsLoss()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        backbone = DataParallel(backbone)
        predictor = DataParallel(predictor)

    backbone = backbone.to(device)
    predictor = predictor.to(device)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP)")

    history = {"train_loss": []}
    global_step = 0
    pair_rng = np.random.RandomState(42)

    print(f"\nStarting context prediction pretraining: {epochs} epochs, "
          f"{len(dataloader)} steps/epoch, k={k}, num_negatives={num_negatives}")

    for epoch in range(epochs):
        backbone.train()
        predictor.train()
        total_loss = 0.0
        total_pairs = 0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)

            # Get node embeddings
            node_emb = backbone(batch, return_node_emb=True)  # (N, hidden)
            batch_idx = batch.batch  # which graph each node belongs to

            # Create positive pairs: nodes from the same graph
            N = node_emb.shape[0]
            num_pos = min(batch_size * 4, N)  # approximate positive pairs

            pos_loss_sum = 0.0
            neg_loss_sum = 0.0
            num_pos_examples = 0

            # Sample positive pairs: pick nodes from the same graph
            for g in batch_idx.unique():
                nodes_in_g = (batch_idx == g).nonzero(as_tuple=True)[0]
                n_nodes = len(nodes_in_g)
                if n_nodes < 2:
                    continue

                # Sample random pairs from same graph (positive)
                for _ in range(min(n_nodes, 8)):
                    i, j = pair_rng.choice(nodes_in_g.tolist(), 2, replace=False)
                    h1, h2 = node_emb[i], node_emb[j]
                    pos_logits = predictor(h1.unsqueeze(0), h2.unsqueeze(0))
                    pos_labels = torch.ones(1, device=device)
                    pos_loss_sum += criterion(pos_logits, pos_labels)
                    num_pos_examples += 1

                    # Negative pairs: same node vs random other node
                    for _ in range(num_negatives):
                        n_idx = pair_rng.randint(0, N)
                        h_neg = node_emb[n_idx]
                        neg_logits = predictor(h1.unsqueeze(0), h_neg.unsqueeze(0))
                        neg_labels = torch.zeros(1, device=device)
                        neg_loss_sum += criterion(neg_logits, neg_labels)

            if num_pos_examples == 0:
                continue

            loss = (pos_loss_sum + neg_loss_sum) / num_pos_examples / (1 + num_negatives)
            loss = loss / gradient_accumulation

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % gradient_accumulation == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(predictor.parameters()), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(predictor.parameters()), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation
            total_pairs += num_pos_examples
            global_step += 1

            if global_step % log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(global_step, 1)
        history["train_loss"].append(avg_loss)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, pairs = {total_pairs:,}")

        ckpt = {
            "epoch": epoch,
            "backbone_state_dict": backbone.module.state_dict() if isinstance(backbone, DataParallel) else backbone.state_dict(),
            "predictor_state_dict": predictor.module.state_dict() if isinstance(predictor, DataParallel) else predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": {"model_type": model_type, "hidden_dim": hidden_dim,
                       "num_layers": num_layers, "k": k, "num_negatives": num_negatives},
        }
        torch.save(ckpt, save_dir / f"context_{model_type}_epoch_{epoch}.pt")

    torch.save(backbone.module.state_dict() if isinstance(backbone, DataParallel) else backbone.state_dict(),
               save_dir / f"{model_type}_pretrained_backbone.pt")

    print(f"\nPretraining complete! Saved to {save_dir}")
    return history
