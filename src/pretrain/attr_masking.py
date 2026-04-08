"""
Strategy 1: Node Attribute Masking (proper implementation)

Based on Hu et al. 2019 "Strategies for Pre-training Graph Neural Networks".

Key idea: Randomly mask 15% of node feature vectors (zero out),
then use a decoder MLP on node embeddings (before pooling) to
reconstruct the original 22-dim node features.

Unlike the "simplified" masking in graph.py which uses graph-level
embeddings, this properly reconstructs at the node level.
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
from rdkit import Chem


# ==================== Masking Transform ====================

class NodeAttributeMasker:
    """
    Randomly masks node features for attribute masking pretraining.

    With probability mask_ratio, zero out the entire feature vector of a node.
    This is equivalent to replacing it with a zero-vector "mask" token.
    """

    def __init__(self, mask_ratio: float = 0.15, seed: int = 42):
        self.mask_ratio = mask_ratio
        self.rng = np.random.RandomState(seed)

    def __call__(self, data):
        """
        Args:
            data: PyG Data object with .x (node features)

        Returns:
            masked_data: Copy with masked node features
            mask: Boolean tensor indicating which nodes were masked
        """
        x = data.x.clone()
        num_nodes = x.shape[0]
        mask = self.rng.rand(num_nodes) < self.mask_ratio
        # Don't mask nodes that are all zeros already
        zero_mask = (x.sum(dim=1) == 0)
        mask = mask & ~zero_mask
        x[mask] = 0.0

        data = data.clone()
        data.x_masked = x
        data.node_mask = mask
        return data


# ==================== Decoder ====================

class NodeFeatureDecoder(nn.Module):
    """
    Decoder that reconstructs node features from node embeddings.

    MLP: Linear(hidden_dim, hidden_dim) -> BatchNorm -> ReLU -> Linear(hidden_dim, node_dim)
    """

    def __init__(self, hidden_dim: int, node_dim: int, dropout: float = 0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        return self.decoder(node_emb)


# ==================== Dataset ====================

class AttrMaskingDataset(torch.utils.data.Dataset):
    """
    Dataset for node attribute masking pretraining.
    Returns PyG Data objects (graphs with masked features applied lazily).
    """

    def __init__(
        self,
        smiles_list: list,
        cache_file: Optional[Path] = None,
        mask_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.smiles_list = smiles_list
        self.data = []
        self._length = 0
        self.masker = NodeAttributeMasker(mask_ratio=mask_ratio, seed=seed)

        if cache_file and cache_file.exists():
            print(f"Loading cached graphs from {cache_file}")
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
                self.data = cached["data"]
                self._length = len(self.data)
            print(f"Loaded {self._length:,} cached graphs")
            return

        print(f"Building {len(smiles_list):,} graphs...")
        for i, smiles in enumerate(tqdm(smiles_list, desc="Building graphs")):
            try:
                graph = smiles_to_pyg_graph(smiles)
                self.data.append(graph)
            except Exception:
                continue

            if (i + 1) % 10000 == 0:
                print(f"  Built {len(self.data):,} / {len(smiles_list):,}")

        self._length = len(self.data)
        print(f"Built {self._length:,} graphs for attribute masking")

        if cache_file:
            with open(cache_file, "wb") as f:
                pickle.dump({"data": self.data}, f)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        graph = self.data[idx]
        masked_graph = self.masker(graph)
        return masked_graph


def collate_attr_mask_batch(batch_list):
    """Collate batch, apply masking in batch mode for efficiency."""
    batched = Batch.from_data_list([b for b in batch_list])
    return batched


# ==================== Training ====================

def pretrain_attr_masking(
    data_dir: str | Path,
    num_samples: int = 100000,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    model_type: Literal["gin", "gat", "gcn"] = "gin",
    hidden_dim: int = 128,
    num_layers: int = 3,
    heads: int = 4,
    mask_ratio: float = 0.15,
    num_workers: int = 4,
    save_dir: str | Path = "artifacts/models/pretrain/attr_masking",
    device: str = "auto",
    gradient_accumulation: int = 1,
    log_interval: int = 100,
) -> dict:
    """
    Pretrain GNN with node attribute masking on ZINC22.

    Args:
        data_dir: Path to ZINC22 directory
        num_samples: Number of molecules
        batch_size: Batch size
        epochs: Number of epochs
        lr: Learning rate
        model_type: "gin", "gat", or "gcn"
        hidden_dim: Hidden dimension
        num_layers: Number of GNN layers
        heads: Attention heads (GAT only)
        mask_ratio: Fraction of nodes to mask
        num_workers: DataLoader workers
        save_dir: Directory to save checkpoints
        device: Device
        gradient_accumulation: Gradient accumulation steps
        log_interval: Logging interval

    Returns:
        Training history
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    # Step 1: Load SMILES
    print(f"\nLoading {num_samples:,} SMILES from {data_dir}")
    zinc_dataset = ZINC22Dataset(
        data_dir=data_dir,
        representation="smiles",
        num_samples=num_samples,
        shuffle=True,
    )
    smiles_list = zinc_dataset.smiles_list

    # Step 2: Build cached graphs
    cache_file = save_dir / f"graph_cache_{num_samples}.pkl"
    graph_dataset = AttrMaskingDataset(
        smiles_list=smiles_list,
        cache_file=cache_file,
        mask_ratio=mask_ratio,
    )

    dataloader = DataLoader(
        graph_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_attr_mask_batch,
        drop_last=True,
    )

    # Step 3: Create model
    if model_type == "gin":
        backbone = GIN(in_dim=22, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.1)
    elif model_type == "gat":
        backbone = GAT(in_dim=22, hidden_dim=hidden_dim, num_layers=num_layers, heads=heads, dropout=0.1)
    elif model_type == "gcn":
        backbone = GCN(in_dim=22, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.1)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    decoder = NodeFeatureDecoder(hidden_dim=hidden_dim, node_dim=22, dropout=0.1)
    criterion = nn.MSELoss(reduction="none")

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        backbone = DataParallel(backbone)
        decoder = DataParallel(decoder)

    backbone = backbone.to(device)
    decoder = decoder.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(decoder.parameters()),
        lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

    # AMP
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP)")

    history = {"train_loss": []}
    global_step = 0

    print(f"\nStarting attribute masking pretraining: {epochs} epochs, "
          f"{len(dataloader)} steps/epoch, mask_ratio={mask_ratio}")

    for epoch in range(epochs):
        backbone.train()
        decoder.train()
        total_loss = 0.0
        total_nodes = 0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)

            # Apply masking on-device (re-randomize for each batch)
            num_nodes = batch.x.shape[0]
            mask = torch.rand(num_nodes, device=device) < mask_ratio
            zero_mask = (batch.x.sum(dim=1) == 0)
            mask = mask & ~zero_mask

            x_masked = batch.x.clone()
            x_masked[mask] = 0.0

            # Forward: backbone returns node embeddings
            node_emb = backbone(batch, return_node_emb=True)  # (N, hidden)
            preds = decoder(node_emb)  # (N, 22)

            # Loss only on masked nodes
            loss_per_node = criterion(preds, batch.x)  # (N, 22)
            masked_loss = loss_per_node[mask]
            if masked_loss.numel() == 0:
                loss = loss_per_node.mean() * 0  # zero loss, no gradient
            else:
                loss = masked_loss.mean()

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = loss / gradient_accumulation
                scaler.scale(loss).backward()
            else:
                loss = loss / gradient_accumulation
                loss.backward()

            loss = loss / gradient_accumulation
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % gradient_accumulation == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(decoder.parameters()), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(decoder.parameters()), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation
            total_nodes += mask.sum().item()
            global_step += 1

            if global_step % log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation:.4f}"})

        avg_loss = total_loss / max(global_step, 1)
        history["train_loss"].append(avg_loss)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, masked_nodes = {total_nodes:,}")

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "backbone_state_dict": backbone.module.state_dict() if isinstance(backbone, DataParallel) else backbone.state_dict(),
            "decoder_state_dict": decoder.module.state_dict() if isinstance(decoder, DataParallel) else decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "config": {
                "model_type": model_type,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "mask_ratio": mask_ratio,
                "node_dim": 22,
            },
        }
        torch.save(ckpt, save_dir / f"attr_mask_{model_type}_epoch_{epoch}.pt")

    # Save final backbone
    final_backbone = backbone.module.state_dict() if isinstance(backbone, DataParallel) else backbone.state_dict()
    torch.save(final_backbone, save_dir / f"{model_type}_pretrained_backbone.pt")

    print(f"\nPretraining complete! Saved to {save_dir}")
    return history
