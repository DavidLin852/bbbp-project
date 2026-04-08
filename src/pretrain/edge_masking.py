"""
Strategy 2: Edge Feature Prediction

Based on Hu et al. 2019 "Strategies for Pre-training Graph Neural Networks".

Key idea: Randomly remove 15% of edges from the input graph,
then predict the original 7-dim edge features from concatenated
endpoint node embeddings.
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional
import torch
import torch.nn as nn
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


# ==================== Edge Perturbation ====================

class EdgeMasker:
    """
    Randomly removes edges for edge feature prediction pretraining.
    Stores original edge features as targets.
    """

    def __init__(self, edge_drop_ratio: float = 0.15, seed: int = 42):
        self.edge_drop_ratio = edge_drop_ratio
        self.rng = np.random.RandomState(seed)

    def __call__(self, data):
        """
        Args:
            data: PyG Data object with .edge_index and .edge_attr

        Returns:
            masked_data: Copy with masked edges removed
            edge_targets: Original edge features for masked edges
            edge_mask: Boolean mask for which edges were masked
        """
        edge_index = data.edge_index  # (2, num_edges)
        edge_attr = data.edge_attr  # (num_edges, 7)
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            data = data.clone()
            data.edge_index_masked = edge_index
            data.edge_attr_masked = edge_attr
            data.edge_targets = edge_attr
            data.edge_mask = torch.zeros(num_edges, dtype=torch.bool)
            return data

        # Randomly select edges to mask
        mask = self.rng.rand(num_edges) < self.edge_drop_ratio

        # Get targets for masked edges
        edge_targets = edge_attr[mask]  # (num_masked, 7)

        # Create masked graph
        masked_edge_index = edge_index[:, ~mask]
        masked_edge_attr = edge_attr[~mask]

        data = data.clone()
        data.edge_index_masked = masked_edge_index
        data.edge_attr_masked = masked_edge_attr
        data.edge_targets = edge_targets
        data.edge_mask = mask
        return data


# ==================== Decoder ====================

class EdgeFeatureDecoder(nn.Module):
    """
    Predicts 7-dim edge features from concatenated endpoint node embeddings.

    MLP: Linear(2 * hidden_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 7)
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 7, dropout: float = 0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim),
        )

    def forward(self, node_emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_emb: Node embeddings (num_nodes, hidden_dim)
            edge_index: Edge connectivity (2, num_masked_edges)

        Returns:
            Predicted edge features (num_masked_edges, 7)
        """
        src, dst = edge_index[0], edge_index[1]
        edge_emb = torch.cat([node_emb[src], node_emb[dst]], dim=1)  # (E, 2*hidden)
        return self.decoder(edge_emb)


# ==================== Dataset ====================

class EdgeMaskingDataset(torch.utils.data.Dataset):
    """
    Dataset for edge feature prediction pretraining.
    """

    def __init__(
        self,
        smiles_list: list,
        cache_file: Optional[Path] = None,
        edge_drop_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.smiles_list = smiles_list
        self.data = []
        self._length = 0
        self.masker = EdgeMasker(edge_drop_ratio=edge_drop_ratio, seed=seed)

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
        print(f"Built {self._length:,} graphs for edge masking")

        if cache_file:
            with open(cache_file, "wb") as f:
                pickle.dump({"data": self.data}, f)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        graph = self.data[idx]
        masked_graph = self.masker(graph)
        return masked_graph


def collate_edge_mask_batch(batch_list):
    """Collate batch with edge masking."""
    batched = Batch.from_data_list([b for b in batch_list])
    return batched


# ==================== Training ====================

def pretrain_edge_masking(
    data_dir: str | Path,
    num_samples: int = 100000,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    model_type: Literal["gin", "gat", "gcn"] = "gin",
    hidden_dim: int = 128,
    num_layers: int = 3,
    heads: int = 4,
    edge_drop_ratio: float = 0.15,
    num_workers: int = 4,
    save_dir: str | Path = "artifacts/models/pretrain/edge_masking",
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
    zinc_dataset = ZINC22Dataset(data_dir=data_dir, representation="smiles", num_samples=num_samples, shuffle=True)
    smiles_list = zinc_dataset.smiles_list

    cache_file = save_dir / f"graph_cache_{num_samples}.pkl"
    graph_dataset = EdgeMaskingDataset(smiles_list=smiles_list, cache_file=cache_file,
                                        edge_drop_ratio=edge_drop_ratio)

    dataloader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True,
                            collate_fn=collate_edge_mask_batch, drop_last=True)

    # Create model
    if model_type == "gin":
        backbone = GIN(in_dim=22, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.1)
    elif model_type == "gat":
        backbone = GAT(in_dim=22, hidden_dim=hidden_dim, num_layers=num_layers, heads=heads, dropout=0.1)
    elif model_type == "gcn":
        backbone = GCN(in_dim=22, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.1)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    decoder = EdgeFeatureDecoder(hidden_dim=hidden_dim, edge_dim=7, dropout=0.1)
    criterion = nn.MSELoss()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        backbone = DataParallel(backbone)
        decoder = DataParallel(decoder)

    backbone = backbone.to(device)
    decoder = decoder.to(device)

    optimizer = torch.optim.AdamW(list(backbone.parameters()) + list(decoder.parameters()),
                                      lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP)")

    history = {"train_loss": []}
    global_step = 0

    print(f"\nStarting edge masking pretraining: {epochs} epochs, "
          f"{len(dataloader)} steps/epoch, edge_drop_ratio={edge_drop_ratio}")

    for epoch in range(epochs):
        backbone.train()
        decoder.train()
        total_loss = 0.0
        total_edges = 0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)

            # Re-randomize masking per batch
            num_edges = batch.edge_attr.shape[0]
            edge_mask = torch.rand(num_edges, device=device) < edge_drop_ratio

            # Get targets
            edge_targets = batch.edge_attr[edge_mask]

            if edge_targets.shape[0] == 0:
                continue

            # Masked edge features → zero
            masked_edge_attr = batch.edge_attr.clone()
            masked_edge_attr[edge_mask] = 0.0

            # Forward
            node_emb = backbone(batch, return_node_emb=True)
            edge_index = batch.edge_index[:, ~edge_mask]
            edge_attr_in = masked_edge_attr[~edge_mask]

            # Create masked batch
            masked_batch = batch.clone()
            masked_batch.edge_attr = edge_attr_in

            # Re-run backbone with masked edges (use the already-computed node_emb from full graph for simplicity)
            # Actually, we need to use the original edge structure for the message passing
            # So we pass the masked edge features but keep original edge_index
            masked_batch.edge_attr = edge_attr_in

            if use_amp:
                with torch.cuda.amp.autocast():
                    node_emb = backbone(masked_batch, return_node_emb=True)
                    preds = decoder(node_emb, edge_index)
                    loss = criterion(preds, edge_targets)
                    loss = loss / gradient_accumulation
                scaler.scale(loss).backward()
            else:
                node_emb = backbone(masked_batch, return_node_emb=True)
                preds = decoder(node_emb, edge_index)
                loss = criterion(preds, edge_targets)
                loss = loss / gradient_accumulation
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
            total_edges += edge_targets.shape[0]
            global_step += 1

            if global_step % log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation:.4f}"})

        avg_loss = total_loss / max(global_step, 1)
        history["train_loss"].append(avg_loss)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, masked_edges = {total_edges:,}")

        ckpt = {
            "epoch": epoch,
            "backbone_state_dict": backbone.module.state_dict() if isinstance(backbone, DataParallel) else backbone.state_dict(),
            "decoder_state_dict": decoder.module.state_dict() if isinstance(decoder, DataParallel) else decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": {"model_type": model_type, "hidden_dim": hidden_dim,
                       "num_layers": num_layers, "edge_drop_ratio": edge_drop_ratio, "edge_dim": 7},
        }
        torch.save(ckpt, save_dir / f"edge_mask_{model_type}_epoch_{epoch}.pt")

    torch.save(backbone.module.state_dict() if isinstance(backbone, DataParallel) else backbone.state_dict(),
               save_dir / f"{model_type}_pretrained_backbone.pt")

    print(f"\nPretraining complete! Saved to {save_dir}")
    return history
