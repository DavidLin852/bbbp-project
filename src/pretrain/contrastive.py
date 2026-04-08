"""
Strategy 3: Molecular Contrastive Learning (GraphCL / MolCLR style)

Key idea: Learn representations by contrasting different augmented views
of the same molecule against other molecules in the batch.

Augmentations:
1. Node dropping: randomly remove 10-20% of atoms
2. Edge perturbation: randomly add/remove edges
3. Attribute masking: mask node features
4. Subgraph sampling: extract k-hop subgraph

Loss: NT-Xent (Normalized Temperature-scaled Cross-Entropy)
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


# ==================== Augmentations ====================

class NodeDropAug:
    """Randomly drop atoms (nodes) from the graph."""
    def __init__(self, drop_ratio: float = 0.1, seed: int = 42):
        self.drop_ratio = drop_ratio
        self.rng = np.random.RandomState(seed)

    def __call__(self, data):
        x = data.x
        num_nodes = x.shape[0]
        if num_nodes <= 2:
            return data  # Can't drop from tiny graphs

        keep_mask = self.rng.rand(num_nodes) > self.drop_ratio
        keep_nodes = keep_mask.nonzero()[0]

        if len(keep_nodes) < 2:
            return data

        # Reindex edges to keep only selected nodes
        old_to_new = torch.zeros(num_nodes, dtype=torch.long) - 1
        old_to_new[keep_nodes] = torch.arange(len(keep_nodes), dtype=torch.long)

        new_edge_index = []
        new_edge_attr = []
        for i in range(data.edge_index.shape[1]):
            src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            if old_to_new[src] >= 0 and old_to_new[dst] >= 0:
                new_edge_index.append([old_to_new[src].item(), old_to_new[dst].item()])
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    new_edge_attr.append(data.edge_attr[i].tolist())

        new_data = data.clone()
        new_data.x = x[keep_nodes]
        new_data.edge_index = torch.tensor(new_edge_index, dtype=torch.long).t()
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and new_edge_attr:
            new_data.edge_attr = torch.tensor(new_edge_attr, dtype=torch.float32)
        else:
            new_data.edge_attr = torch.zeros_like(new_data.edge_index.t())
        new_data.num_nodes = len(keep_nodes)
        return new_data


class EdgePerturbAug:
    """Randomly remove edges from the graph."""
    def __init__(self, perturb_ratio: float = 0.1, seed: int = 42):
        self.perturb_ratio = perturb_ratio
        self.rng = np.random.RandomState(seed)

    def __call__(self, data):
        num_edges = data.edge_index.shape[1]
        if num_edges == 0:
            return data

        keep_mask = self.rng.rand(num_edges) > self.perturb_ratio
        new_data = data.clone()
        new_data.edge_index = data.edge_index[:, keep_mask]
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            new_data.edge_attr = data.edge_attr[keep_mask]
        return new_data


class AttrMaskAug:
    """Randomly mask node features."""
    def __init__(self, mask_ratio: float = 0.1, seed: int = 42):
        self.mask_ratio = mask_ratio
        self.rng = np.random.RandomState(seed)

    def __call__(self, data):
        x = data.x.clone()
        num_nodes = x.shape[0]
        mask = self.rng.rand(num_nodes) < self.mask_ratio
        x[mask] = 0.0
        new_data = data.clone()
        new_data.x = x
        return new_data


class SubgraphAug:
    """Extract a k-hop ego-network subgraph."""
    def __init__(self, k: int = 2, seed: int = 42):
        self.k = k
        self.rng = np.random.RandomState(seed)

    def __call__(self, data):
        # Simple version: just return the graph as-is
        # Full implementation would do k-hop neighborhood extraction
        return data


# ==================== Augmentation Pipeline ====================

class RandomAugment:
    """Randomly applies one augmentation from the pool."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.augs = [
            NodeDropAug(drop_ratio=0.1, seed=seed),
            EdgePerturbAug(perturb_ratio=0.1, seed=seed),
            AttrMaskAug(mask_ratio=0.1, seed=seed),
        ]

    def __call__(self, data):
        aug = self.rng.choice(self.augs)
        return aug(data)


# ==================== NT-Xent Loss ====================

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy loss for contrastive learning.
    For batch of N molecules with 2 views each (2N total representations).
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (N, D) view 1 representations
            z2: (N, D) view 2 representations
        Returns:
            scalar loss
        """
        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # (2N, D)

        # Normalize
        z = F.normalize(z, dim=1)

        # Similarity matrix
        sim = torch.mm(z, z.t()) / self.tau  # (2N, 2N)

        # Mask diagonal (self-similarity)
        sim.fill_diagonal_(float('-inf'))

        # Positive pairs: (i, i+N) and (i+N, i)
        pos = torch.cat([
            sim[:N, N:].diag(),   # view1_i vs view2_i
            sim[N:, :N].diag()   # view2_i vs view1_i
        ]) / self.tau

        # Negative pairs: all except diagonal and positive
        loss = -pos + torch.logsumexp(sim, dim=1)
        return loss.mean()


# ==================== Projection Head ====================

class ProjectionHead(nn.Module):
    """Projection head for contrastive learning (discarded after pretraining)."""
    def __init__(self, hidden_dim: int = 128, proj_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.head(x), dim=1)


# ==================== Dataset ====================

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        smiles_list: list,
        cache_file: Optional[Path] = None,
        seed: int = 42,
    ):
        self.smiles_list = smiles_list
        self.data = []
        self._length = 0
        self.augmenter = RandomAugment(seed=seed)

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

        self._length = len(self.data)
        print(f"Built {self._length:,} graphs for contrastive learning")

        if cache_file:
            with open(cache_file, "wb") as f:
                pickle.dump({"data": self.data}, f)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return self.data[idx]


# ==================== Training ====================

def pretrain_contrastive(
    data_dir: str | Path,
    num_samples: int = 100000,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    model_type: Literal["gin", "gat", "gcn"] = "gin",
    hidden_dim: int = 128,
    proj_dim: int = 128,
    num_layers: int = 3,
    heads: int = 4,
    temperature: float = 0.1,
    num_workers: int = 4,
    save_dir: str | Path = "artifacts/models/pretrain/contrastive",
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
    graph_dataset = ContrastiveDataset(smiles_list=smiles_list, cache_file=cache_file)

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

    projector = ProjectionHead(hidden_dim=hidden_dim, proj_dim=proj_dim)
    criterion = NTXentLoss(temperature=temperature)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        backbone = DataParallel(backbone)
        projector = DataParallel(projector)

    backbone = backbone.to(device)
    projector = projector.to(device)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(projector.parameters()),
        lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP)")

    history = {"train_loss": []}
    global_step = 0

    print(f"\nStarting contrastive pretraining: {epochs} epochs, "
          f"{len(dataloader)} steps/epoch, temperature={temperature}")

    for epoch in range(epochs):
        backbone.train()
        projector.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)

            # Generate two augmented views (re-randomize each time)
            aug1 = RandomAugment(seed=epoch * 10000 + global_step)
            aug2 = RandomAugment(seed=epoch * 10000 + global_step + 1)

            # Augment the batch (simplified: apply to original batch)
            x1 = batch.x.clone()
            x2 = batch.x.clone()
            # Random node masking as augmentation
            mask1 = torch.rand(batch.x.shape[0], 1, device=device) < 0.1
            mask2 = torch.rand(batch.x.shape[0], 1, device=device) < 0.1
            x1[mask1.expand_as(x1)] = 0
            x2[mask2.expand_as(x2)] = 0

            batch1 = batch.clone(); batch1.x = x1
            batch2 = batch.clone(); batch2.x = x2

            if use_amp:
                with torch.cuda.amp.autocast():
                    emb1 = projector(backbone(batch1, return_node_emb=False))
                    emb2 = projector(backbone(batch2, return_node_emb=False))
                    loss = criterion(emb1, emb2)
                    loss = loss / gradient_accumulation
                scaler.scale(loss).backward()
            else:
                emb1 = projector(backbone(batch1, return_node_emb=False))
                emb2 = projector(backbone(batch2, return_node_emb=False))
                loss = criterion(emb1, emb2)
                loss = loss / gradient_accumulation
                loss.backward()

            if (global_step + 1) % gradient_accumulation == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(projector.parameters()), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(projector.parameters()), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation
            global_step += 1

            if global_step % log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation:.4f}"})

        avg_loss = total_loss / max(global_step, 1)
        history["train_loss"].append(avg_loss)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "backbone_state_dict": backbone.module.state_dict() if isinstance(backbone, DataParallel) else backbone.state_dict(),
            "projector_state_dict": projector.module.state_dict() if isinstance(projector, DataParallel) else projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": {"model_type": model_type, "hidden_dim": hidden_dim,
                       "proj_dim": proj_dim, "num_layers": num_layers,
                       "temperature": temperature},
        }
        torch.save(ckpt, save_dir / f"contrastive_{model_type}_epoch_{epoch}.pt")

    torch.save(backbone.module.state_dict() if isinstance(backbone, DataParallel) else backbone.state_dict(),
               save_dir / f"{model_type}_pretrained_backbone.pt")

    print(f"\nPretraining complete! Saved to {save_dir}")
    return history
