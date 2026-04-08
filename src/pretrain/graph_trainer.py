"""
Graph Pretraining Trainer for ZINC22

Optimized for speed and multi-GPU training.
Property prediction pretraining with molecular descriptors as targets.

Key optimizations:
1. Parallel graph construction (multiprocessing.Pool) — ~10x faster than sequential
2. Precomputed pickle cache — skip graph building on subsequent runs
3. AMP mixed precision
4. Multi-GPU support via DataParallel
5. Gradient accumulation for effective larger batch size
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
import multiprocessing as mp

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn.models import GIN, GAT, GCN
from pretrain.data import ZINC22Dataset
from pretrain.graph import GraphPretrainer, PROPERTY_NAMES
from features.graph import smiles_to_pyg_graph


# ==================== Parallel Graph Builder ====================

def _build_single_graph(smiles: str) -> Optional[tuple]:
    """
    Build a single PyG graph + property target from a SMILES string.
    Returns (graph, target) or None if invalid.
    Must be a top-level function (not a method) for multiprocessing pickling.
    """
    try:
        graph = smiles_to_pyg_graph(smiles)
        if graph is None:
            return None

        from pretrain.graph import compute_zinc_properties
        target = compute_zinc_properties(smiles)
        if target is None:
            return None

        return (graph, target)
    except Exception:
        return None


def _chunked_parallel_build(smiles_list: list, num_workers: int, chunk_size: int = 500) -> tuple:
    """
    Build graphs in parallel using multiprocessing.Pool.

    Returns (data_list, targets_list).
    """
    if num_workers <= 1:
        # Sequential fallback
        data, targets = [], []
        for s in tqdm(smiles_list, desc="Building graphs"):
            result = _build_single_graph(s)
            if result:
                data.append(result[0])
                targets.append(result[1])
        return data, targets

    # Parallel: split into chunks to reduce pickle overhead
    # Each chunk is processed by one worker
    chunks = [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]

    print(f"Building {len(smiles_list):,} graphs with {num_workers} workers "
          f"({len(chunks)} chunks of ~{chunk_size} each)...")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    results = []
    with mp.Pool(num_workers) as pool:
        for chunk_result in tqdm(
            pool.imap_unordered(_build_chunk, chunks),
            total=len(chunks),
            desc="Building graphs",
        ):
            results.extend(chunk_result)

    data = [r[0] for r in results]
    targets = [r[1] for r in results]
    return data, targets


def _build_chunk(smiles_chunk: list) -> list:
    """Process a chunk of SMILES strings. Worker function for Pool.imap."""
    return [r for r in (_build_single_graph(s) for s in smiles_chunk) if r is not None]


# ==================== Dataset ====================

class CachedGraphDataset(torch.utils.data.Dataset):
    """
    Precomputes all graphs and property targets once.
    Returns cached PyG Data objects + targets for fast training.
    """

    def __init__(
        self,
        smiles_list: list,
        cache_file: Optional[Path] = None,
        num_workers: int = 1,  # Set to 1 to avoid shared memory exhaustion on limited systems
        pin_memory: bool = True,
    ):
        self.data = []
        self.targets = []
        self._length = 0

        if cache_file and cache_file.exists():
            print(f"Loading cached graphs from {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                    self.data = cached["data"]
                    self.targets = cached["targets"]
                    self._length = len(self.data)
                print(f"Loaded {self._length:,} cached graphs")
                return
            except Exception as e:
                print(f"Cache corrupted ({e}), rebuilding...")
                cache_file.unlink()

        # Build graphs in parallel
        self.data, self.targets = _chunked_parallel_build(smiles_list, num_workers)
        self._length = len(self.data)
        print(f"Built {self._length:,} valid graphs")

        if cache_file:
            print(f"Caching to {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump({"data": self.data, "targets": self.targets}, f)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def collate_graph_batch(batch_list):
    """Collate graph batch with property targets."""
    graphs = [item[0] for item in batch_list]
    targets = torch.stack([item[1] for item in batch_list])
    batched = Batch.from_data_list(graphs)
    batched.y_property = targets
    return batched


# ==================== Training ====================

def pretrain_gnn_model(
    data_dir: str | Path,
    num_samples: int = 100000,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    model_type: Literal["gin", "gat"] = "gin",
    hidden_dim: int = 128,
    num_layers: int = 3,
    heads: int = 4,
    num_workers: int = 1,  # Keep at 1 to avoid shared memory exhaustion
    save_dir: str | Path = "artifacts/models/pretrain",
    device: str = "auto",
    gradient_accumulation: int = 1,
    log_interval: int = 100,
) -> dict:
    """
    Pretrain a GNN on ZINC22 with molecular property prediction.

    Args:
        data_dir: Path to ZINC22 directory
        num_samples: Number of molecules to pretrain on
        batch_size: Batch size per GPU
        epochs: Number of epochs
        lr: Learning rate
        model_type: "gin", "gat", or "gcn"
        hidden_dim: Hidden dimension
        num_layers: Number of GNN layers
        heads: Number of attention heads (for GAT)
        num_workers: Set to 1 to avoid shared memory exhaustion on cluster nodes
        save_dir: Directory to save checkpoints
        device: Device to use
        gradient_accumulation: Gradient accumulation steps
        log_interval: Steps between log outputs

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

    # Step 1: Load SMILES from ZINC22
    print(f"\nLoading {num_samples:,} SMILES from {data_dir}")
    zinc_dataset = ZINC22Dataset(
        data_dir=data_dir,
        representation="smiles",
        num_samples=num_samples,
        shuffle=True,
    )
    smiles_list = zinc_dataset.smiles_list

    # Step 2: Build cached graphs with property targets
    cache_file = save_dir / f"graph_cache_{num_samples}.pkl"
    graph_dataset = CachedGraphDataset(
        smiles_list=smiles_list,
        cache_file=cache_file,
        num_workers=num_workers,
    )

    # Step 3: Create dataloader
    # Use num_workers=0 because graphs are precomputed and cached in memory
    dataloader = DataLoader(
        graph_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Graphs are already built, no need for workers
        pin_memory=True,
        collate_fn=collate_graph_batch,
        drop_last=True,
    )

    # Step 4: Compute property normalization stats
    all_targets = torch.stack(graph_dataset.targets)
    prop_mean = all_targets.mean(dim=0)
    prop_std = all_targets.std(dim=0) + 1e-8
    print(f"\nProperty normalization (mean / std):")
    for i, name in enumerate(PROPERTY_NAMES):
        print(f"  {name:20s}: {prop_mean[i]:8.3f} / {prop_std[i]:8.3f}")

    # Step 5: Create model
    pretrainer = GraphPretrainer(
        model_type=model_type,
        node_dim=22,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
        pretraining_task="property_prediction",
        num_properties=len(PROPERTY_NAMES),
    )

    # Multi-GPU: wrap the whole model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        pretrainer = DataParallel(pretrainer)

    pretrainer = pretrainer.to(device)

    # Step 6: Optimizer and scheduler
    optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)
    criterion = nn.MSELoss()

    # AMP
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP)")

    # Normalize targets in the batch
    prop_mean = prop_mean.to(device)
    prop_std = prop_std.to(device)

    # Step 6: Training loop
    history = {"train_loss": []}
    global_step = 0

    print(f"\nStarting pretraining: {epochs} epochs, "
          f"{len(dataloader)} steps/epoch, "
          f"effective batch size: {batch_size * gradient_accumulation}")

    for epoch in range(epochs):
        pretrainer.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = pretrainer(batch)
                    targets = (batch.y_property - prop_mean) / prop_std
                    loss = criterion(preds, targets)
                    loss = loss / gradient_accumulation
                scaler.scale(loss).backward()
            else:
                preds = pretrainer(batch)
                targets = (batch.y_property - prop_mean) / prop_std
                loss = criterion(preds, targets)
                loss = loss / gradient_accumulation
                loss.backward()

            if (global_step + 1) % gradient_accumulation == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation
            num_batches += 1
            global_step += 1

            if num_batches % log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation:.4f}"})

        avg_loss = total_loss / num_batches
        history["train_loss"].append(avg_loss)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": pretrainer.module.state_dict() if isinstance(pretrainer, DataParallel) else pretrainer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "prop_mean": prop_mean.cpu(),
            "prop_std": prop_std.cpu(),
            "property_names": PROPERTY_NAMES,
            "config": {
                "model_type": model_type,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_properties": len(PROPERTY_NAMES),
            },
        }
        torch.save(ckpt, save_dir / f"{model_type}_pretrain_epoch_{epoch}.pt")

    # Save final backbone
    final_state = pretrainer.module.state_dict() if isinstance(pretrainer, DataParallel) else pretrainer.state_dict()
    torch.save(pretrainer.backbone.state_dict(), save_dir / f"{model_type}_pretrained_backbone.pt")

    print(f"\nPretraining complete! Saved to {save_dir}")
    return history
