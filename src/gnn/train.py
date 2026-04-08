"""
GNN training loop for BBB permeability prediction.

Supports both classification and regression with:
- Early stopping on validation metric
- Multi-seed evaluation (seeds 0-4)
- Consistent metrics with classical baseline pipeline
- CUDA-first design with CPU fallback
- Proper GPU memory management

Classification metrics: AUC, F1 (at 0.5 threshold)
Regression metrics: R², RMSE, MAE
Loss: BCE for classification, MSE for regression
"""

from __future__ import annotations

import gc
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from .models import GCN, GIN, GAT, GNNClassificationHead, GNNRegressionHead, GNNConfig
from .dataset import B3DBGNNDataset, GNNTrainingResult, GNNRegressionResult


# ==================== Device Utilities ====================

def get_device(device: str | None = None) -> torch.device:
    """
    Resolve the compute device.

    Args:
        device: "cuda", "cpu", or "auto" (default). If "auto", prefers CUDA.

    Returns:
        torch.device
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")

    return torch.device(device)


def print_device_info(device: torch.device) -> None:
    """Print GPU/CUDA diagnostic information."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        mem_total = props.total_memory / 1024**3
        print(f"  CUDA device: {props.name}")
        print(f"  CUDA memory: {mem_total:.1f} GB")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print(f"  Device: CPU")


def clear_gpu_memory() -> None:
    """Free GPU memory between experiment runs."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


# ==================== Training Functions ====================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    task: Literal["classification", "regression"],
) -> float:
    """Train for one epoch. All compute on the target device."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        # Move entire batch to device
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        h = model.backbone(batch)
        y = batch.y.squeeze()  # Already on GPU from .to(device)

        if task == "classification":
            pred = torch.sigmoid(model.head(h)).squeeze()
        else:
            pred = model.head(h).squeeze()

        # Loss on GPU
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: Literal["classification", "regression"],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluate model on a data loader.

    Returns:
        predictions (numpy, CPU), labels (numpy, CPU), average loss
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    # Loss functions created once outside the loop
    if task == "classification":
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h = model.backbone(batch)
            y = batch.y.view(-1)  # Flatten to 1D
            pred = (
                torch.sigmoid(model.head(h)).view(-1)
                if task == "classification"
                else model.head(h).view(-1)
            )

            # Move predictions and labels back to CPU for sklearn
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())

            loss = criterion(pred, y)
            total_loss += loss.item()
            n_batches += 1

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return preds, labels, total_loss / n_batches


# ==================== Metrics ====================

def compute_cls_metrics(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    """Compute AUC and F1 for classification."""
    if len(np.unique(labels)) < 2:
        return float("nan"), 0.0
    auc = roc_auc_score(labels, preds)
    binary_preds = (preds >= 0.5).astype(int)
    f1 = f1_score(labels, binary_preds, zero_division=0)
    return auc, f1


def compute_reg_metrics(
    labels: np.ndarray, preds: np.ndarray
) -> tuple[float, float, float]:
    """Compute R², RMSE, MAE for regression."""
    r2 = r2_score(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae = mean_absolute_error(labels, preds)
    return r2, rmse, mae


# ==================== Model Building ====================

class GraphClassifier(nn.Module):
    """Full GNN model with a classification head."""

    def __init__(self, backbone: nn.Module, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.head = GNNClassificationHead(hidden_dim, hidden_dim, dropout)


class GraphRegressor(nn.Module):
    """Full GNN model with a regression head."""

    def __init__(self, backbone: nn.Module, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.head = GNNRegressionHead(hidden_dim, hidden_dim, dropout)


def build_model(
    model_type: str,
    in_dim: int,
    config: GNNConfig,
    task: Literal["classification", "regression"],
    pretrained_encoder_path: str | None = None,
) -> nn.Module:
    """
    Build a GNN model with the appropriate task head.

    Args:
        model_type: One of "gcn", "gin", "gat"
        in_dim: Input node feature dimension
        config: GNN configuration
        task: Task type
        pretrained_encoder_path: Optional path to pretrained backbone state_dict

    Returns:
        Full model ready for training
    """
    hidden = config.hidden_dim

    if model_type == "gcn":
        backbone = GCN(in_dim, hidden, config.num_layers, config.dropout)
    elif model_type == "gin":
        backbone = GIN(in_dim, hidden, config.num_layers, config.dropout)
    elif model_type == "gat":
        backbone = GAT(
            in_dim, hidden, config.num_layers, heads=config.heads, dropout=config.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if pretrained_encoder_path is not None:
        state_dict = torch.load(pretrained_encoder_path, map_location="cpu")
        backbone.load_state_dict(state_dict)
        print(f"  Loaded pretrained backbone from {pretrained_encoder_path}")

    if task == "classification":
        return GraphClassifier(backbone, hidden, config.dropout)
    return GraphRegressor(backbone, hidden, config.dropout)


# ==================== Main Training Function ====================

def train_gnn(
    model_type: str,
    dataset: B3DBGNNDataset,
    seed: int,
    task: Literal["classification", "regression"],
    config: GNNConfig | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    num_workers: int = 0,
    verbose: bool = True,
    pretrained_encoder_path: str | None = None,
) -> GNNTrainingResult | GNNRegressionResult:
    """
    Train a single GNN model on a dataset.

    Args:
        model_type: One of "gcn", "gin", "gat"
        dataset: B3DBGNNDataset instance
        seed: Random seed for reproducibility
        task: "classification" or "regression"
        config: GNN configuration
        output_dir: Where to save model checkpoints and results
        device: "cuda", "cpu", or "auto" (default)
        num_workers: DataLoader num_workers (0 = main process, safe for all platforms)
        verbose: Print progress

    Returns:
        GNNTrainingResult or GNNRegressionResult
    """
    cfg = config or GNNConfig()
    output_dir = Path(output_dir) if output_dir else Path("artifacts/models/gnn")

    # Full reproducibility seeding
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # covers DataParallel/fully_sharded
    torch.backends.cudnn.deterministic = True  # reproducible conv on CUDA

    # Resolve device
    dev = get_device(device)

    # Build model and move to device
    in_dim = dataset.get_input_dim()
    model = build_model(model_type, in_dim, cfg, task, pretrained_encoder_path).to(dev)

    if verbose:
        pretrained_tag = f" (pretrained: {pretrained_encoder_path})" if pretrained_encoder_path else ""
        print(f"  Model: {model_type} ({task}){pretrained_tag}")
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print_device_info(dev)
        print(f"  Config: hidden={cfg.hidden_dim}, layers={cfg.num_layers}, "
              f"heads={cfg.heads}, dropout={cfg.dropout}")
        print(f"  LR={cfg.lr}, wd={cfg.weight_decay}, epochs={cfg.epochs}, "
              f"patience={cfg.patience}, batch={cfg.batch_size}, workers={num_workers}")

    # Data loaders with pin_memory for GPU efficiency
    train_loader = DataLoader(
        dataset.get_split("train"),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),  # Only pin memory on GPU
    )
    val_loader = DataLoader(
        dataset.get_split("val"),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
    )
    test_loader = DataLoader(
        dataset.get_split("test"),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Loss function on GPU
    criterion = nn.BCELoss() if task == "classification" else nn.MSELoss()

    # Training loop with early stopping
    best_val_metric = -float("inf")
    patience_counter = 0
    best_state: dict | None = None
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, dev, task)
        val_preds, val_labels, val_loss = evaluate(model, val_loader, dev, task)

        if task == "classification":
            val_auc, val_f1 = compute_cls_metrics(val_labels, val_preds)
            val_metric = val_auc
            msg = (f"    Epoch {epoch:3d}: train_loss={train_loss:.4f} | "
                   f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_f1={val_f1:.4f}")
        else:
            val_r2, val_rmse, val_mae = compute_reg_metrics(val_labels, val_preds)
            val_metric = val_r2
            msg = (f"    Epoch {epoch:3d}: train_loss={train_loss:.4f} | "
                   f"val_loss={val_loss:.4f} val_r2={val_r2:.4f} val_rmse={val_rmse:.4f}")

        improved = val_metric > best_val_metric
        if improved:
            best_val_metric = val_metric
            # Keep state on GPU if on GPU to avoid unnecessary transfer
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
            msg += " *"

        if epoch % 20 == 0 or improved or patience_counter >= cfg.patience:
            if verbose:
                print(msg)

        if patience_counter >= cfg.patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch}")
            break

        patience_counter += 1

    if best_state is None:
        raise RuntimeError("Training completed without improvement")

    # Load best model state
    model.load_state_dict(best_state)

    # Final evaluation on all splits
    train_preds, train_labels, train_loss_final = evaluate(model, train_loader, dev, task)
    val_preds, val_labels, val_loss_final = evaluate(model, val_loader, dev, task)
    test_preds, test_labels, test_loss_final = evaluate(model, test_loader, dev, task)

    if task == "classification":
        train_auc, train_f1 = compute_cls_metrics(train_labels, train_preds)
        val_auc, val_f1 = compute_cls_metrics(val_labels, val_preds)
        test_auc, test_f1 = compute_cls_metrics(test_labels, test_preds)

        result = GNNTrainingResult(
            model_name=model_type,
            seed=seed,
            task=task,
            train_auc=float(train_auc),
            train_f1=float(train_f1),
            train_loss=float(train_loss_final),
            val_auc=float(val_auc),
            val_f1=float(val_f1),
            val_loss=float(val_loss_final),
            test_auc=float(test_auc),
            test_f1=float(test_f1),
            test_loss=float(test_loss_final),
            best_epoch=best_epoch,
            total_epochs=epoch,
        )

        if verbose:
            print(f"  Test AUC: {test_auc:.4f}, F1: {test_f1:.4f} (best epoch {best_epoch})")

    else:
        train_r2, train_rmse, train_mae = compute_reg_metrics(train_labels, train_preds)
        val_r2, val_rmse, val_mae = compute_reg_metrics(val_labels, val_preds)
        test_r2, test_rmse, test_mae = compute_reg_metrics(test_labels, test_preds)

        result = GNNRegressionResult(
            model_name=model_type,
            seed=seed,
            task=task,
            train_r2=float(train_r2),
            train_rmse=float(train_rmse),
            train_mae=float(train_mae),
            train_loss=float(train_loss_final),
            val_r2=float(val_r2),
            val_rmse=float(val_rmse),
            val_mae=float(val_mae),
            val_loss=float(val_loss_final),
            test_r2=float(test_r2),
            test_rmse=float(test_rmse),
            test_mae=float(test_mae),
            test_loss=float(test_loss_final),
            best_epoch=best_epoch,
            total_epochs=epoch,
        )

        if verbose:
            print(f"  Test R2: {test_r2:.4f}, RMSE: {test_rmse:.4f} (best epoch {best_epoch})")

    # Save checkpoint and results
    model_dir = output_dir / f"seed_{seed}" / task / model_type
    model_dir.mkdir(parents=True, exist_ok=True)

    # Always save CPU state so checkpoints are portable
    cpu_state = {k: v.cpu() for k, v in best_state.items()}
    torch.save(cpu_state, model_dir / "model.pt")
    with open(model_dir / "result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Free GPU memory
    del model, best_state, cpu_state
    clear_gpu_memory()

    return result


# ==================== Multi-seed Helper ====================

def train_gnn_multiple_seeds(
    model_type: str,
    seed_base_dir: str | Path,
    split_name: str,
    seeds: list[int],
    task: Literal["classification", "regression"],
    config: GNNConfig | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    num_workers: int = 0,
    verbose: bool = True,
    pretrained_encoder_path: str | None = None,
) -> list:
    """
    Train a GNN model across multiple seeds.

    Args:
        model_type: GNN type ("gcn", "gin", "gat")
        seed_base_dir: Base path to seeds, e.g. "data/splits"
        split_name: Split directory name, e.g. "classification_scaffold"
        seeds: List of seeds
        task: "classification" or "regression"
        config: GNN configuration
        output_dir: Output directory
        device: "cuda", "cpu", or "auto"
        num_workers: DataLoader num_workers
        verbose: Print progress

    Returns:
        List of results, one per seed
    """
    results = []

    for seed in seeds:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  {model_type.upper()} | {task} | seed={seed}")
            print(f"{'=' * 60}")

        split_dir = Path(seed_base_dir) / f"seed_{seed}" / split_name

        dataset = B3DBGNNDataset(
            split_dir=split_dir,
            task=task,
        )

        result = train_gnn(
            model_type=model_type,
            dataset=dataset,
            seed=seed,
            task=task,
            config=config,
            output_dir=output_dir,
            device=device,
            num_workers=num_workers,
            verbose=verbose,
            pretrained_encoder_path=pretrained_encoder_path,
        )

        results.append(result)

    return results
