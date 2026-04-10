#!/usr/bin/env python
"""
Fine-tune a pretrained Transformer encoder on the BBB task (Group D).

Usage:
    python scripts/finetune/finetune_transformer.py \
        --pretrain_id T_E10_TRANS_1M \
        --seed 0 \
        --task classification \
        --device cuda

This script loads the pretrained encoder from a ZINC22 MLM pretrain experiment,
adds a fresh classification/regression head, and fine-tunes on B3DB.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.finetune.finetune_config import get_pretrain_config
from src.transformer.smiles_transformer import (
    SMILESTransformerEncoder,
    SMILESTransformerClassifier,
    SMILESTransformerRegressor,
)
from src.transformer.smiles_tokenizer import SMILESTokenizer
from src.transformer.trainer import collate_fn, SMILESDataset


# ============================================================
# Training Loop
# ============================================================

def train_epoch_transformer(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    task: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).float()

        optimizer.zero_grad()

        if task == "classification":
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.squeeze(), labels)
        else:
            pred = model(input_ids, attention_mask).squeeze()
            loss = criterion(pred, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_transformer(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate and return (predictions, labels, loss)."""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    criterion = nn.BCELoss() if task == "classification" else nn.MSELoss()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).float()

            if task == "classification":
                logits = model(input_ids, attention_mask)
                preds = torch.sigmoid(logits.squeeze()).cpu().numpy()
                loss = criterion(logits.squeeze(), labels)
            else:
                preds = model(input_ids, attention_mask).squeeze().cpu().numpy()
                loss = criterion(preds, labels)

            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return preds, labels, total_loss / n_batches


# ============================================================
# Metrics
# ============================================================

def compute_cls_metrics(labels, preds):
    from sklearn.metrics import roc_auc_score, f1_score
    if len(np.unique(labels)) < 2:
        return float("nan"), 0.0, 0.0
    auc = roc_auc_score(labels, preds)
    binary = (preds >= 0.5).astype(int)
    f1 = f1_score(labels, binary, zero_division=0)
    acc = float((binary == labels).mean())
    return auc, f1, acc


def compute_reg_metrics(labels, preds):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2 = r2_score(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae = mean_absolute_error(labels, preds)
    return r2, rmse, mae


# ============================================================
# Main Fine-tuning Function
# ============================================================

def finetune_transformer(
    pretrain_id: str,
    seed: int,
    task: str,
    output_dir: str | Path = "artifacts/models/finetune",
    embedding_dir: str | Path = "artifacts/embeddings",
    device: str = "auto",
    epochs: int = 200,
    patience: int = 25,
    lr: float = 5e-4,
    batch_size: int = 128,
    verbose: bool = True,
    skip_existing: bool = True,
) -> dict | None:
    """
    Fine-tune a pretrained Transformer encoder on the BBB task.

    Args:
        pretrain_id: e.g. "T_E10_TRANS_1M"
        seed: Random seed (0-4)
        task: "classification" or "regression"
        output_dir: Where to save results
        device: Device to use
        epochs: Max training epochs
        patience: Early stopping patience
        lr: Learning rate
        batch_size: Batch size
        verbose: Print progress
        skip_existing: Skip if result.json exists

    Returns:
        Result dict on success, None on failure
    """
    output_dir = Path(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)

    # --- Check pretrain config ---
    cfg = get_pretrain_config(pretrain_id)
    if cfg["model_type"] != "transformer":
        if verbose:
            print(f"[ERROR] {pretrain_id} is not a Transformer model")
        return None

    # --- Output subdir ---
    model_dir = output_dir / "D" / f"seed_{seed}" / task / pretrain_id
    model_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing:
        result_file = model_dir / "result.json"
        if result_file.exists():
            if verbose:
                print(f"  SKIP: {model_dir} already has result.json")
            return json.loads(result_file.read_text())

    if verbose:
        print(f"  TRANS-FT {pretrain_id}, seed={seed}, {task}")

    # --- Paths ---
    encoder_path = project_root / "artifacts" / "models" / "pretrain" / "exp_matrix" / pretrain_id / "transformer_pretrained_encoder.pt"
    tokenizer_path = project_root / "artifacts" / "models" / "pretrain" / "exp_matrix" / pretrain_id / "tokenizer.pkl"

    if not encoder_path.exists():
        if verbose:
            print(f"  [ERROR] Encoder not found: {encoder_path}")
        return None
    if not tokenizer_path.exists():
        if verbose:
            print(f"  [ERROR] Tokenizer not found: {tokenizer_path}")
        return None

    # --- Set seeds ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Load tokenizer ---
    with open(tokenizer_path, "rb") as f:
        tokenizer: SMILESTokenizer = pickle.load(f)
    if verbose:
        print(f"    Tokenizer vocab: {len(tokenizer)}, max_len=128")

    # --- Load B3DB splits ---
    splits_dir = project_root / "data" / "splits" / f"seed_{seed}" / f"{task}_scaffold"
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    if verbose:
        print(f"    Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # --- Create datasets ---
    label_col = "y_cls" if task == "classification" else "logBB"
    max_len = 128

    train_ds = SMILESDataset(train_df["SMILES_canon"].tolist(), train_df[label_col].tolist(), tokenizer, max_length=max_len)
    val_ds = SMILESDataset(val_df["SMILES_canon"].tolist(), val_df[label_col].tolist(), tokenizer, max_length=max_len)
    test_ds = SMILESDataset(test_df["SMILES_canon"].tolist(), test_df[label_col].tolist(), tokenizer, max_length=max_len)

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id), num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id), num_workers=0)

    # --- Create model with pretrained encoder ---
    encoder = SMILESTransformerEncoder(
        vocab_size=len(tokenizer),
        d_model=cfg["hidden_dim"],
        n_heads=cfg["heads"],
        n_layers=cfg["num_layers"],
        d_ff=cfg["hidden_dim"] * 4,
        dropout=0.1,
        max_len=max_len,
    )

    # Load pretrained weights
    state_dict = torch.load(encoder_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(state_dict)
    if verbose:
        print(f"    Loaded pretrained encoder: {cfg['num_layers']}L, d_model={cfg['hidden_dim']}, heads={cfg['heads']}")

    # Wrap with head
    if task == "classification":
        model = SMILESTransformerClassifier(
            vocab_size=len(tokenizer),
            d_model=cfg["hidden_dim"],
            n_heads=cfg["heads"],
            n_layers=cfg["num_layers"],
            d_ff=cfg["hidden_dim"] * 4,
            dropout=0.1,
            max_len=max_len,
        )
    else:
        model = SMILESTransformerRegressor(
            vocab_size=len(tokenizer),
            d_model=cfg["hidden_dim"],
            n_heads=cfg["heads"],
            n_layers=cfg["num_layers"],
            d_ff=cfg["hidden_dim"] * 4,
            dropout=0.1,
            max_len=max_len,
        )

    # Transfer pretrained encoder weights
    model.encoder.load_state_dict(encoder.state_dict())
    model = model.to(device)

    if verbose:
        print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"    Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Optimizer + scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max" if task == "classification" else "max", factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

    # --- Training loop ---
    best_val_metric = -float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch_transformer(model, train_loader, optimizer, criterion, device, task)
        val_preds, val_labels, val_loss = evaluate_transformer(model, val_loader, device, task)

        if task == "classification":
            val_auc, val_f1, val_acc = compute_cls_metrics(val_labels, val_preds)
            val_metric = val_auc
            msg = f"    Ep {epoch:3d}: train={train_loss:.4f} val={val_loss:.4f} val_auc={val_auc:.4f} val_f1={val_f1:.4f}"
        else:
            val_r2, val_rmse, val_mae = compute_reg_metrics(val_labels, val_preds)
            val_metric = val_r2
            msg = f"    Ep {epoch:3d}: train={train_loss:.4f} val={val_loss:.4f} val_r2={val_r2:.4f} val_rmse={val_rmse:.4f}"

        improved = val_metric > best_val_metric
        if improved:
            best_val_metric = val_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
            msg += " *"
        else:
            patience_counter += 1

        scheduler.step(val_metric)

        if epoch % 20 == 0 or improved or patience_counter >= patience:
            if verbose:
                print(msg)

        if patience_counter >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch}")
            break

    if best_state is None:
        if verbose:
            print(f"    [ERROR] No improvement during training")
        return None

    # --- Load best and evaluate ---
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    train_preds, train_labels, _ = evaluate_transformer(model, train_loader, device, task)
    val_preds, val_labels, _ = evaluate_transformer(model, val_loader, device, task)
    test_preds, test_labels, _ = evaluate_transformer(model, test_loader, device, task)

    if task == "classification":
        train_auc, train_f1, train_acc = compute_cls_metrics(train_labels, train_preds)
        val_auc, val_f1, val_acc = compute_cls_metrics(val_labels, val_preds)
        test_auc, test_f1, test_acc = compute_cls_metrics(test_labels, test_preds)

        result = {
            "exp_id": pretrain_id,
            "group": "D",
            "pretrain_id": pretrain_id,
            "model_type": "transformer",
            "strategy": cfg["strategy"],
            "task": task,
            "feature_type": None,
            "seed": seed,
            "train_auc": float(train_auc),
            "train_f1": float(train_f1),
            "train_acc": float(train_acc),
            "val_auc": float(val_auc),
            "val_f1": float(val_f1),
            "val_acc": float(val_acc),
            "test_auc": float(test_auc),
            "test_f1": float(test_f1),
            "test_acc": float(test_acc),
            "best_epoch": best_epoch,
            "total_epochs": epoch,
        }
        if verbose:
            print(f"    Test AUC: {test_auc:.4f}, F1: {test_f1:.4f}, Acc: {test_acc:.4f} (best ep {best_epoch})")
    else:
        train_r2, train_rmse, train_mae = compute_reg_metrics(train_labels, train_preds)
        val_r2, val_rmse, val_mae = compute_reg_metrics(val_labels, val_preds)
        test_r2, test_rmse, test_mae = compute_reg_metrics(test_labels, test_preds)

        result = {
            "exp_id": pretrain_id,
            "group": "D",
            "pretrain_id": pretrain_id,
            "model_type": "transformer",
            "strategy": cfg["strategy"],
            "task": task,
            "feature_type": None,
            "seed": seed,
            "train_r2": float(train_r2),
            "train_rmse": float(train_rmse),
            "train_mae": float(train_mae),
            "val_r2": float(val_r2),
            "val_rmse": float(val_rmse),
            "val_mae": float(val_mae),
            "test_r2": float(test_r2),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "best_epoch": best_epoch,
            "total_epochs": epoch,
        }
        if verbose:
            print(f"    Test R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f} (best ep {best_epoch})")

    # --- Save ---
    torch.save(best_state, model_dir / "model.pt")
    with open(model_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"    Saved: {model_dir}")

    # Cleanup
    del model, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Transformer on BBB task")
    parser.add_argument("--pretrain_id", type=str, required=True,
                        help="Pretrain experiment ID (e.g. T_E10_TRANS_1M)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "regression"])
    parser.add_argument("--output_dir", type=str, default="artifacts/models/finetune")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip", action="store_true")

    args = parser.parse_args()
    skip = not args.no_skip

    finetune_transformer(
        pretrain_id=args.pretrain_id,
        seed=args.seed,
        task=args.task,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        verbose=True,
        skip_existing=skip,
    )


if __name__ == "__main__":
    main()
