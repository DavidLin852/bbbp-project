"""
SMILES Transformer Pretraining Module

Implements masked language modeling (MLM) pretraining on ZINC22.
Target: SMILES Transformer.

Pretraining Task:
- Masked Token Modeling (similar to BERT)

Usage:
    # Pretrain Transformer on ZINC22
    python scripts/pretrain/pretrain_smiles.py \\
        --representation smiles \\
        --num_samples 10000 \\
        --epochs 10
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from transformer.smiles_transformer import (
    SMILESTransformerEncoder,
    TransformerConfig,
)
from transformer.smiles_tokenizer import SMILESTokenizer, create_tokenizer_from_data
from pretrain.data import ZINC22Dataset


# ==================== Pretraining Models ====================

class SMILESMasking:
    """
    Apply random masking to SMILES tokens for MLM pretraining.

    Similar to BERT masking: randomly replace tokens with [MASK] token.
    """

    def __init__(self, mask_token_id: int = 1, mask_ratio: float = 0.15):
        """
        Args:
            mask_token_id: Token ID for [MASK]
            mask_ratio: Fraction of tokens to mask
        """
        self.mask_token_id = mask_token_id
        self.mask_ratio = mask_ratio

    def __call__(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply masking to input tokens.

        Args:
            input_ids: Input token IDs (batch, seq_len)

        Returns:
            Tuple of (masked_ids, mask):
            - masked_ids: Input IDs with masked tokens
            - mask: Boolean mask indicating masked positions
        """
        # Create random mask
        batch_size, seq_len = input_ids.shape
        mask = torch.rand(batch_size, seq_len, device=input_ids.device) < self.mask_ratio

        # Don't mask padding tokens (assuming 0 is padding)
        mask = mask & (input_ids != 0)

        # Create masked input
        masked_ids = input_ids.clone()
        masked_ids[mask] = self.mask_token_id

        return masked_ids, mask


class SMILESPretrainer(nn.Module):
    """
    SMILES Transformer with masked language modeling head.

    Architecture:
    - Token embedding
    - Positional encoding
    - Transformer encoder
    - MLM head (predict masked tokens)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        padding_idx: int = 0,
        mask_token_id: int = 1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token_id = mask_token_id

        # Use existing Transformer encoder
        self.encoder = SMILESTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
            padding_idx=padding_idx,
        )

        # MLM head: predict masked tokens
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for MLM.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            mask: Boolean mask for MLM (batch, seq_len)

        Returns:
            Logits for masked positions (batch, num_masked, vocab_size)
        """
        # Encode
        encoded = self.encoder(input_ids, attention_mask)  # (batch, seq_len, d_model)

        # Get predictions for masked positions only
        masked_encoded = encoded[mask]  # (num_masked, d_model)
        logits = self.mlm_head(masked_encoded)  # (num_masked, vocab_size)

        return logits


# ==================== Pretraining Functions ====================

def pretrain_smiles_model(
    data_dir: str | Path,
    num_samples: int = 10000,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-4,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    mask_ratio: float = 0.15,
    save_dir: str | Path = "artifacts/models/pretrain",
    device: str = "auto",
    tokenizer_path: Optional[str | Path] = None,
    num_workers: int = 0,
) -> dict:
    """
    Pretrain SMILES Transformer on ZINC22 with MLM.

    Args:
        data_path: Path to ZINC22 SMILES file
        num_samples: Number of samples to use
        batch_size: Batch size
        epochs: Number of training epochs
        lr: Learning rate
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        mask_ratio: Ratio of tokens to mask
        save_dir: Directory to save checkpoints
        device: Device to use
        tokenizer_path: Path to save/load tokenizer

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

    # Create or load tokenizer
    if tokenizer_path and Path(tokenizer_path).exists():
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = SMILESTokenizer.load(tokenizer_path)
    else:
        print("Creating new tokenizer from ZINC22 data")
        dataset = ZINC22Dataset(
            data_dir=data_dir,
            representation="smiles",
            num_samples=min(num_samples, 50000),  # Sample for vocab building
        )
        tokenizer = create_tokenizer_from_data(dataset.smiles_list)

        tokenizer_path = save_dir / "tokenizer.pkl"
        tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

    print(f"Vocab size: {len(tokenizer)}")

    # Load data
    print(f"Loading {num_samples} samples from {data_dir}")
    dataset = ZINC22Dataset(
        data_dir=data_dir,
        representation="smiles",
        num_samples=num_samples,
    )

    # Create masking (reused across all batches, not recreated each time)
    masking = SMILESMasking(
        mask_token_id=tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else 1,
        mask_ratio=mask_ratio,
    )

    # Determine optimal num_workers if not specified
    if num_workers == 0:
        import multiprocessing as mp
        num_workers = min(8, mp.cpu_count())

    def collate_fn(batch):
        """Custom collate function for SMILES MLM."""
        from transformer.trainer import collate_smiles_batch

        batched = collate_smiles_batch(
            batch,
            labels=[0] * len(batch),  # Dummy labels
            tokenizer=tokenizer,
            max_length=128,
        )

        masked_ids, mask = masking(batched["input_ids"])

        return {
            "input_ids": masked_ids,
            "attention_mask": batched["attention_mask"],
            "mask": mask,
            "original_ids": batched["input_ids"],
        }

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model
    model = SMILESPretrainer(
        vocab_size=len(tokenizer),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=4 * d_model,
        dropout=0.1,
        max_len=128,
        padding_idx=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else 1,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 10
    )

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Mixed precision training
    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP)")

    # Training loop
    history = {"train_loss": [], "learning_rates": []}

    print(f"Starting pretraining: {epochs} epochs")
    print(f"Model: Transformer ({d_model}d, {n_heads}heads, {n_layers}layers)")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mask = batch["mask"].to(device)
            original_ids = batch["original_ids"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    logits = model(input_ids, attention_mask, mask)
                    targets = original_ids[mask]
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask, mask)
                targets = original_ids[mask]
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / num_batches
        history["train_loss"].append(avg_loss)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        # Update learning rate
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "history": history,
            "config": {
                "vocab_size": len(tokenizer),
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "mask_ratio": mask_ratio,
            },
        }

        torch.save(checkpoint, save_dir / f"transformer_pretrain_epoch_{epoch}.pt")

    # Save final encoder (for fine-tuning)
    torch.save(model.encoder.state_dict(), save_dir / "transformer_pretrained_encoder.pt")

    # Save training history
    import json as _json
    with open(save_dir / "training_history.json", "w") as f:
        _json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)

    print(f"\nPretraining complete!")
    print(f"Saved pretrained encoder to: {save_dir / 'transformer_pretrained_encoder.pt'}")

    return history
