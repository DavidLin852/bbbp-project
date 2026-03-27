"""
Transformer-based Molecular Fingerprint Classifier

Treats molecular fingerprints as sequence features and uses
multi-head self-attention to learn high-order interactions.

Architecture:
- Input embedding from fingerprint bits
- Multi-layer Transformer Encoder
- Global pooling (mean/max)
- Classification head with sigmoid output
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder layer with multi-head attention."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int = 512, dropout: float = 0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through encoder layer."""
        # Self-attention
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feed-forward
        ff_output = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    """Stack of Transformer Encoder layers."""

    def __init__(self, d_model: int, n_heads: int, num_layers: int,
                 dim_feedforward: int = 512, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through all encoder layers."""
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)


class FingerprintEmbedding(nn.Module):
    """Learnable embedding for fingerprint bits."""

    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class TransformerClassifier(nn.Module):
    """Transformer-based classifier for molecular fingerprints.

    Args:
        input_dim: Dimension of input fingerprint
        hidden_dim: Hidden dimension for transformer
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        feedforward_dim: Feedforward dimension
        dropout: Dropout rate
        num_classes: Number of output classes (default 2 for BBB+/BBB-)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        feedforward_dim: int = 512,
        dropout: float = 0.3,
        num_classes: int = 1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Project input to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding (learnable)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=5000, dropout=dropout)

        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=hidden_dim,
            n_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            mask: Optional attention mask

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Project input
        x = self.input_projection(x)

        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer(x)

        # Global mean pooling over sequence
        x = x.mean(dim=1)  # (batch, hidden_dim)

        # Classification
        logits = self.classifier(x)

        return logits

    def predict_proba(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x, mask)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return F.softmax(logits, dim=-1)


@dataclass
class TransformerTrainingConfig:
    """Configuration for Transformer training."""
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    feedforward_dim: int = 512
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10
    eval_metric: str = "auc"


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def get_criterion(name: str = "bce") -> nn.Module:
    """Get loss function by name."""
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "focal":
        return FocalLoss(alpha=0.25, gamma=2.0)
    elif name == "balanced_bce":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))  # Adjust for imbalance
    else:
        return nn.BCEWithLogitsLoss()


def train_transformer(
    model: TransformerClassifier,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    config: TransformerTrainingConfig,
    device: torch.device,
    save_dir: str
) -> dict:
    """Train Transformer model.

    Args:
        model: TransformerClassifier instance
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        config: Training configuration
        device: Training device
        save_dir: Directory to save checkpoints

    Returns:
        Training history dictionary
    """
    from pathlib import Path
    import numpy as np
    from sklearn.metrics import roc_auc_score

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    criterion = get_criterion("bce")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    history = {
        'train_loss': [],
        'val_auc': [],
        'best_auc': 0.0,
        'best_epoch': 0
    }

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        if val_loader is not None:
            model.eval()
            val_probs = []
            val_targets = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    logits = model(X_batch)
                    probs = torch.sigmoid(logits.squeeze())
                    val_probs.extend(probs.cpu().numpy())
                    val_targets.extend(y_batch.numpy())

            val_auc = roc_auc_score(val_targets, val_probs)
            history['val_auc'].append(val_auc)

            scheduler.step(val_auc)

            if val_auc > history['best_auc']:
                history['best_auc'] = val_auc
                history['best_epoch'] = epoch
                torch.save(model.state_dict(), f"{save_dir}/best_model.pt")

            print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f}")

    return history


def evaluate_transformer(
    model: TransformerClassifier,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> dict:
    """Evaluate Transformer model on test set."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix
    )

    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits.squeeze())
            preds = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    return {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds),
        'recall': recall_score(all_targets, all_preds),
        'f1': f1_score(all_targets, all_preds),
        'auc': roc_auc_score(all_targets, all_probs),
        'auprc': average_precision_score(all_targets, all_probs),
        'confusion_matrix': confusion_matrix(all_targets, all_preds),
        'y_true': all_targets,
        'y_prob': all_probs,
        'y_pred': all_preds
    }
