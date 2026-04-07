"""
SMILES Transformer Model for Molecular Property Prediction

Transformer encoder architecture that operates on tokenized SMILES sequences.
Designed for both classification (BBB+) and regression (logBB) tasks.

Architecture:
- Token embedding + positional encoding
- Multi-layer Transformer encoder
- Global pooling (mean/attention)
- Task-specific head (classification/regression)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Literal


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SMILESTransformerEncoder(nn.Module):
    """Transformer encoder for SMILES sequences."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        padding_idx: int = 0
    ):
        """Initialize SMILES Transformer encoder.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feedforward dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
            padding_idx: Padding token index
        """
        super().__init__()

        self.d_model = d_model
        self.padding_idx = padding_idx

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN architecture for better stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            input_ids: Token indices of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)

        Returns:
            Encoded representations of shape (batch, seq_len, d_model)
        """
        # Embed tokens
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create transformer mask (invert attention mask)
        if attention_mask is not None:
            # Convert from (batch, seq_len) with 1=real, 0=pad
            # to (batch, seq_len) with True=pad, False=real
            mask = (attention_mask == 0)
        else:
            mask = None

        # Pass through transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Layer norm
        x = self.layer_norm(x)

        return x


class SMILESTransformerClassifier(nn.Module):
    """SMILES Transformer for classification tasks (BBB+ prediction)."""

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
        pooling: Literal['mean', 'max', 'cls'] = 'mean'
    ):
        """Initialize classifier.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feedforward dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
            padding_idx: Padding token index
            pooling: Pooling strategy ('mean', 'max', 'cls')
        """
        super().__init__()

        self.encoder = SMILESTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
            padding_idx=padding_idx
        )

        self.pooling = pooling

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for classification.

        Args:
            input_ids: Token indices of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)

        Returns:
            Logits of shape (batch, 1)
        """
        # Encode
        x = self.encoder(input_ids, attention_mask)  # (batch, seq_len, d_model)

        # Pool
        if self.pooling == 'mean':
            if attention_mask is not None:
                # Masked mean pooling
                mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
                x = (x * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                x = x.mean(dim=1)
        elif self.pooling == 'max':
            if attention_mask is not None:
                # Masked max pooling
                mask = (attention_mask == 0).unsqueeze(-1)
                x = x.masked_fill(mask, float('-inf'))
                x = x.max(dim=1)[0]
            else:
                x = x.max(dim=1)[0]
        elif self.pooling == 'cls':
            # Use first token (like BERT's [CLS])
            x = x[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Classify
        logits = self.classifier(x)  # (batch, 1)

        return logits

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get probability predictions.

        Args:
            input_ids: Token indices
            attention_mask: Attention mask

        Returns:
            Probabilities of shape (batch,)
        """
        logits = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits).squeeze(-1)


class SMILESTransformerRegressor(nn.Module):
    """SMILES Transformer for regression tasks (logBB prediction)."""

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
        pooling: Literal['mean', 'max', 'cls'] = 'mean'
    ):
        """Initialize regressor.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feedforward dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
            padding_idx: Padding token index
            pooling: Pooling strategy ('mean', 'max', 'cls')
        """
        super().__init__()

        self.encoder = SMILESTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
            padding_idx=padding_idx
        )

        self.pooling = pooling

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for regression.

        Args:
            input_ids: Token indices of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)

        Returns:
            Predictions of shape (batch, 1)
        """
        # Encode
        x = self.encoder(input_ids, attention_mask)  # (batch, seq_len, d_model)

        # Pool
        if self.pooling == 'mean':
            if attention_mask is not None:
                # Masked mean pooling
                mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
                x = (x * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                x = x.mean(dim=1)
        elif self.pooling == 'max':
            if attention_mask is not None:
                # Masked max pooling
                mask = (attention_mask == 0).unsqueeze(-1)
                x = x.masked_fill(mask, float('-inf'))
                x = x.max(dim=1)[0]
            else:
                x = x.max(dim=1)[0]
        elif self.pooling == 'cls':
            # Use first token
            x = x[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Regress
        pred = self.regressor(x)  # (batch, 1)

        return pred


@dataclass
class TransformerConfig:
    """Configuration for SMILES Transformer model."""

    # Model architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 512
    pooling: Literal['mean', 'max', 'cls'] = 'mean'

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15
    warmup_epochs: int = 5

    # Data
    max_smiles_length: int = 128

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'max_len': self.max_len,
            'pooling': self.pooling,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'warmup_epochs': self.warmup_epochs,
            'max_smiles_length': self.max_smiles_length
        }


def get_model(
    task: Literal['classification', 'regression'],
    vocab_size: int,
    config: TransformerConfig
) -> nn.Module:
    """Get a SMILES Transformer model.

    Args:
        task: Task type ('classification' or 'regression')
        vocab_size: Size of vocabulary
        config: Model configuration

    Returns:
        Model instance
    """
    if task == 'classification':
        return SMILESTransformerClassifier(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_len=config.max_len,
            padding_idx=0,  # PAD token is always at index 0
            pooling=config.pooling
        )
    elif task == 'regression':
        return SMILESTransformerRegressor(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_len=config.max_len,
            padding_idx=0,
            pooling=config.pooling
        )
    else:
        raise ValueError(f"Unknown task: {task}")
