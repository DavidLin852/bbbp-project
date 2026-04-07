"""
Training utilities for SMILES Transformer models.

Includes:
- Dataset class for SMILES data
- Training loop with early stopping
- Evaluation functions
- Checkpoint management
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Literal, List, Tuple
from tqdm import tqdm
import pandas as pd

from .smiles_transformer import TransformerConfig, get_model
from .smiles_tokenizer import SMILESTokenizer, collate_smiles_batch


class SMILESDataset(Dataset):
    """Dataset for SMILES-based molecular property prediction."""

    def __init__(
        self,
        smiles_list: List[str],
        labels: List[float],
        tokenizer: SMILESTokenizer,
        max_length: int = 128
    ):
        """Initialize dataset.

        Args:
            smiles_list: List of SMILES strings
            labels: List of labels (float for regression, 0/1 for classification)
            tokenizer: SMILESTokenizer instance
            max_length: Maximum sequence length
        """
        self.smiles_list = smiles_list
        self.labels = np.array(labels, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        smiles = self.smiles_list[idx]
        label = self.labels[idx]

        # Tokenize
        encoded = self.tokenizer.encode(smiles, add_special_tokens=True)

        # Truncate if necessary
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]

        return {
            'smiles': smiles,
            'input_ids': encoded,
            'label': label
        }


def collate_fn(batch: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader.

    Args:
        batch: List of dataset items
        pad_token_id: Padding token ID

    Returns:
        Batched tensors
    """
    # Find max length in batch
    max_len = max(len(item['input_ids']) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        ids = item['input_ids']
        length = len(ids)

        # Pad or truncate
        if length < max_len:
            pad_length = max_len - length
            padded_ids = ids + [pad_token_id] * pad_length
            mask = [1] * length + [0] * pad_length
        else:
            padded_ids = ids[:max_len]
            mask = [1] * max_len

        input_ids.append(padded_ids)
        attention_mask.append(mask)
        labels.append(item['label'])

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.float)
    }


class Trainer:
    """Trainer for SMILES Transformer models."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: SMILESTokenizer,
        config: TransformerConfig,
        task: Literal['classification', 'regression'],
        device: torch.device,
        save_dir: str
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            tokenizer: SMILES tokenizer
            config: Training configuration
            task: Task type
            device: Training device
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.task = task
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        if task == 'classification':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min' if task == 'regression' else 'max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Training state
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metric': [],  # AUC for classification, R² for regression
            'learning_rates': []
        }
        self.best_metric = None
        self.best_epoch = 0
        self.patience_counter = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask).squeeze(-1)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / len(train_loader)

    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float, Dict]:
        """Evaluate model on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (loss, metric, detailed_metrics)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask).squeeze(-1)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                if self.task == 'classification':
                    probs = torch.sigmoid(outputs)
                    all_preds.extend(probs.cpu().numpy())
                else:
                    all_preds.extend(outputs.cpu().numpy())

                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        avg_loss = total_loss / len(val_loader)

        if self.task == 'classification':
            from sklearn.metrics import roc_auc_score, f1_score

            auc = roc_auc_score(all_labels, all_preds)
            preds_binary = (all_preds > 0.5).astype(int)
            f1 = f1_score(all_labels, preds_binary)

            metric = auc  # AUC is our main metric
            detailed_metrics = {
                'auc': auc,
                'f1': f1,
                'loss': avg_loss
            }
        else:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

            r2 = r2_score(all_labels, all_preds)
            rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
            mae = mean_absolute_error(all_labels, all_preds)

            metric = r2  # R² is our main metric
            detailed_metrics = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'loss': avg_loss
            }

        return avg_loss, metric, detailed_metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """Train model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)

        Returns:
            Training history
        """
        print(f"Training {self.task} model...")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Validate
            if val_loader is not None:
                val_loss, val_metric, detailed_metrics = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_metric'].append(val_metric)

                metric_name = 'AUC' if self.task == 'classification' else 'R²'
                print(f"Epoch {epoch+1}/{self.config.epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val {metric_name}: {val_metric:.4f}")

                # Update scheduler
                self.scheduler.step(val_metric if self.task == 'classification' else -val_metric)

                # Early stopping
                if self.best_metric is None or \
                   (self.task == 'classification' and val_metric > self.best_metric) or \
                   (self.task == 'regression' and val_metric > self.best_metric):

                    self.best_metric = val_metric
                    self.best_epoch = epoch
                    self.patience_counter = 0

                    # Save best model
                    self.save_checkpoint(epoch, detailed_metrics, is_best=True)
                    print(f"  → New best {metric_name}: {self.best_metric:.4f}")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}/{self.config.epochs} | Train Loss: {train_loss:.4f}")

        # Save final model
        self.save_checkpoint(epoch, detailed_metrics, is_best=False)

        return self.history

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Evaluation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict(),
            'history': self.history,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch
        }

        if is_best:
            path = self.save_dir / 'best_model.pt'
        else:
            path = self.save_dir / 'final_model.pt'

        torch.save(checkpoint, path)

        # Save metrics as JSON
        metrics_path = self.save_dir / ('best_metrics.json' if is_best else 'final_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_metric = checkpoint['best_metric']
        self.best_epoch = checkpoint['best_epoch']

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    task: Literal['classification', 'regression']
) -> Dict:
    """Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        task: Task type

    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model(input_ids, attention_mask).squeeze(-1)

            if task == 'classification':
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
            else:
                all_preds.extend(outputs.cpu().numpy())

            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if task == 'classification':
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix
        )

        preds_binary = (all_preds > 0.5).astype(int)

        return {
            'accuracy': accuracy_score(all_labels, preds_binary),
            'precision': precision_score(all_labels, preds_binary, zero_division=0),
            'recall': recall_score(all_labels, preds_binary, zero_division=0),
            'f1': f1_score(all_labels, preds_binary, zero_division=0),
            'auc': roc_auc_score(all_labels, all_preds),
            'auprc': average_precision_score(all_labels, all_preds),
            'confusion_matrix': confusion_matrix(all_labels, preds_binary).tolist(),
            'y_true': all_labels.tolist(),
            'y_pred': all_preds.tolist()
        }
    else:
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        return {
            'r2': r2_score(all_labels, all_preds),
            'rmse': np.sqrt(mean_squared_error(all_labels, all_preds)),
            'mae': mean_absolute_error(all_labels, all_preds),
            'y_true': all_labels.tolist(),
            'y_pred': all_preds.tolist()
        }
