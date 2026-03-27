"""
Multi-task GAT: Simultaneous BBB+/- classification and logBB regression

Strategy:
- Use classification groups A,B (with logBB labels from group A)
- Task 1: BBB+/- classification (all A,B samples)
- Task 2: logBB regression (group A only, masked for group B)

This leverages:
1. Large classification dataset (A,B groups)
2. Regression labels from group A
3. Shared representation learning

Usage:
    python scripts/04b_train_gat_multitask_cls_reg.py --seed 0 --lambda_cls 1.0 --lambda_reg 0.5
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from ..featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig
from ..utils.seed import seed_everything
from ..utils.metrics import classification_metrics
from ..utils.plotting import plot_roc_curves


@dataclass
class MultiTaskTrainCfg:
    seed: int = 0
    device: str = "cuda"
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 80
    batch_size: int = 64
    hidden: int = 128
    gat_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.2
    lambda_cls: float = 1.0      # Classification loss weight
    lambda_reg: float = 0.5      # Regression loss weight
    grad_clip: float = 5.0


class GATMultiTaskClsReg(nn.Module):
    """Multi-task GAT for BBB classification and logBB regression"""
    def __init__(self, in_dim: int, hidden: int, heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(GATConv(in_dim, hidden, heads=heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads, concat=True))
        self.convs.append(GATConv(hidden * heads, hidden, heads=1, concat=True))

        # Classification head (BBB+/-)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

        # Regression head (logBB)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch_idx)
        cls_logit = self.cls_head(g).view(-1)
        reg_pred = self.reg_head(g).view(-1)
        return cls_logit, reg_pred


def train_multitask_cls_reg(train_ds, val_ds, test_ds, out_dir: Path, cfg: MultiTaskTrainCfg):
    """Train multi-task model"""
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    in_dim = train_ds[0].x.size(-1)
    model = GATMultiTaskClsReg(in_dim, cfg.hidden, cfg.gat_heads, cfg.num_layers, cfg.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last = out_dir / "last.pt"
    ckpt_best = out_dir / "best.pt"

    best_val_auc = -1.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            cls_logit, reg_pred = model(batch)

            # Classification loss (all samples)
            y_cls = batch.y_cls.view(-1)
            loss_cls = bce_loss(cls_logit, y_cls)

            # Regression loss (only samples with logBB labels)
            # Check if y_logp is valid (not NaN)
            has_logbb = batch.y_logp.view(-1) > -100  # Placeholder check
            if has_logbb.sum() > 0:
                y_reg = batch.y_logp.view(-1)[has_logbb]
                reg_pred_valid = reg_pred[has_logbb]
                loss_reg = mse_loss(reg_pred_valid, y_reg)
            else:
                loss_reg = torch.tensor(0.0, device=device)

            # Combined loss
            loss = cfg.lambda_cls * loss_cls + cfg.lambda_reg * loss_reg

            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            total_loss += float(loss.item()) * batch.num_graphs
            total_cls_loss += float(loss_cls.item()) * batch.num_graphs
            if loss_reg.item() > 0:
                total_reg_loss += float(loss_reg.item()) * has_logbb.sum()

        # Validation
        model.eval()
        with torch.no_grad():
            val_cls_losses = []
            val_reg_losses = []
            val_y_true, val_y_prob = [], []
            val_reg_true, val_reg_pred = [], []

            for batch in val_loader:
                batch = batch.to(device)
                cls_logit, reg_pred = model(batch)

                y_cls = batch.y_cls.view(-1)
                val_cls_losses.append(float(bce_loss(cls_logit, y_cls).item()) * len(y_cls))

                prob = torch.sigmoid(cls_logit).detach().cpu().numpy()
                val_y_true.append(y_cls.cpu().numpy().astype(int))
                val_y_prob.append(prob)

                # Regression (valid samples only)
                has_logbb = batch.y_logp.view(-1) > -100
                if has_logbb.sum() > 0:
                    y_reg = batch.y_logp.view(-1)[has_logbb]
                    reg_pred_valid = reg_pred[has_logbb]
                    val_reg_losses.append(float(mse_loss(reg_pred_valid, y_reg).item()) * len(y_reg))
                    val_reg_true.append(y_reg.cpu().numpy())
                    val_reg_pred.append(reg_pred_valid.detach().cpu().numpy())

            val_cls_loss = sum(val_cls_losses) / len(val_ds)
            val_reg_loss = sum(val_reg_losses) / sum(len(x) for x in val_reg_true) if val_reg_true else 0.0

            val_y_true = np.concatenate(val_y_true)
            val_y_prob = np.concatenate(val_y_prob)
            val_m = classification_metrics(val_y_true, val_y_prob, threshold=0.5)

            if val_reg_true:
                val_reg_true_all = np.concatenate(val_reg_true)
                val_reg_pred_all = np.concatenate(val_reg_pred)
                val_reg_mae = float(np.mean(np.abs(val_reg_true_all - val_reg_pred_all)))
                val_reg_rmse = float(np.sqrt(np.mean((val_reg_true_all - val_reg_pred_all) ** 2)))
            else:
                val_reg_mae = val_reg_rmse = float('nan')

        history.append({
            "seed": cfg.seed,
            "epoch": epoch,
            "train_loss_avg": total_loss / len(train_ds),
            "train_cls_loss": total_cls_loss / len(train_ds),
            "train_reg_loss": total_reg_loss / sum(has_logbb.sum() for _ in train_loader) if total_reg_loss > 0 else 0.0,
            "val_cls_loss": val_cls_loss,
            "val_reg_loss": val_reg_loss,
            "val_auc": val_m.auc,
            "val_reg_mae": val_reg_mae,
            "val_reg_rmse": val_reg_rmse,
        })

        torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch}, ckpt_last)

        if val_m.auc > best_val_auc:
            best_val_auc = val_m.auc
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch}, ckpt_best)

    # Load best and evaluate
    best = torch.load(ckpt_best, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])

    model.eval()
    with torch.no_grad():
        test_y_true, test_y_prob = [], []
        test_reg_true, test_reg_pred = [], []

        for batch in test_loader:
            batch = batch.to(device)
            cls_logit, reg_pred = model(batch)

            prob = torch.sigmoid(cls_logit).detach().cpu().numpy()
            y = batch.y_cls.view(-1).detach().cpu().numpy().astype(int)
            test_y_true.append(y)
            test_y_prob.append(prob)

            has_logbb = batch.y_logp.view(-1) > -100
            if has_logbb.sum() > 0:
                y_reg = batch.y_logp.view(-1)[has_logbb]
                reg_pred_valid = reg_pred[has_logbb]
                test_reg_true.append(y_reg.cpu().numpy())
                test_reg_pred.append(reg_pred_valid.detach().cpu().numpy())

        test_y_true = np.concatenate(test_y_true)
        test_y_prob = np.concatenate(test_y_prob)
        test_m = classification_metrics(test_y_true, test_y_prob, threshold=0.5)

        if test_reg_true:
            test_reg_true_all = np.concatenate(test_reg_true)
            test_reg_pred_all = np.concatenate(test_reg_pred)
            test_reg_mae = float(np.mean(np.abs(test_reg_true_all - test_reg_pred_all)))
            test_reg_rmse = float(np.sqrt(np.mean((test_reg_true_all - test_reg_pred_all) ** 2)))
            test_reg_r2 = float(1 - np.sum((test_reg_true_all - test_reg_pred_all)**2) / np.sum((test_reg_true_all - test_reg_true_all.mean())**2))
        else:
            test_reg_mae = test_reg_rmse = test_reg_r2 = float('nan')

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)

    final_row = {
        "seed": cfg.seed,
        "model": "GAT_MultiTask_ClsReg",
        "feature": "graph",
        "lambda_cls": cfg.lambda_cls,
        "lambda_reg": cfg.lambda_reg,
        "best_epoch": int(best["epoch"]),
        # Classification metrics
        "test_auc": test_m.auc,
        "test_auprc": test_m.auprc,
        "test_accuracy": test_m.accuracy,
        "test_precision_pos": test_m.precision_pos,
        "test_recall_pos": test_m.recall_pos,
        "test_f1_pos": test_m.f1_pos,
        "tn": test_m.tn, "fp": test_m.fp, "fn": test_m.fn, "tp": test_m.tp,
        # Regression metrics
        "test_reg_mae": test_reg_mae,
        "test_reg_rmse": test_reg_rmse,
        "test_reg_r2": test_reg_r2,
    }

    plot_roc_curves(
        [{"name": "GAT MultiTask (Cls+Reg)", "y_true": test_y_true, "y_prob": test_y_prob}],
        out_dir / "roc_multitask.png",
        title=f"GAT MultiTask ROC (seed={cfg.seed})"
    )

    return final_row, out_dir / "roc_multitask.png"
