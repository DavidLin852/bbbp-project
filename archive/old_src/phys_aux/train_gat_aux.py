from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from ..utils.metrics import classification_metrics
from ..utils.plotting import plot_roc_curves
from ..utils.seed import seed_everything

@dataclass
class TrainCfg:
    seed: int = 0
    device: str = "cuda"
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    batch_size: int = 64
    hidden: int = 128
    gat_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.2
    lambda_logp: float = 0.3
    lambda_tpsa: float = 0.3
    grad_clip: float = 5.0

class GATMultiTask(nn.Module):
    def __init__(self, in_dim: int, hidden: int, heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(GATConv(in_dim, hidden, heads=heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads, concat=True))
        self.convs.append(GATConv(hidden * heads, hidden, heads=1, concat=True))

        self.cls_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.logp_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.tpsa_head = nn.Sequential(
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
        logp_pred = self.logp_head(g).view(-1)
        tpsa_pred = self.tpsa_head(g).view(-1)
        return cls_logit, logp_pred, tpsa_pred

def _mae_rmse(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return mae, rmse

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    logp_true, logp_pred = [], []
    tpsa_true, tpsa_pred = [], []

    for batch in loader:
        batch = batch.to(device)
        cls_logit, lp, tp = model(batch)

        prob = torch.sigmoid(cls_logit).detach().cpu().numpy()
        y = batch.y_cls.view(-1).detach().cpu().numpy().astype(int)
        y_true.append(y)
        y_prob.append(prob)

        logp_true.append(batch.y_logp.view(-1).detach().cpu().numpy())
        logp_pred.append(lp.detach().cpu().numpy())

        tpsa_true.append(batch.y_tpsa.view(-1).detach().cpu().numpy())
        tpsa_pred.append(tp.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)

    logp_true = np.concatenate(logp_true)
    logp_pred = np.concatenate(logp_pred)

    tpsa_true = np.concatenate(tpsa_true)
    tpsa_pred = np.concatenate(tpsa_pred)

    m = classification_metrics(y_true, y_prob, threshold=0.5)
    logp_mae, logp_rmse = _mae_rmse(logp_pred, logp_true)
    tpsa_mae, tpsa_rmse = _mae_rmse(tpsa_pred, tpsa_true)

    return m, (logp_mae, logp_rmse), (tpsa_mae, tpsa_rmse), y_true, y_prob

def train_gat_multitask(train_ds, val_ds, test_ds, out_dir: Path, cfg: TrainCfg):
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    in_dim = train_ds[0].x.size(-1)
    model = GATMultiTask(in_dim, cfg.hidden, cfg.gat_heads, cfg.num_layers, cfg.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last = out_dir / "last.pt"
    ckpt_best = out_dir / "best.pt"

    best_val_auc = -1.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            cls_logit, logp_p, tpsa_p = model(batch)

            y_cls = batch.y_cls.view(-1)
            y_logp = batch.y_logp.view(-1)
            y_tpsa = batch.y_tpsa.view(-1)

            loss_cls = bce(cls_logit, y_cls)
            loss_logp = mse(logp_p, y_logp)
            loss_tpsa = mse(tpsa_p, y_tpsa)

            loss = loss_cls + cfg.lambda_logp * loss_logp + cfg.lambda_tpsa * loss_tpsa

            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            total_loss += float(loss.item()) * batch.num_graphs

        val_m, (val_logp_mae, val_logp_rmse), (val_tpsa_mae, val_tpsa_rmse), _, _ = eval_epoch(model, val_loader, device)
        test_m, (test_logp_mae, test_logp_rmse), (test_tpsa_mae, test_tpsa_rmse), y_true_test, y_prob_test = eval_epoch(model, test_loader, device)

        history.append({
            "seed": cfg.seed,
            "epoch": epoch,
            "lambda_logp": cfg.lambda_logp,
            "lambda_tpsa": cfg.lambda_tpsa,
            "train_loss_avg": total_loss / len(train_ds),
            "val_auc": val_m.auc,
            "val_auprc": val_m.auprc,
            "val_logp_mae": val_logp_mae,
            "val_logp_rmse": val_logp_rmse,
            "val_tpsa_mae": val_tpsa_mae,
            "val_tpsa_rmse": val_tpsa_rmse,
            "test_auc": test_m.auc,
            "test_auprc": test_m.auprc,
            "test_logp_mae": test_logp_mae,
            "test_logp_rmse": test_logp_rmse,
            "test_tpsa_mae": test_tpsa_mae,
            "test_tpsa_rmse": test_tpsa_rmse,
            "test_fp": test_m.fp,
            "test_fn": test_m.fn,
        })

        torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch}, ckpt_last)

        if val_m.auc > best_val_auc:
            best_val_auc = val_m.auc
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch}, ckpt_best)

    # load best and final test
    best = torch.load(ckpt_best, map_location=device)
    model.load_state_dict(best["model"])
    test_m, (test_logp_mae, test_logp_rmse), (test_tpsa_mae, test_tpsa_rmse), y_true_test, y_prob_test = eval_epoch(model, test_loader, device)

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)

    final_row = {
        "seed": cfg.seed,
        "model": "GAT_MTL",
        "feature": "graph",
        "aux": "logP+TPSA",
        "lambda_logp": cfg.lambda_logp,
        "lambda_tpsa": cfg.lambda_tpsa,
        "best_epoch": int(best["epoch"]),
        "test_auc": test_m.auc,
        "test_auprc": test_m.auprc,
        "test_accuracy": test_m.accuracy,
        "test_precision_pos": test_m.precision_pos,
        "test_recall_pos": test_m.recall_pos,
        "test_f1_pos": test_m.f1_pos,
        "tn": test_m.tn, "fp": test_m.fp, "fn": test_m.fn, "tp": test_m.tp,
        "test_logp_mae": test_logp_mae,
        "test_logp_rmse": test_logp_rmse,
        "test_tpsa_mae": test_tpsa_mae,
        "test_tpsa_rmse": test_tpsa_rmse,
    }

    plot_roc_curves(
        [{"name": "GAT_MTL(logP+TPSA)", "y_true": y_true_test, "y_prob": y_prob_test}],
        out_dir / "roc_gat_mtl.png",
        title=f"GAT Multi-task ROC (seed={cfg.seed})"
    )

    return final_row, out_dir / "roc_gat_mtl.png"
