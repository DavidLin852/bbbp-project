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
from sklearn.metrics import f1_score, average_precision_score

from ..utils.seed import seed_everything

@dataclass
class PretrainCfg:
    seed: int = 0
    device: str = "cuda"
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    batch_size: int = 64
    hidden: int = 128
    gat_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.2
    pos_weight_clip: float = 20.0  # 稀有标签的权重上限
    min_freq: float = 0.01         # 低于该频率的标签不参与loss（避免噪声）

class GATBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int, heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden, heads=heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads, concat=True))
        self.convs.append(GATConv(hidden * heads, hidden, heads=1, concat=True))

    def forward(self, x, edge_index, batch_idx):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch_idx)
        return g

class GATSmartsPretrain(nn.Module):
    def __init__(self, in_dim: int, hidden: int, heads: int, num_layers: int, dropout: float, num_labels: int):
        super().__init__()
        self.backbone = GATBackbone(in_dim, hidden, heads, num_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels)  # multi-label logits
        )

    def forward(self, batch):
        g = self.backbone(batch.x, batch.edge_index, batch.batch)
        logits = self.head(g)
        return logits

@torch.no_grad()
def eval_epoch(model, loader, device, label_mask):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch.y_smarts[:, label_mask].detach().cpu().numpy()
        ys.append(y); ps.append(prob)
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)


    # threshold 0.5 for F1
    yhat = (p >= 0.5).astype(int)

    # 只保留在 y 或 yhat 中出现过的 label
    valid = (y.sum(axis=0) + yhat.sum(axis=0)) > 0
    if valid.sum() == 0:
        macro_f1 = 0.0
        micro_f1 = 0.0
    else:
        macro_f1 = float(
            f1_score(y[:, valid], yhat[:, valid],
                     average="macro", zero_division=0)
        )
        micro_f1 = float(
            f1_score(y[:, valid], yhat[:, valid],
                     average="micro", zero_division=0)
        )

    # AUPRC per label, then macro
    auprc_each = []
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0:
            continue
        auprc_each.append(average_precision_score(y[:, j], p[:, j]))
    macro_auprc = float(np.mean(auprc_each)) if len(auprc_each) else 0.0

    return macro_f1, micro_f1, macro_auprc

def pretrain_smarts(train_ds, val_ds, out_dir: Path, cfg: PretrainCfg):
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    in_dim = train_ds[0].x.size(-1)
    num_labels = train_ds[0].y_smarts.numel()

    # 计算标签频率，筛掉过稀标签
    y_all = []
    for d in train_ds:
        y_all.append(d.y_smarts.numpy())
    y_all = np.stack(y_all, axis=0)
    freq = y_all.mean(axis=(0, 1))
    label_mask = (freq >= cfg.min_freq)
    kept = int(label_mask.sum())

    # pos_weight：解决多标签极不均衡
    eps = 1e-6
    pos_weight = (1.0 - freq[label_mask] + eps) / (freq[label_mask] + eps)
    pos_weight = np.clip(pos_weight, 1.0, cfg.pos_weight_clip)
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float, device=device)

    model = GATSmartsPretrain(in_dim, cfg.hidden, cfg.gat_heads, cfg.num_layers, cfg.dropout, kept).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = out_dir / "best.pt"
    ckpt_last = out_dir / "last.pt"
    hist = []
    best_val_macro_f1 = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            y = batch.y_smarts[:, label_mask].to(device)
            logits = model(batch)

            loss = bce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * batch.num_graphs

        val_macro_f1, val_micro_f1, val_macro_auprc = eval_epoch(
            model, val_loader, device, label_mask
        )

        hist.append({
            "seed": cfg.seed,
            "epoch": epoch,
            "kept_labels": kept,
            "train_loss_avg": total / len(train_ds),
            "val_macro_f1": val_macro_f1,
            "val_micro_f1": val_micro_f1,
            "val_macro_auprc": val_macro_auprc,
        })

        torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch, "label_mask": label_mask}, ckpt_last)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch, "label_mask": label_mask}, ckpt_best)

    pd.DataFrame(hist).to_csv(out_dir / "pretrain_history.csv", index=False)
    return ckpt_best, out_dir / "pretrain_history.csv"
