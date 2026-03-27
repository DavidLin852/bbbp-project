from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from ..pretrain.backbone_gat import GATBackbone
from ..utils.seed import seed_everything
from ..utils.metrics import classification_metrics
from ..utils.plotting import plot_roc_curves

@dataclass
class FinetuneCfg:
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
    grad_clip: float = 5.0
    strategy: str = "full"  # freeze | partial | full
    partial_k: int = 2      # freeze first k conv layers (for partial)
    init: str = "pretrained" # pretrained | random | shuffled

class BBBHead(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

class GATBBB(nn.Module):
    def __init__(self, in_dim: int, cfg: FinetuneCfg):
        super().__init__()
        self.backbone = GATBackbone(in_dim, cfg.hidden, cfg.gat_heads, cfg.num_layers, cfg.dropout)
        self.head = BBBHead(cfg.hidden, cfg.dropout)

    def forward(self, data):
        x = self.backbone(data)
        out = self.head(x)
        return out

def _apply_strategy(model: GATBBB, cfg: FinetuneCfg):
    # default: all trainable
    for p in model.parameters():
        p.requires_grad = True

    if cfg.strategy == "freeze":
        # freeze all backbone
        for p in model.backbone.parameters():
            p.requires_grad = False

    elif cfg.strategy == "partial":
        # freeze first k conv layers in backbone
        k = int(cfg.partial_k)
        for i, conv in enumerate(model.backbone.convs):
            if i < k:
                for p in conv.parameters():
                    p.requires_grad = False

    elif cfg.strategy == "full":
        pass
    else:
        raise ValueError(f"Unknown strategy: {cfg.strategy}")

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        batch = batch.to(device)
        logit = model(batch)
        prob = torch.sigmoid(logit).detach().cpu().numpy()
        y = batch.y_cls.view(-1).detach().cpu().numpy().astype(int)
        y_true.append(y); y_prob.append(prob)
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    m = classification_metrics(y_true, y_prob, threshold=0.5)
    return m, y_true, y_prob

def _load_pretrained_backbone(model: GATBBB, pretrain_ckpt: Path, init: str):
    ckpt = torch.load(pretrain_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["model"]

    # pretrain model keys include: backbone.convs.*, head.* (SMARTS head)
    # We only copy backbone weights.
    bb_sd = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            bb_sd[k.replace("backbone.", "")] = v

    if init == "pretrained":
        model.backbone.load_state_dict(bb_sd, strict=True)

    elif init == "shuffled":
        # Same tensor shapes, but permute values to destroy structure knowledge.
        shuffled = {}
        for k, v in bb_sd.items():
            if torch.is_floating_point(v) and v.numel() > 1:
                flat = v.flatten()
                idx = torch.randperm(flat.numel())
                shuffled[k] = flat[idx].view_as(v)
            else:
                shuffled[k] = v
        model.backbone.load_state_dict(shuffled, strict=True)

    elif init == "random":
        # leave random init
        return
    else:
        raise ValueError(f"Unknown init: {init}")

def finetune_bbb(train_ds, val_ds, test_ds, out_dir: Path, cfg: FinetuneCfg, pretrain_ckpt: Path):
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    in_dim = train_ds[0].x.size(-1)
    model = GATBBB(in_dim, cfg).to(device)

    # init
    if cfg.init in ("pretrained", "shuffled"):
        _load_pretrained_backbone(model, pretrain_ckpt, cfg.init)

    _apply_strategy(model, cfg)

    # optimizer only on trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()

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
            logit = model(batch).view(-1)
            y = batch.y_cls.view(-1)
            loss = bce(logit, y)

            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            total_loss += float(loss.item()) * batch.num_graphs

        val_m, _, _ = eval_epoch(model, val_loader, device)
        test_m, y_true_test, y_prob_test = eval_epoch(model, test_loader, device)

        history.append({
            "seed": cfg.seed,
            "epoch": epoch,
            "init": cfg.init,
            "strategy": cfg.strategy,
            "train_loss_avg": total_loss / len(train_ds),
            "val_auc": val_m.auc,
            "val_auprc": val_m.auprc,
            "test_auc": test_m.auc,
            "test_auprc": test_m.auprc,
            "test_fp": test_m.fp,
            "test_fn": test_m.fn,
        })

        torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch}, ckpt_last)
        if val_m.auc > best_val_auc:
            best_val_auc = val_m.auc
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch}, ckpt_best)

    # final eval on best
    best = torch.load(ckpt_best, map_location=device)
    model.load_state_dict(best["model"])
    test_m, y_true_test, y_prob_test = eval_epoch(model, test_loader, device)

    pd.DataFrame(history).to_csv(out_dir / "finetune_history.csv", index=False)

    row = {
        "seed": cfg.seed,
        "model": "GAT_BBB",
        "feature": "graph",
        "pretrain": "smarts_v1",
        "init": cfg.init,
        "strategy": cfg.strategy,
        "partial_k": cfg.partial_k if cfg.strategy == "partial" else "",
        "best_epoch": int(best["epoch"]),
        "test_auc": test_m.auc,
        "test_auprc": test_m.auprc,
        "test_accuracy": test_m.accuracy,
        "test_precision_pos": test_m.precision_pos,
        "test_recall_pos": test_m.recall_pos,
        "test_f1_pos": test_m.f1_pos,
        "tn": test_m.tn, "fp": test_m.fp, "fn": test_m.fn, "tp": test_m.tp,
    }

    roc_path = out_dir / "roc_bbb.png"
    plot_roc_curves(
        [{"name": f"{cfg.init}-{cfg.strategy}", "y_true": y_true_test, "y_prob": y_prob_test}],
        roc_path,
        title=f"BBB ROC (seed={cfg.seed})"
    )
    return row, roc_path
