"""
ZINC20 Pretraining Pipeline

Pretrain GNN models on large-scale molecular data from ZINC20.

Usage:
    # Step 1: Download and prepare ZINC20 data
    python pretrain_zinc20.py --step download --num-molecules 1000000

    # Step 2: Pretrain with multi-task learning
    python pretrain_zinc20.py --step pretrain --epochs 100 --batch-size 256

    # Step 3: Fine-tune on BBB task
    python pretrain_zinc20.py --step finetune --epochs 50
"""
from __future__ import annotations
import sys
from pathlib import Path
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pretrain.zinc20_loader import (
    ZINC20GraphDataset,
    ZINC20StreamingDataset,
    download_zinc20_tranches,
    create_zinc20_splits,
    compute_zinc_properties
)
from src.pretrain.zinc20_pretrain import (
    ZINC20PretrainModel,
    ZINC20ContextOnly,
    ZINC20PropertyOnly,
    PretrainConfig,
    load_pretrained_backbone
)
from src.utils.seed import seed_everything


@dataclass
class TrainConfig:
    """Training configuration"""
    # Data
    data_dir: str = "data/zinc20"
    num_molecules: int = 1_000_000

    # Training
    seed: int = 42
    epochs: int = 100
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # Model
    hidden: int = 128
    heads: int = 4
    num_layers: int = 3
    dropout: float = 0.2

    # Pretraining tasks
    use_context: bool = True
    use_property: bool = True
    use_mask: bool = False  # Disabled by default (slower)
    lambda_context: float = 1.0
    lambda_property: float = 1.0
    lambda_mask: float = 0.5

    # System
    device: str = "cuda"
    num_workers: int = 4
    log_interval: int = 10
    save_interval: int = 10


def step_download_zinc20(args):
    """Download ZINC20 dataset"""
    print("\n" + "=" * 80)
    print("ZINC20 Data Download")
    print("=" * 80)

    output_dir = PROJECT_ROOT / args.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles_file = download_zinc20_tranches(
        output_dir=output_dir,
        num_molecules=args.num_molecules,
        tranches=None,  # Sample from all tranches
        seed=args.seed,
        verbose=True
    )

    print(f"\nSMILES saved to: {smiles_file}")

    # Verify file
    if not smiles_file.exists() or smiles_file.stat().st_size == 0:
        raise RuntimeError(f"Download failed: {smiles_file} is empty or missing")

    # Optional: filter by property range for drug-like molecules
    print("\nFiltering for drug-like molecules...")
    df = pd.read_csv(smiles_file)

    if 'SMILES' not in df.columns or len(df) == 0:
        raise RuntimeError(f"Invalid data in {smiles_file}")

    valid_smiles = []
    for smi in df['SMILES']:
        props = compute_zinc_properties(smi)
        if props is None:
            continue

        # Lipinski's Rule of 5 (loose filter)
        if (props.mw >= 150 and props.mw <= 500 and
            props.logp >= -2 and props.logp <= 5 and
            props.num_hbd <= 5 and props.num_hba <= 10):
            valid_smiles.append(smi)

    filtered_file = output_dir / f"zinc20_filtered_{len(valid_smiles)}.csv"
    pd.DataFrame({'SMILES': valid_smiles}).to_csv(filtered_file, index=False)

    print(f"Filtered from {len(df):,} to {len(valid_smiles):,} drug-like molecules")
    print(f"Saved to: {filtered_file}")

    return filtered_file


def step_prepare_datasets(args, smiles_file: Path):
    """Create train/val/test splits and prepare graph datasets"""
    print("\n" + "=" * 80)
    print("Preparing Graph Datasets")
    print("=" * 80)

    # Create splits
    split_dir = PROJECT_ROOT / args.data_dir / "splits" / f"seed_{args.seed}"
    create_zinc20_splits(
        smiles_file=smiles_file,
        output_dir=split_dir,
        train_ratio=0.9,
        val_ratio=0.05,
        test_ratio=0.05,
        seed=args.seed
    )

    # For large datasets, use streaming to avoid OOM
    use_streaming = args.num_molecules > 500000

    if use_streaming:
        print("Using streaming dataset (large-scale)")
        cache_dir = PROJECT_ROOT / "artifacts" / "features" / "zinc20"

        train_ds = ZINC20StreamingDataset(
            smiles_file=split_dir / "train.csv",
            cache_dir=cache_dir / "train",
            context_pred=args.use_context
        )
        val_ds = ZINC20StreamingDataset(
            smiles_file=split_dir / "val.csv",
            cache_dir=cache_dir / "val",
            context_pred=args.use_context
        )
    else:
        print("Using in-memory dataset")
        cache_root = PROJECT_ROOT / "artifacts" / "features" / "zinc20" / f"seed_{args.seed}"
        cache_root.mkdir(parents=True, exist_ok=True)

        train_ds = ZINC20GraphDataset(
            root=str(cache_root / "train"),
            smiles_file=split_dir / "train.csv",
            context_pred=args.use_context
        )
        val_ds = ZINC20GraphDataset(
            root=str(cache_root / "val"),
            smiles_file=split_dir / "val.csv",
            context_pred=args.use_context
        )

    print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    return train_ds, val_ds


def train_epoch(model, loader, optimizer, cfg, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_context = 0.0
    total_property = 0.0
    total_mask = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)

        # Generate mask for reconstruction task
        mask_indices = None
        if cfg.use_mask:
            mask_indices = model.generate_mask(batch.batch)

        # Forward
        losses = model(batch, mask_indices=mask_indices, return_components=True)
        loss = losses['loss']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # Track
        total_loss += loss.item()
        if 'context_loss' in losses:
            total_context += losses['context_loss'].item()
        if 'property_loss' in losses:
            total_property += losses['property_loss'].item()
        if 'mask_loss' in losses:
            total_mask += losses['mask_loss'].item()

        num_batches += 1

        if (batch_idx + 1) % cfg.log_interval == 0:
            avg_loss = total_loss / num_batches
            print(f"  Batch {batch_idx + 1}/{len(loader)}: loss={avg_loss:.4f}")

    return {
        'loss': total_loss / num_batches,
        'context_loss': total_context / num_batches if num_batches > 0 else 0.0,
        'property_loss': total_property / num_batches if num_batches > 0 else 0.0,
        'mask_loss': total_mask / num_batches if num_batches > 0 else 0.0,
    }


@torch.no_grad()
def eval_epoch(model, loader, cfg, device):
    """Evaluate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_context = 0.0
    total_property = 0.0
    total_mask = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)

        losses = model(batch, mask_indices=None, return_components=True)
        loss = losses['loss']

        total_loss += loss.item()
        if 'context_loss' in losses:
            total_context += losses['context_loss'].item()
        if 'property_loss' in losses:
            total_property += losses['property_loss'].item()
        if 'mask_loss' in losses:
            total_mask += losses['mask_loss'].item()

        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'context_loss': total_context / num_batches if num_batches > 0 else 0.0,
        'property_loss': total_property / num_batches if num_batches > 0 else 0.0,
        'mask_loss': total_mask / num_batches if num_batches > 0 else 0.0,
    }


def step_pretrain(args, train_ds, val_ds):
    """Pretrain on ZINC20"""
    print("\n" + "=" * 80)
    print("ZINC20 Pretraining")
    print("=" * 80)

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create config
    train_cfg = TrainConfig(
        num_molecules=args.num_molecules,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        heads=args.heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_context=args.use_context,
        use_property=args.use_property,
        use_mask=args.use_mask,
        lambda_context=args.lambda_context,
        lambda_property=args.lambda_property,
        lambda_mask=args.lambda_mask,
    )

    # Create model
    in_dim = train_ds[0].x.size(-1)
    num_atom_types = train_ds[0].context.size(-1) if args.use_context else 9
    num_props = train_ds[0].props.size(-1) if args.use_property else 9

    model_cfg = PretrainConfig(
        in_dim=in_dim,
        hidden=args.hidden,
        heads=args.heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_atom_types=num_atom_types,
        num_props=num_props,
        lambda_context=args.lambda_context,
        lambda_property=args.lambda_property,
        lambda_mask=args.lambda_mask,
        mask_ratio=0.15 if args.use_mask else 0.0,
    )

    model = ZINC20PretrainModel(model_cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5
    )

    # Data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Output directory
    out_dir = PROJECT_ROOT / "artifacts" / "models" / "zinc20_pretrain" / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    history = []
    best_val_loss = float('inf')

    print(f"\nTraining for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, train_cfg, device, epoch)

        # Validate
        val_metrics = eval_epoch(model, val_loader, train_cfg, device)

        # Step scheduler
        scheduler.step()

        # Log
        lr = optimizer.param_groups[0]['lr']
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  LR:         {lr:.6f}")

        if train_metrics['context_loss'] > 0:
            print(f"  Context:    {train_metrics['context_loss']:.4f}")
        if train_metrics['property_loss'] > 0:
            print(f"  Property:   {train_metrics['property_loss']:.4f}")

        # Save history
        history.append({
            'epoch': epoch,
            'lr': lr,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_context': train_metrics['context_loss'],
            'train_property': train_metrics['property_loss'],
            'val_context': val_metrics['context_loss'],
            'val_property': val_metrics['property_loss'],
        })

        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            ckpt_path = out_dir / "best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'cfg': asdict(model_cfg),
                'train_cfg': asdict(train_cfg),
            }, ckpt_path)
            print(f"  * Saved best checkpoint (val_loss={val_metrics['loss']:.4f})")

        if epoch % args.save_interval == 0:
            ckpt_path = out_dir / f"epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'cfg': asdict(model_cfg),
                'train_cfg': asdict(train_cfg),
            }, ckpt_path)

    # Save final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['loss'],
        'cfg': asdict(model_cfg),
        'train_cfg': asdict(train_cfg),
    }, out_dir / "last.pt")

    # Save history
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {out_dir}")

    return out_dir / "best.pt"


def step_finetune_bbb(args, pretrain_ckpt: Path):
    """Fine-tune pretrained model on BBB task"""
    print("\n" + "=" * 80)
    print("Fine-tuning on BBB Task")
    print("=" * 80)

    from src.featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig
    from src.utils.metrics import classification_metrics

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load BBB data
    split_dir = PROJECT_ROOT / "data" / "splits" / f"seed_{args.seed}_full"
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    gcfg = GraphBuildConfig(smiles_col="SMILES", label_col="y_cls", id_col="row_id")
    cache_root = PROJECT_ROOT / "artifacts" / "features" / f"seed_{args.seed}_full" / "pyg_graphs_baseline"

    train_ds = BBBGraphDataset(root=str(cache_root / "train"), df=train_df, cfg=gcfg)
    val_ds = BBBGraphDataset(root=str(cache_root / "val"), df=val_df, cfg=gcfg)
    test_ds = BBBGraphDataset(root=str(cache_root / "test"), df=test_df, cfg=gcfg)

    print(f"BBB datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Load pretrained config
    checkpoint = torch.load(pretrain_ckpt, map_location='cpu', weights_only=False)
    pretrain_cfg_dict = checkpoint['cfg']
    pretrain_cfg = PretrainConfig(**pretrain_cfg_dict)

    # Load pretrained backbone
    backbone = load_pretrained_backbone(pretrain_ckpt, pretrain_cfg, freeze=False)

    # Create BBB classifier
    class BBBClassifier(nn.Module):
        def __init__(self, backbone, hidden_dim, dropout=0.2):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, batch):
            graph_emb = self.backbone(batch)
            return self.classifier(graph_emb).view(-1)

    model = BBBClassifier(backbone, pretrain_cfg.hidden, pretrain_cfg.dropout).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    out_dir = PROJECT_ROOT / "artifacts" / "models" / "zinc20_finetuned" / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = -1.0
    epochs = 50

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            logit = model(batch)
            y = batch.y_cls.view(-1)
            loss = criterion(logit, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        # Validate
        model.eval()
        with torch.no_grad():
            val_probs, val_labels = [], []
            for batch in val_loader:
                batch = batch.to(device)
                logit = model(batch)
                prob = torch.sigmoid(logit).cpu().numpy()
                y = batch.y_cls.view(-1).cpu().numpy().astype(int)
                val_probs.append(prob)
                val_labels.append(y)

            val_probs = np.concatenate(val_probs)
            val_labels = np.concatenate(val_labels)
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.0

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), out_dir / "best.pt")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}: Loss={total_loss/len(train_ds):.4f}, Val AUC={val_auc:.4f}")

    # Test
    model.load_state_dict(torch.load(out_dir / "best.pt"))
    model.eval()

    with torch.no_grad():
        test_probs, test_labels = [], []
        for batch in test_loader:
            batch = batch.to(device)
            logit = model(batch)
            prob = torch.sigmoid(logit).cpu().numpy()
            y = batch.y_cls.view(-1).cpu().numpy().astype(int)
            test_probs.append(prob)
            test_labels.append(y)

        test_probs = np.concatenate(test_probs)
        test_labels = np.concatenate(test_labels)

    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    auc = roc_auc_score(test_labels, test_probs)
    y_pred = (test_probs >= 0.5).astype(int)
    precision = precision_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred)

    print(f"\n[ZINC20 Fine-tuned] Test Results:")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    return {
        'model': 'ZINC20_Pretrained',
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


def main():
    ap = argparse.ArgumentParser(description="ZINC20 Pretraining Pipeline")

    # Step selection
    ap.add_argument("--step", type=str, required=True,
                    choices=["download", "pretrain", "finetune"],
                    help="Pipeline step to run")

    # Data args
    ap.add_argument("--data-dir", type=str, default="data/zinc20",
                    help="Directory for ZINC20 data")
    ap.add_argument("--num-molecules", type=int, default=1_000_000,
                    help="Number of molecules to download/use")

    # Training args
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--save-interval", type=int, default=10)

    # Task selection
    ap.add_argument("--use-context", action="store_true", default=True,
                    help="Use context prediction task")
    ap.add_argument("--no-context", action="store_false", dest="use_context")
    ap.add_argument("--use-property", action="store_true", default=True,
                    help="Use property prediction task")
    ap.add_argument("--no-property", action="store_false", dest="use_property")
    ap.add_argument("--use-mask", action="store_true", default=False,
                    help="Use masked reconstruction task")

    # Task weights
    ap.add_argument("--lambda-context", type=float, default=1.0)
    ap.add_argument("--lambda-property", type=float, default=1.0)
    ap.add_argument("--lambda-mask", type=float, default=0.5)

    # System
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)

    # Fine-tuning
    ap.add_argument("--pretrain-ckpt", type=str,
                    help="Path to pretrained checkpoint for fine-tuning")

    args = ap.parse_args()

    try:
        if args.step == "download":
            smiles_file = step_download_zinc20(args)
            print(f"\nNext step: Prepare datasets and pretrain")
            print(f"  python pretrain_zinc20.py --step pretrain --data-dir {args.data_dir}")

        elif args.step == "pretrain":
            # Check for data
            data_dir = PROJECT_ROOT / args.data_dir
            smiles_files = list(data_dir.glob("zinc20_*.csv"))
            smiles_files += list(data_dir.glob("splits/*/train.csv"))

            if not smiles_files:
                print(f"No ZINC20 data found in {data_dir}")
                print("Please run: python pretrain_zinc20.py --step download")
                return

            # Use the most recent file
            smiles_file = sorted(smiles_files, key=lambda p: p.stat().st_mtime)[-1]
            if "splits" not in str(smiles_file):
                train_ds, val_ds = step_prepare_datasets(args, smiles_file)
            else:
                # Already split
                split_dir = smiles_file.parent
                smiles_file = list(data_dir.glob("zinc20_*.csv"))[0]
                train_ds, val_ds = step_prepare_datasets(args, smiles_file)

            ckpt_path = step_pretrain(args, train_ds, val_ds)

            print(f"\nPretraining complete!")
            print(f"Checkpoint: {ckpt_path}")
            print(f"\nNext step: Fine-tune on BBB task")
            print(f"  python pretrain_zinc20.py --step finetune --pretrain-ckpt {ckpt_path}")

        elif args.step == "finetune":
            if not args.pretrain_ckpt:
                # Find latest checkpoint
                ckpt_dir = PROJECT_ROOT / "artifacts" / "models" / "zinc20_pretrain"
                ckpts = list(ckpt_dir.glob("*/best.pt"))
                if not ckpts:
                    print("No pretrained checkpoint found!")
                    print("Please run pretraining first")
                    return
                pretrain_ckpt = sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1]
                print(f"Using checkpoint: {pretrain_ckpt}")
            else:
                pretrain_ckpt = Path(args.pretrain_ckpt)

            result = step_finetune_bbb(args, pretrain_ckpt)

            # Save results
            out_dir = PROJECT_ROOT / "artifacts" / "metrics"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([result]).to_csv(
                out_dir / f"zinc20_pretrained_seed{args.seed}.csv",
                index=False
            )

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
