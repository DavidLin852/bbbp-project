#!/usr/bin/env python
"""
Extract and cache molecular embeddings from pretrained models.

Saves embeddings as .npy files for use in Groups B and C (LightGBM training).

Usage:
    # Extract GNN embeddings for one pretrain experiment
    python scripts/finetune/extract_embeddings.py \
        --pretrain_id P_E10_GIN_1M --model_type gin \
        --seed 0 --task classification

    # Extract Transformer embeddings
    python scripts/finetune/extract_embeddings.py \
        --pretrain_id T_E10_TRANS_1M --model_type transformer \
        --seed 0 --task classification

    # Skip existing (resume)
    python scripts/finetune/extract_embeddings.py \
        --pretrain_id P_E10_GIN_1M --model_type gin \
        --seed 0 --task classification --skip_existing
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Add project root
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.finetune.finetune_config import get_pretrain_config
from src.gnn.models import GIN, GAT
from src.gnn.dataset import B3DBGNNDataset


# ============================================================
# GNN Embedding Extraction
# ============================================================

def extract_gnn_embeddings(
    pretrain_id: str,
    model_type: str,
    seed: int,
    task: str,
    output_dir: str | Path,
    device: str = "auto",
    skip_existing: bool = True,
) -> bool:
    """
    Extract graph-level embeddings from a pretrained GNN backbone.

    Saves X_{split}.npy and y_{split}.npy for each split.

    Returns True if extraction succeeded or was skipped.
    """
    output_dir = Path(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)

    # --- Resolve architecture ---
    cfg = get_pretrain_config(pretrain_id)
    if model_type not in ("gin", "gat"):
        print(f"[ERROR] {pretrain_id}: model_type must be gin/gat, got {model_type}")
        return False

    # --- Check backbone exists ---
    backbone_path = output_dir / ".." / ".." / ".." / cfg.get("id", pretrain_id) / f"{model_type}_pretrained_backbone.pt"
    # More reliably: build path from project root
    backbone_path = project_root / "artifacts" / "models" / "pretrain" / "exp_matrix" / pretrain_id / f"{model_type}_pretrained_backbone.pt"

    if not backbone_path.exists():
        print(f"[SKIP] Backbone not found: {backbone_path}")
        return False

    print(f"[GNN] {pretrain_id} ({model_type}), seed={seed}, task={task}")
    print(f"  Backbone: {backbone_path}")

    # --- Create output subdir ---
    out_subdir = output_dir / pretrain_id / f"seed_{seed}" / task
    splits_dir = project_root / "data" / "splits" / f"seed_{seed}" / f"{task}_scaffold"

    # --- Check skip ---
    if skip_existing:
        check_file = out_subdir / "X_train.npy"
        if check_file.exists():
            print(f"  [SKIP] Embeddings already exist: {check_file}")
            return True

    out_subdir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    dataset = B3DBGNNDataset(split_dir=splits_dir, task=task)

    # --- Create backbone ---
    if model_type == "gin":
        backbone = GIN(
            in_dim=22,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=0.0,  # No dropout during inference
        )
    elif model_type == "gat":
        backbone = GAT(
            in_dim=22,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            heads=cfg["heads"],
            dropout=0.0,
        )

    # --- Load pretrained weights ---
    state_dict = torch.load(backbone_path, map_location="cpu", weights_only=False)
    backbone.load_state_dict(state_dict)
    backbone = backbone.to(device)
    backbone.eval()

    # --- Extract embeddings ---
    loader = DataLoader(
        dataset.get_split("train"),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    def _extract_split(split_name: str, loader) -> tuple[np.ndarray, np.ndarray]:
        """Extract embeddings and labels for one split."""
        embeddings_list = []
        labels_list = []

        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                emb = backbone(batch)  # (batch_size, hidden_dim)
            embeddings_list.append(emb.cpu().numpy())
            labels_list.append(batch.y.squeeze().cpu().numpy())

        X = np.concatenate(embeddings_list, axis=0)
        y = np.concatenate(labels_list, axis=0)

        print(f"    {split_name}: X.shape={X.shape}, y.shape={y.shape}")
        return X, y

    for split_name, loader in [
        ("train", DataLoader(dataset.get_split("train"), batch_size=128, shuffle=False, num_workers=0)),
        ("val", DataLoader(dataset.get_split("val"), batch_size=128, shuffle=False, num_workers=0)),
        ("test", DataLoader(dataset.get_split("test"), batch_size=128, shuffle=False, num_workers=0)),
    ]:
        X, y = _extract_split(split_name, loader)
        np.save(out_subdir / f"X_{split_name}.npy", X.astype(np.float32))
        np.save(out_subdir / f"y_{split_name}.npy", y.astype(np.float32))

    # Save metadata
    meta = {
        "pretrain_id": pretrain_id,
        "model_type": model_type,
        "hidden_dim": cfg["hidden_dim"],
        "seed": seed,
        "task": task,
    }
    with open(out_subdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [DONE] Saved to {out_subdir}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return True


# ============================================================
# Transformer Embedding Extraction
# ============================================================

def extract_transformer_embeddings(
    pretrain_id: str,
    seed: int,
    task: str,
    output_dir: str | Path,
    device: str = "auto",
    skip_existing: bool = True,
) -> bool:
    """
    Extract sequence embeddings from a pretrained Transformer encoder.

    Uses masked mean pooling over the encoded sequence.
    Saves X_{split}.npy and y_{split}.npy for each split.
    """
    output_dir = Path(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)

    cfg = get_pretrain_config(pretrain_id)
    if cfg["model_type"] != "transformer":
        print(f"[ERROR] {pretrain_id}: model_type is not transformer")
        return False

    # --- Check files exist ---
    encoder_path = project_root / "artifacts" / "models" / "pretrain" / "exp_matrix" / pretrain_id / "transformer_pretrained_encoder.pt"
    tokenizer_path = project_root / "artifacts" / "models" / "pretrain" / "exp_matrix" / pretrain_id / "tokenizer.pkl"

    if not encoder_path.exists():
        print(f"[SKIP] Encoder not found: {encoder_path}")
        return False

    print(f"[TRANS] {pretrain_id}, seed={seed}, task={task}")
    print(f"  Encoder: {encoder_path}")

    # --- Infer architecture from checkpoint state dict ---
    from src.transformer.smiles_transformer import SMILESTransformerEncoder
    state_dict = torch.load(encoder_path, map_location="cpu", weights_only=False)

    vocab_size = state_dict["token_embedding.weight"].shape[0]
    d_model = state_dict["token_embedding.weight"].shape[1]
    print(f"  Inferred vocab_size={vocab_size}, d_model={d_model}")

    encoder = SMILESTransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=cfg["heads"],
        n_layers=cfg["num_layers"],
    )
    encoder.load_state_dict(state_dict)
    encoder = encoder.to(device)
    encoder.eval()

    # --- Load tokenizer (or rebuild from B3DB data) ---
    from src.transformer.smiles_tokenizer import SMILESTokenizer, create_tokenizer_from_data
    if tokenizer_path.exists():
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        print(f"  Tokenizer vocab: {len(tokenizer)}")
    else:
        # Rebuild tokenizer from B3DB SMILES with matching vocab_size
        # This won't be identical to the original ZINC22-trained tokenizer
        # but char-level token coverage overlaps well enough for embedding extraction
        print(f"  [WARN] Tokenizer not found, rebuilding from B3DB data (vocab={vocab_size})...")
        import pandas as pd
        smiles_list = []
        for split_name in ["train", "val", "test"]:
            split_path = project_root / "data" / "splits" / f"seed_{seed}" / f"{task}_scaffold" / f"{split_name}.csv"
            if split_path.exists():
                df = pd.read_csv(split_path)
                smiles_list.extend(df["SMILES_canon"].astype(str).tolist())
        tokenizer = create_tokenizer_from_data(smiles_list, vocab_size=vocab_size, min_freq=1)
        print(f"  Rebuilt tokenizer vocab: {len(tokenizer)}")

    # --- Load B3DB splits ---
    splits_dir = project_root / "data" / "splits" / f"seed_{seed}" / f"{task}_scaffold"

    for split_name in ["train", "val", "test"]:
        csv_path = splits_dir / f"{split_name}.csv"
        if not csv_path.exists():
            print(f"  [WARN] Split not found: {csv_path}")
            continue

        import pandas as pd
        df = pd.read_csv(csv_path)

        # --- Tokenize all SMILES ---
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        max_len = cfg.get("max_len", 128)
        pad_id = tokenizer.pad_token_id

        for _, row in df.iterrows():
            smiles = str(row["SMILES_canon"])
            label = float(row["y_cls"] if task == "classification" else row["logBB"])

            tokens = tokenizer.encode(smiles, add_special_tokens=True)
            # Truncate if too long
            if len(tokens) > max_len:
                tokens = tokens[:max_len]

            length = len(tokens)
            # Pad
            if length < max_len:
                tokens = tokens + [pad_id] * (max_len - length)
            else:
                tokens = tokens[:max_len]

            mask = [1 if t != pad_id else 0 for t in tokens]

            input_ids_list.append(tokens)
            attention_mask_list.append(mask)
            labels_list.append(label)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
        labels = np.array(labels_list, dtype=np.float32)

        # --- Extract embeddings in batches ---
        batch_size = 128
        embeddings_list = []

        with torch.no_grad():
            for i in range(0, len(input_ids), batch_size):
                ids_batch = input_ids[i:i+batch_size].to(device)
                mask_batch = attention_mask[i:i+batch_size].to(device)

                encoded = encoder(ids_batch, mask_batch)  # (batch, seq_len, d_model)

                # Masked mean pooling
                mask_expanded = mask_batch.unsqueeze(-1).float()  # (batch, seq_len, 1)
                emb = (encoded * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
                embeddings_list.append(emb.cpu().numpy())

        X = np.concatenate(embeddings_list, axis=0).astype(np.float32)
        y = labels

        print(f"    {split_name}: X.shape={X.shape}, y.shape={y.shape}")
        np.save(out_subdir / f"X_{split_name}.npy", X)
        np.save(out_subdir / f"y_{split_name}.npy", y.astype(np.float32))

    # Save metadata
    meta = {
        "pretrain_id": pretrain_id,
        "model_type": "transformer",
        "hidden_dim": cfg["hidden_dim"],
        "seed": seed,
        "task": task,
        "num_layers": cfg["num_layers"],
        "n_heads": cfg["heads"],
    }
    with open(out_subdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [DONE] Saved to {out_subdir}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return True


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from pretrained models")
    parser.add_argument("--pretrain_id", type=str, required=True,
                        help="Pretrain experiment ID (e.g. P_E10_GIN_1M)")
    parser.add_argument("--model_type", type=str, default=None,
                        help="Model type: gin, gat, transformer. Auto-detected if omitted.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "regression"],
                        help="Downstream task (default: classification)")
    parser.add_argument("--output_dir", type=str,
                        default="artifacts/embeddings",
                        help="Output directory for embeddings")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda, cpu, auto (default: auto)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip if embeddings already exist (default: True)")
    parser.add_argument("--no_skip", action="store_true",
                        help="Force re-extraction even if files exist")

    args = parser.parse_args()

    # Auto-detect model type
    model_type = args.model_type
    if model_type is None:
        cfg = get_pretrain_config(args.pretrain_id)
        model_type = cfg["model_type"]

    skip = not args.no_skip

    if model_type in ("gin", "gat"):
        success = extract_gnn_embeddings(
            pretrain_id=args.pretrain_id,
            model_type=model_type,
            seed=args.seed,
            task=args.task,
            output_dir=args.output_dir,
            device=args.device,
            skip_existing=skip,
        )
    elif model_type == "transformer":
        success = extract_transformer_embeddings(
            pretrain_id=args.pretrain_id,
            seed=args.seed,
            task=args.task,
            output_dir=args.output_dir,
            device=args.device,
            skip_existing=skip,
        )
    else:
        print(f"[ERROR] Unknown model_type: {model_type}")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
