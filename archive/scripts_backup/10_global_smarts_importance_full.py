import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from types import SimpleNamespace

from rdkit import Chem

from src.config import DatasetConfig
from src.featurize.graph_pyg import BBBGraphDataset
from src.finetune.train_gat_bbb_from_pretrain import GATBBB




# ==============================
# 参数区（按你项目实际路径对齐）
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / "seed_0"
TEST_CSV = SPLIT_DIR / "test.csv"

MODEL_CKPT = PROJECT_ROOT / "artifacts" / "models" / "gat_finetune_bbb" / "seed_0" / "pretrained_partial" / "best.pt"

SMARTS_JSON = PROJECT_ROOT / "artifacts" / "explain" / "smarts_bbb_explain.json"

OUT_CSV = PROJECT_ROOT / "artifacts" / "explain" / "global_smarts_importance_full.csv"

CACHE_ROOT = PROJECT_ROOT / "cache" / "explain_tmp"


def load_smarts_list(smarts_json_path: Path):
    with open(smarts_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    smarts_list = []
    for item in data:
        name = item["name"]
        smarts = item["smarts"]
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            print(f"[WARN] Invalid SMARTS skipped: {name} -> {smarts}")
            continue
        smarts_list.append((name, smarts, patt))

    print(f"[INFO] Loaded {len(smarts_list)} SMARTS patterns")
    return smarts_list


def mask_atoms_by_smarts(mol: Chem.Mol, patt: Chem.Mol):
    matches = mol.GetSubstructMatches(patt)
    mask = set()
    for m in matches:
        for idx in m:
            mask.add(idx)
    return mask


def zero_out_atom_features(data, atom_indices):
    if len(atom_indices) == 0:
        return data
    x = data.x.clone()
    for idx in atom_indices:
        x[idx, :] = 0.0
    data.x = x
    return data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")

    # ====== 1. 读 test 集 ======
    print(f"[INFO] Loading split file: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    print(f"[INFO] Loaded {len(df)} molecules from test")

    # ====== 2. DatasetConfig（只用 smiles_col）======
    cfg = DatasetConfig()

    # ====== 3. 加载模型 ======
    print(f"[INFO] Loading model from: {MODEL_CKPT}")
    ckpt = torch.load(MODEL_CKPT, map_location="cpu", weights_only=False)

    from types import SimpleNamespace

    print(f"[INFO] Loading model from: {MODEL_CKPT}")
    ckpt = torch.load(MODEL_CKPT, map_location="cpu", weights_only=False)

    # 1) cfg: dict -> namespace（让 cfg.hidden 这种访问不炸）
    raw_cfg = ckpt["cfg"]
    if isinstance(raw_cfg, dict):
        model_cfg = SimpleNamespace(**raw_cfg)
    else:
        model_cfg = raw_cfg

    # 2) 先构建 dataset，直接从数据推 in_dim（关键！）
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    ds = BBBGraphDataset(root=CACHE_ROOT, df=df, cfg=cfg)

    in_dim = int(ds[0].x.shape[1])  # <- 这就是你真实的 node feature dim
    print(f"[INFO] Inferred in_dim from dataset: {in_dim}")

    # 3) 用正确的 in_dim 构建模型，再 load 权重
    model = GATBBB(in_dim=in_dim, cfg=model_cfg)
    state_dict = ckpt["model"]

    # ===== 自动修正 head.net -> head =====
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("head.net."):
            new_key = k.replace("head.net.", "head.")
            new_state[new_key] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=True)

    model.to(device)
    model.eval()

    # ====== 4. 加载 SMARTS 列表 ======
    smarts_list = load_smarts_list(SMARTS_JSON)

    # ====== 5. 构建 Dataset（一次性）======
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    ds = BBBGraphDataset(root=CACHE_ROOT, df=df, cfg=cfg)

    # ====== 6. 预计算 baseline logits ======
    print("[INFO] Computing baseline logits...")
    baseline_logits = []
    for i in tqdm(range(len(ds))):
        data = ds[i].to(device)
        with torch.no_grad():
            logit = model(data).view(-1).item()
        baseline_logits.append(logit)
    baseline_logits = np.array(baseline_logits)

    # ====== 7. SMARTS 贡献统计 ======
    stats = defaultdict(list)
    freq_counter = defaultdict(int)

    print("[INFO] Running SMARTS perturbation analysis...")

    for smarts_name, smarts_str, patt in smarts_list:
        print(f"[INFO] Processing SMARTS: {smarts_name}")
        deltas = []

        for i in tqdm(range(len(df)), leave=False):
            smi = df.loc[i, cfg.smiles_col]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            atom_mask = mask_atoms_by_smarts(mol, patt)
            if len(atom_mask) == 0:
                continue

            freq_counter[smarts_name] += 1

            data = ds[i]
            data_masked = zero_out_atom_features(data, atom_mask)
            data_masked = data_masked.to(device)

            with torch.no_grad():
                logit_masked = model(data_masked).view(-1).item()

            delta = logit_masked - baseline_logits[i]
            deltas.append(delta)

        if len(deltas) > 0:
            stats[smarts_name] = deltas

    # ====== 8. 汇总成 DataFrame ======
    rows = []
    for name, deltas in stats.items():
        deltas = np.array(deltas)
        rows.append({
            "smarts_name": name,
            "support": len(deltas),
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "mean_abs_delta": float(np.abs(deltas).mean()),
            "std_delta": float(deltas.std()),
            "freq_mols": freq_counter[name]
        })

    df_out = pd.DataFrame(rows).sort_values(by="mean_delta", ascending=False)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"[INFO] Saved full SMARTS importance to: {OUT_CSV}")
    print(df_out.head(10))


if __name__ == "__main__":
    main()
