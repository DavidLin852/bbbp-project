from __future__ import annotations
import argparse
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from src.config import Paths, DatasetConfig
from src.featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig
from src.finetune.train_gat_bbb_from_pretrain import GATBBB, FinetuneCfg


@torch.no_grad()
def predict_prob(model, data, device):
    model.eval()
    data = data.to(device)
    logit = model(data)
    return float(torch.sigmoid(logit)[0].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--topk", type=int, default=20,
                    help="Top-K SMARTS (by mean_abs_delta) from step-10 to use for interaction")
    ap.add_argument("--max_mols", type=int, default=0,
                    help="Optional cap on number of molecules (0 = all)")
    ap.add_argument(
        "--smarts_json",
        type=str,
        default="artifacts/explain/smarts_bbb_explain.json",
        help="SMARTS explain vocabulary (name -> smarts mapping)"
    )

    args = ap.parse_args()

    P = Paths()
    D = DatasetConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # 1. Load SMARTS list ONLY from step-10 outputs (already filtered)
    # ============================================================
    base_dir = P.metrics / "global_explain"
    pos_path = base_dir / f"global_smarts_positive_{args.split}_seed{args.seed}.csv"
    neg_path = base_dir / f"global_smarts_negative_{args.split}_seed{args.seed}.csv"

    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(
            "Step-10 outputs not found. "
            "Please run 10_global_smarts_importance.py with --min_freq first."
        )

    df_pos = pd.read_csv(pos_path)
    df_neg = pd.read_csv(neg_path)

    df_all = pd.concat([df_pos, df_neg], ignore_index=True)

    # use most important structures only
    df_all = df_all.sort_values("mean_abs_delta", ascending=False)
    df_all = df_all.head(args.topk)

    smarts_names = df_all["smarts_name"].tolist()



    # ------------------------------------------------
    # load SMARTS explain vocab (authoritative source)
    # ------------------------------------------------
    import json

    with open(args.smarts_json, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    name_to_smarts = {
        x["name"]: x["smarts"]
        for x in vocab
        if isinstance(x, dict) and "name" in x and "smarts" in x
    }
    
    # map name -> smarts string
    smarts_pairs = []
    for name in smarts_names:
        if name in name_to_smarts:
            smarts_pairs.append((name, name_to_smarts[name]))
        else:
            print(f"[WARN] SMARTS name not found in vocab: {name}")
    print(f"Using {len(smarts_names)} SMARTS from step-10 (freq-filtered)")

    # compile SMARTS patterns
    patterns = {}
    for name, s in smarts_pairs:
        patt = Chem.MolFromSmarts(s)
        if patt is not None:
            patterns[name] = patt
        else:
            print(f"[WARN] Invalid SMARTS skipped: {name} -> {s}")

    names = list(patterns.keys())
    pairs = list(itertools.combinations(names, 2))
    print(f"Evaluating {len(pairs)} SMARTS pairs")

    # ============================================================
    # 2. Load dataset
    # ============================================================
    split_dir = P.data_splits / f"seed_{args.seed}"
    df = pd.read_csv(split_dir / f"{args.split}.csv")

    if args.max_mols > 0:
        df = df.iloc[:args.max_mols].copy()

    if "row_id" not in df.columns:
        df["row_id"] = range(len(df))

    gcfg = GraphBuildConfig(
        smiles_col=D.smiles_col,
        label_col="y_cls",
        id_col="row_id"
    )
    cache_root = P.features / f"seed_{args.seed}" / f"interaction_{args.split}"
    ds = BBBGraphDataset(root=str(cache_root), df=df, cfg=gcfg)

    # ============================================================
    # 3. Load model (same as step-10)
    # ============================================================
    ckpt = (
        P.models
        / "gat_finetune_bbb"
        / f"seed_{args.seed}"
        / "pretrained_partial"
        / "best.pt"
    )
    assert ckpt.exists(), f"Model not found: {ckpt}"

    in_dim = ds[0].x.size(-1)
    cfg = FinetuneCfg(init="pretrained", strategy="partial", partial_k=2)
    model = GATBBB(in_dim, cfg).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    # ============================================================
    # 4. Pairwise interaction analysis
    # ============================================================
    rows = []

    for i in range(len(ds)):
        smi = df.loc[i, D.smiles_col]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        data = ds[i]
        base_prob = predict_prob(model, data, device)
        x0 = data.x.detach().clone()

        # precompute atom sets
        atomsets = {}
        for name, patt in patterns.items():
            matches = mol.GetSubstructMatches(patt)
            if matches:
                atomsets[name] = sorted(set(a for m in matches for a in m))

        for a, b in pairs:
            if a not in atomsets or b not in atomsets:
                continue

            A = atomsets[a]
            B = atomsets[b]
            U = sorted(set(A) | set(B))

            # ΔA
            data.x = x0.clone()
            data.x[A, :] = 0.0
            pA = predict_prob(model, data, device)
            dA = base_prob - pA

            # ΔB
            data.x = x0.clone()
            data.x[B, :] = 0.0
            pB = predict_prob(model, data, device)
            dB = base_prob - pB

            # Δ(A ∪ B)
            data.x = x0.clone()
            data.x[U, :] = 0.0
            pU = predict_prob(model, data, device)
            dU = base_prob - pU

            interaction = dU - (dA + dB)

            rows.append({
                "mol_idx": i,
                "smiles": smi,
                "A": a,
                "B": b,
                "dA": dA,
                "dB": dB,
                "dU": dU,
                "interaction": interaction
            })

        data.x = x0

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(ds)}")

    # ============================================================
    # 5. Aggregate & save
    # ============================================================
    outdir = P.metrics / "global_explain"
    outdir.mkdir(parents=True, exist_ok=True)

    dfI = pd.DataFrame(rows)
    raw_path = outdir / f"smarts_interactions_raw_{args.split}_seed{args.seed}.csv"
    dfI.to_csv(raw_path, index=False)

    if not dfI.empty:
        summary = (
            dfI.groupby(["A", "B"])["interaction"]
            .agg(
                support="count",
                mean_interaction="mean",
                median_interaction="median",
                mean_abs_interaction=lambda x: float(np.mean(np.abs(x))),
                std_interaction="std"
            )
            .reset_index()
            .sort_values("mean_abs_interaction", ascending=False)
        )

        sum_path = outdir / f"global_smarts_interactions_{args.split}_seed{args.seed}.csv"
        summary.to_csv(sum_path, index=False)

        print("Saved:")
        print(" -", raw_path)
        print(" -", sum_path)
    else:
        print("No valid SMARTS interactions found.")

if __name__ == "__main__":
    main()
