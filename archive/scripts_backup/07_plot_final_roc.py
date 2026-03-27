from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

from src.config import Paths, DatasetConfig
from src.featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig
from src.finetune.train_gat_bbb_from_pretrain import GATBBB, FinetuneCfg
from src.utils.seed import seed_everything

# ============ 全局绘图风格 ============
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14

@torch.no_grad()
def eval_gat(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        batch = batch.to(device)
        logit = model(batch)
        prob = torch.sigmoid(logit).cpu().numpy()
        y = batch.y_cls.view(-1).cpu().numpy()
        y_true.append(y)
        y_prob.append(prob)
    return np.concatenate(y_true), np.concatenate(y_prob)

def load_gat_model(ckpt_path: Path, cfg: FinetuneCfg, in_dim: int, device):
    model = GATBBB(in_dim, cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model

def featurize_morgan(smiles_list, radius=2, n_bits=2048):
    X = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        ConvertToNumpyArray(fp, arr)
        X[i] = arr
    return X

def main():
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    P = Paths()
    D = DatasetConfig()

    # ---------- Load test dataset ----------
    split_dir = P.data_splits / "seed_0"
    test_df = pd.read_csv(split_dir / "test.csv")

    gcfg = GraphBuildConfig(
        smiles_col=D.smiles_col,
        label_col="y_cls",
        id_col="row_id"
    )

    cache_root = P.features / "seed_0" / "pyg_graphs_bbb"
    test_ds = BBBGraphDataset(root=str(cache_root / "test"), df=test_df, cfg=gcfg)

    from torch_geometric.loader import DataLoader
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    in_dim = test_ds[0].x.size(-1)

    curves = []

    # ============================================================
    # 1) GAT baseline (random init)
    # ============================================================
    cfg_random = FinetuneCfg(init="random", strategy="full")
    ckpt_random = (
        P.models / "gat_finetune_bbb" / "seed_0" / "random_full" / "best.pt"
    )
    model_random = load_gat_model(ckpt_random, cfg_random, in_dim, device)
    y_true, y_prob = eval_gat(model_random, test_loader, device)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    curves.append(("GAT (random)", fpr, tpr, auc(fpr, tpr)))

    # ============================================================
    # 2) GAT + SMARTS pretrained (partial finetune)
    # ============================================================
    cfg_pre = FinetuneCfg(init="pretrained", strategy="partial", partial_k=2)
    ckpt_pre = (
        P.models / "gat_finetune_bbb" / "seed_0" / "pretrained_partial" / "best.pt"
    )
    model_pre = load_gat_model(ckpt_pre, cfg_pre, in_dim, device)
    y_true, y_prob = eval_gat(model_pre, test_loader, device)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    curves.append(("GAT + SMARTS (partial)", fpr, tpr, auc(fpr, tpr)))

    # ============================================================
    # 3) XGB baseline（重新加载模型推断）
    # ============================================================
    import joblib

    xgb_model_path = P.models / "xgb_morgan_seed0.joblib"
    xgb = joblib.load(xgb_model_path)

    X_test = featurize_morgan(test_df[D.smiles_col].tolist())
    y_true = test_df["y_cls"].values
    y_prob = xgb.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    curves.append(("XGB (Morgan)", fpr, tpr, auc(fpr, tpr)))

    # ================== Plot ==================
    fig = plt.figure(figsize=(6.4, 5.4))
    ax = fig.add_subplot(111)

    for name, fpr, tpr, roc_auc in curves:
        ax.plot(fpr, tpr, linewidth=2.2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="gray")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("BBB Permeability Prediction (Test Set)")
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    out_path = P.metrics / "final_roc_bbb.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print("Saved final ROC to:", out_path)




if __name__ == "__main__":
    main()
