import argparse
import torch
import pandas as pd
from rdkit import Chem
from pathlib import Path

from src.config import Paths, DatasetConfig
from src.featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig
from src.finetune.train_gat_bbb_from_pretrain import GATBBB, FinetuneCfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", type=str, default="", help="Single SMILES string")
    ap.add_argument("--input_csv", type=str, default="", help="CSV file with column 'smiles'")
    ap.add_argument("--out_csv", type=str, default="artifacts/predictions/predict_out.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.smiles and not args.input_csv:
        raise ValueError("Provide either --smiles or --input_csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    P = Paths()
    D = DatasetConfig()

    # ===== load model =====
    ckpt = (
        P.models
        / "gat_finetune_bbb"
        / f"seed_{args.seed}"
        / "pretrained_partial"
        / "best.pt"
    )
    assert ckpt.exists(), f"Model not found: {ckpt}"

    # dummy dataset to get in_dim
    split_dir = P.data_splits / f"seed_{args.seed}"
    df_dummy = pd.read_csv(split_dir / "train.csv").head(1)
    df_dummy["row_id"] = [0]

    gcfg = GraphBuildConfig(
        smiles_col=D.smiles_col,
        label_col="y_cls",
        id_col="row_id"
    )
    tmp_root = P.features / "tmp_predict_dim"
    ds_dummy = BBBGraphDataset(root=str(tmp_root), df=df_dummy, cfg=gcfg)
    in_dim = ds_dummy[0].x.size(-1)

    cfg = FinetuneCfg(init="pretrained", strategy="partial", partial_k=2)
    model = GATBBB(in_dim, cfg).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    rows = []

    # ===== load input =====
    smiles_col = D.smiles_col  # usually "SMILES"

    if args.smiles:
        data_rows = [{smiles_col: args.smiles}]
    else:
        df = pd.read_csv(args.input_csv)
        assert smiles_col in df.columns, f"CSV must contain column '{smiles_col}'"
        data_rows = df[[smiles_col]].to_dict("records")

    df_pred = pd.DataFrame(data_rows)
    df_pred["row_id"] = range(len(df_pred))
    df_pred[gcfg.label_col] = 0

    import uuid

    unique_root = P.features / "predict_external" / f"run_{uuid.uuid4().hex}"
    ds = BBBGraphDataset(
        root=str(unique_root),
        df=df_pred,
        cfg=gcfg
    )

    with torch.no_grad():
        for i in range(len(ds)):
            smi = df_pred.loc[i, D.smiles_col]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"[WARN] invalid SMILES skipped: {smi}")
                continue

            data = ds[i].to(device)
            logit = model(data)
            logit_scalar = float(logit.view(-1)[0].item())
            prob = float(torch.sigmoid(torch.tensor(logit_scalar)).item())

            rows.append({
                D.smiles_col: smi,
                "bbb_prob": prob
            })

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(out_df)


if __name__ == "__main__":
    main()
