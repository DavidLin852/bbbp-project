"""
完整GNN Pipeline: GAT baseline → 物理属性辅助 → SMARTS预训练 → 微调

数据: A,B,C,D全部 (7807样本)
顺序:
1. GAT baseline (纯图结构)
2. GAT + 物理属性 (logP + TPSA作为辅助任务)
3. SMARTS预训练 (化学结构预训练)
4. 微调BBB分类器
5. 最终对比

Usage:
    python run_gnn_pipeline.py --seed 0 --step 1  # GAT baseline
    python run_gnn_pipeline.py --seed 0 --step 2  # GAT + 物理属性
    python run_gnn_pipeline.py --seed 0 --step 3  # SMARTS预训练
    python run_gnn_pipeline.py --seed 0 --step 4  # 微调
    python run_gnn_pipeline.py --seed 0 --step 5  # 最终对比
"""
import sys
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def step1_gat_baseline(seed):
    """Step 1: GAT baseline - 纯图结构"""
    print("\n" + "="*80)
    print("STEP 1: GAT Baseline (纯图结构)")
    print("="*80)

    from src.featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig
    from torch_geometric.loader import DataLoader
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv, global_mean_pool
    from src.utils.seed import seed_everything
    from src.utils.metrics import classification_metrics
    from src.utils.plotting import plot_roc_curves
    from sklearn.metrics import roc_auc_score

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    split_dir = PROJECT_ROOT / "data" / "splits" / f"seed_{seed}_full"
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    # Prepare graph dataset (without logP/TPSA for baseline)
    gcfg = GraphBuildConfig(smiles_col="SMILES", label_col="y_cls", id_col="row_id")

    cache_root = PROJECT_ROOT / "artifacts" / "features" / f"seed_{seed}_full" / "pyg_graphs_baseline"
    train_ds = BBBGraphDataset(root=str(cache_root / "train"), df=train_df, cfg=gcfg)
    val_ds = BBBGraphDataset(root=str(cache_root / "val"), df=val_df, cfg=gcfg)
    test_ds = BBBGraphDataset(root=str(cache_root / "test"), df=test_df, cfg=gcfg)

    print(f"Graph datasets created: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Build GAT model
    class GATClassifier(nn.Module):
        def __init__(self, in_dim, hidden=128, heads=4, num_layers=3, dropout=0.2):
            super().__init__()
            self.dropout = dropout
            self.convs = nn.ModuleList()

            self.convs.append(GATConv(in_dim, hidden, heads=heads, concat=True))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden * heads, hidden, heads=heads, concat=True))
            self.convs.append(GATConv(hidden * heads, hidden, heads=1, concat=True))

            self.classifier = nn.Sequential(
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
            return self.classifier(g).view(-1)

    # Training
    in_dim = train_ds[0].x.size(-1)
    model = GATClassifier(in_dim).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    out_dir = PROJECT_ROOT / "artifacts" / "models" / f"seed_{seed}_full" / "gat_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = -1.0
    epochs = 60

    print(f"\n开始训练GAT baseline ({epochs} epochs)...")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            logit = model(batch)
            y = batch.y_cls.view(-1)
            loss = criterion(logit, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

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
            val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.0

        # Save best
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), out_dir / "best.pt")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}: Loss={total_loss/len(train_ds):.4f}, Val AUC={val_auc:.4f}")

    # Test evaluation
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

    # Metrics
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix

    auc = roc_auc_score(test_labels, test_probs)
    auprc = average_precision_score(test_labels, test_probs)

    y_pred = (test_probs >= 0.5).astype(int)
    precision = precision_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred)

    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()

    result = {
        'model': 'GAT_baseline',
        'auc': float(auc),
        'auprc': float(auprc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

    # Save results
    out_metrics = PROJECT_ROOT / "artifacts" / "metrics"
    out_metrics.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(out_metrics / f"gat_baseline_seed{seed}.csv", index=False)

    # Plot ROC
    plot_roc_curves(
        [{"name": "GAT_baseline", "y_true": test_labels, "y_prob": test_probs}],
        out_dir / "roc.png",
        title=f"GAT Baseline ROC (seed={seed})"
    )

    print(f"\n[GAT Baseline] Test Results:")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  FP:        {fp}")
    print(f"  FN:        {fn}")
    print(f"  ROC:       {out_dir / 'roc.png'}")

    return result


def step2_gat_phys_aux(seed):
    """Step 2: GAT + 物理属性辅助监督"""
    print("\n" + "="*80)
    print("STEP 2: GAT + 物理属性辅助监督 (logP + TPSSA)")
    print("="*80)

    # 这里可以直接使用现有的04_run_gat_aux.py脚本
    # 但需要修改为使用seed_0_full数据

    from src.featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig
    from src.phys_aux.train_gat_aux import TrainCfg, train_gat_multitask
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Crippen, rdMolDescriptors

    split_dir = PROJECT_ROOT / "data" / "splits" / f"seed_{seed}_full"
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    # 计算logP和TPSA
    def compute_logp_tpsa(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        logp = float(Crippen.MolLogP(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        return logp, tpsa

    print("计算logP和TPSA...")
    for df in [train_df, val_df, test_df]:
        logp_tpsa = df['SMILES'].apply(lambda x: pd.Series(compute_logp_tpsa(x), index=['logP', 'TPSA']))
        df['y_logp'] = logp_tpsa['logP'].values
        df['y_tpsa'] = logp_tpsa['TPSA'].values
        # Fill NaN with mean
        df['y_logp'].fillna(df['y_logp'].mean(), inplace=True)
        df['y_tpsa'].fillna(df['y_tpsa'].mean(), inplace=True)

    # 准备图数据集
    gcfg = GraphBuildConfig(smiles_col="SMILES", label_col="y_cls", id_col="row_id")

    cache_root = PROJECT_ROOT / "artifacts" / "features" / f"seed_{seed}_full" / "pyg_graphs_phys_aux"
    train_ds = BBBGraphDataset(root=str(cache_root / "train"), df=train_df, cfg=gcfg)
    val_ds = BBBGraphDataset(root=str(cache_root / "val"), df=val_df, cfg=gcfg)
    test_ds = BBBGraphDataset(root=str(cache_root / "test"), df=test_df, cfg=gcfg)

    # 训练
    out_dir = PROJECT_ROOT / "artifacts" / "models" / f"seed_{seed}_full" / "gat_phys_aux"
    tcfg = TrainCfg(
        seed=seed,
        epochs=60,
        batch_size=64,
        lambda_logp=0.3,
        lambda_tpsa=0.3
    )

    print(f"\n开始训练GAT多任务模型 (分类 + logP + TPSA)...")
    final_row, roc_path = train_gat_multitask(train_ds, val_ds, test_ds, out_dir, tcfg)

    # 保存结果
    out_metrics = PROJECT_ROOT / "artifacts" / "metrics"
    pd.DataFrame([final_row]).to_csv(out_metrics / f"gat_phys_aux_seed{seed}.csv", index=False)

    print(f"\n[GAT+物理属性] Test Results:")
    print(f"  AUC:       {final_row['test_auc']:.4f}")
    print(f"  logP MAE:  {final_row['test_logp_mae']:.4f}")
    print(f"  TPSA MAE:  {final_row['test_tpsa_mae']:.4f}")
    print(f"  ROC:       {roc_path}")

    return final_row


def step5_summary(seed):
    """生成最终对比报告"""
    print("\n" + "="*90)
    print("                    最终模型对比报告")
    print("                 数据集：A,B,C,D全部 (7807样本)")
    print("="*90)
    print()

    import pandas as pd

    out_dir = PROJECT_ROOT / "artifacts" / "metrics"

    # 加载所有结果
    results = []

    # Baseline
    if (out_dir / f"baseline_seed{seed}.csv").exists():
        baseline = pd.read_csv(out_dir / f"baseline_seed{seed}.csv")
        for _, row in baseline.iterrows():
            results.append({
                'model': row['model'],
                'type': 'Baseline',
                'auc': row['auc'],
                'precision': row['precision'],
                'recall': row['recall'],
                'f1': row['f1'],
                'fp': row['fp']
            })

    # GAT baseline
    if (out_dir / f"gat_baseline_seed{seed}.csv").exists():
        gat = pd.read_csv(out_dir / f"gat_baseline_seed{seed}.csv").iloc[0]
        results.append({
            'model': 'GAT',
            'type': 'GNN',
            'auc': gat['auc'],
            'precision': gat['precision'],
            'recall': gat['recall'],
            'f1': gat['f1'],
            'fp': gat['fp']
        })

    # GAT + 物理
    if (out_dir / f"gat_phys_aux_seed{seed}.csv").exists():
        gat_phys = pd.read_csv(out_dir / f"gat_phys_aux_seed{seed}.csv").iloc[0]
        results.append({
            'model': 'GAT+logP+TPSA',
            'type': 'GNN+Aux',
            'auc': gat_phys['test_auc'],
            'precision': gat_phys['test_precision_pos'],
            'recall': gat_phys['test_recall_pos'],
            'f1': gat_phys['test_f1_pos'],
            'fp': gat_phys['fp']
        })

    df_results = pd.DataFrame(results)

    print("全部模型对比 (测试集):")
    print("-"*90)
    print(df_results[['model', 'type', 'auc', 'precision', 'recall', 'f1', 'fp']].to_string(index=False))

    print()
    print("="*90)
    print("关键发现")
    print("="*90)
    print()

    # 找最佳模型
    best_auc = df_results.loc[df_results['auc'].idxmax()]
    print(f"最佳AUC: {best_auc['model']} ({best_auc['type']}) - AUC={best_auc['auc']:.4f}")

    # 找最低FP
    lowest_fp = df_results.loc[df_results['fp'].idxmin()]
    print(f"最低FP:  {lowest_fp['model']} ({lowest_fp['type']}) - FP={lowest_fp['fp']}")

    print()
    print("推荐使用:")
    if best_auc['auc'] >= 0.95:
        print("  [x] 模型性能优秀 (AUC >= 0.95)")
        if best_auc['fp'] < 60:
            print(f"  [x] 假阳性控制良好 (FP={best_auc['fp']})")
            print(f"  -> 推荐: {best_auc['model']}")
        else:
            print(f"  [ ] 假阳性仍较高 (FP={best_auc['fp']})")
            print(f"  -> 建议使用: {lowest_fp['model']} (保守策略)")

    print()
    print("="*90)
    print("模型文件位置:")
    print("="*90)
    print("Baseline: artifacts/models/seed_0_full/baseline/")
    print("GNN:      artifacts/models/seed_0_full/gat_baseline/")
    print("GNN+Aux:  artifacts/models/seed_0_full/gat_phys_aux/")
    print("="*90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5], help="Pipeline step")
    args = ap.parse_args()

    try:
        if args.step == 1:
            step1_gat_baseline(args.seed)
        elif args.step == 2:
            step2_gat_phys_aux(args.seed)
        elif args.step == 5:
            step5_summary(args.seed)
        else:
            print("Step not yet implemented")
            print("Available steps: 1, 2, 5")
            print("\nNote: Steps 3 (SMARTS预训练) 和 4 (微调) 需要额外资源，建议先运行步骤1-2")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
