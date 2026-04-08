#!/usr/bin/env python
"""
Build a unified CSV with ALL baseline results.

Reads classification, regression, GNN, and Transformer benchmarks
and outputs a single sorted CSV.
"""

import pandas as pd
from pathlib import Path

reports_dir = Path("artifacts/reports")

rows = []

# 1. Classical Classification
cls_df = pd.read_csv(reports_dir / "cls_benchmark_scaffold.csv")
for _, r in cls_df.iterrows():
    rows.append({
        "category": "classical",
        "task": "classification",
        "model": r["model_name"],
        "feature": r["feature"],
        "n_seeds": int(r["test_auc_count"]),
        "test_auc": round(r["test_auc_mean"], 4),
        "test_auc_std": round(r["test_auc_std"], 4),
        "test_f1": round(r["test_f1_mean"], 4),
        "test_f1_std": round(r["test_f1_std"], 4),
        "test_r2": "",
        "test_r2_std": "",
        "test_rmse": "",
        "test_rmse_std": "",
    })

# 2. Classical Regression
reg_df = pd.read_csv(reports_dir / "reg_benchmark_scaffold.csv")
for _, r in reg_df.iterrows():
    rows.append({
        "category": "classical",
        "task": "regression",
        "model": r["model_name"],
        "feature": r["feature"],
        "n_seeds": int(r["test_r2_count"]),
        "test_auc": "",
        "test_auc_std": "",
        "test_f1": "",
        "test_f1_std": "",
        "test_r2": round(r["test_r2_mean"], 4),
        "test_r2_std": round(r["test_r2_std"], 4),
        "test_rmse": round(r["test_rmse_mean"], 4),
        "test_rmse_std": round(r["test_rmse_std"], 4),
    })

# 3. GNN (from the latest benchmark file)
gnn_files = sorted(reports_dir.glob("gnn/gnn_benchmark_scaffold_*.csv"))
if gnn_files:
    gnn_df = pd.read_csv(gnn_files[-1])
    for _, r in gnn_df.iterrows():
        task = r["task"]
        if task == "classification":
            rows.append({
                "category": "gnn",
                "task": "classification",
                "model": r["model"],
                "feature": "graph (22d nodes, 7d edges)",
                "n_seeds": int(r["n_seeds"]),
                "test_auc": round(r["test_auc_mean"], 4),
                "test_auc_std": round(r["test_auc_std"], 4),
                "test_f1": round(r["test_f1_mean"], 4),
                "test_f1_std": round(r["test_f1_std"], 4),
                "test_r2": "",
                "test_r2_std": "",
                "test_rmse": "",
                "test_rmse_std": "",
            })
        elif task == "regression":
            rows.append({
                "category": "gnn",
                "task": "regression",
                "model": r["model"],
                "feature": "graph (22d nodes, 7d edges)",
                "n_seeds": int(r["n_seeds"]),
                "test_auc": "",
                "test_auc_std": "",
                "test_f1": "",
                "test_f1_std": "",
                "test_r2": round(r["test_r2_mean"], 4),
                "test_r2_std": round(r["test_r2_std"], 4),
                "test_rmse": round(r["test_rmse_mean"], 4),
                "test_rmse_std": round(r["test_rmse_std"], 4),
            })

# 4. Transformer
transformer_files = sorted(reports_dir.glob("transformer/transformer_benchmark_scaffold_*.csv"))
if transformer_files:
    # Aggregate across all transformer files (multiple seeds)
    all_tf_rows = []
    for tf_file in transformer_files:
        tf_df = pd.read_csv(tf_file)
        all_tf_rows.append(tf_df)
    tf_all = pd.concat(all_tf_rows, ignore_index=True)

    # Classification: aggregate by seed
    tf_cls = tf_all[tf_all["task"] == "classification"]
    if len(tf_cls) > 0:
        rows.append({
            "category": "transformer",
            "task": "classification",
            "model": "SMILES Transformer",
            "feature": "SMILES tokenized (128 len)",
            "n_seeds": len(tf_cls),
            "test_auc": round(tf_cls["auc"].mean(), 4),
            "test_auc_std": round(tf_cls["auc"].std(), 4),
            "test_f1": round(tf_cls["f1"].mean(), 4),
            "test_f1_std": round(tf_cls["f1"].std(), 4),
            "test_r2": "",
            "test_r2_std": "",
            "test_rmse": "",
            "test_rmse_std": "",
        })

    # Regression: aggregate by seed
    tf_reg = tf_all[tf_all["task"] == "regression"]
    if len(tf_reg) > 0:
        rows.append({
            "category": "transformer",
            "task": "regression",
            "model": "SMILES Transformer",
            "feature": "SMILES tokenized (128 len)",
            "n_seeds": len(tf_reg),
            "test_auc": "",
            "test_auc_std": "",
            "test_f1": "",
            "test_f1_std": "",
            "test_r2": round(tf_reg["r2"].mean(), 4),
            "test_r2_std": round(tf_reg["r2"].std(), 4),
            "test_rmse": round(tf_reg["rmse"].mean(), 4),
            "test_rmse_std": round(tf_reg["rmse"].std(), 4),
        })

# Build DataFrame
df = pd.DataFrame(rows)

# Sort: classification by auc desc, regression by r2 desc
cls_mask = df["task"] == "classification"
reg_mask = df["task"] == "regression"

# Convert to numeric for sorting
df_cls = df[cls_mask].copy()
df_cls["_sort"] = pd.to_numeric(df_cls["test_auc"], errors="coerce")
df_cls = df_cls.sort_values("_sort", ascending=False).drop(columns=["_sort"])

df_reg = df[reg_mask].copy()
df_reg["_sort"] = pd.to_numeric(df_reg["test_r2"], errors="coerce")
df_reg = df_reg.sort_values("_sort", ascending=False).drop(columns=["_sort"])

# Add rank
df_cls.insert(0, "rank", range(1, len(df_cls) + 1))
df_reg.insert(0, "rank", range(1, len(df_reg) + 1))

df_final = pd.concat([df_cls, df_reg], ignore_index=True)

# Save
out_path = reports_dir / "all_baselines_summary.csv"
df_final.to_csv(out_path, index=False)

print(f"Saved {len(df_final)} results to {out_path}")
print(f"  Classification: {len(df_cls)} models")
print(f"  Regression: {len(df_reg)} models")
print()
print(df_final.to_string(index=False))
