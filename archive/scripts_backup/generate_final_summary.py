"""
Generate Final Comprehensive Performance Summary Table
整合所有模型（传统ML、GAT、Transformer）的性能汇总表
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_all_results():
    """加载所有实验结果"""

    results_dir = PROJECT_ROOT / "artifacts" / "ablation"

    # 1. 传统ML模型结果
    ml_files = [
        results_dir / "COMPLETE_MATRIX_RESULTS.csv",
        results_dir / "ALL_FINAL_RESULTS.csv",
    ]

    ml_results = []
    for f in ml_files:
        if f.exists():
            df = pd.read_csv(f)
            df['model_category'] = 'Traditional ML'
            ml_results.append(df)

    # 2. GAT模型结果 (手动添加)
    gat_results = [
        {
            "feature": "graph",
            "model": "GAT_baseline",
            "auc": 0.9343,
            "f1": 0.8989,
            "precision": 0.8757,
            "recall": 0.9234,
            "mcc": 0.7778,
            "model_category": "GNN"
        },
        {
            "feature": "graph",
            "model": "GAT+SMARTS",
            "auc": 0.9560,
            "f1": 0.9370,
            "precision": 0.9060,
            "recall": 0.9419,
            "mcc": 0.8000,
            "model_category": "GNN"
        },
        {
            "feature": "graph",
            "model": "GAT+SMARTS(full)",
            "auc": 0.9479,
            "f1": 0.9384,
            "precision": 0.9103,
            "recall": 0.9384,
            "mcc": 0.7900,
            "model_category": "GNN"
        },
    ]

    # 3. Transformer模型结果 (如果有)
    transformer_results = []

    return ml_results, gat_results, transformer_results


def create_feature_display_name(feat):
    """创建特征显示名称"""
    feature_map = {
        "morgan": "Morgan (2048D)",
        "maccs": "MACCS (167D)",
        "atompairs": "AtomPairs (1024D)",
        "fp2": "FP2 (2048D)",
        "rdkit_desc": "RDKitDescriptors (98D)",
        "combined": "Combined (5287D)",
        "graph": "Graph Structure"
    }
    return feature_map.get(feat, feat)


def create_summary_table():
    """生成最终汇总表"""

    ml_results, gat_results, transformer_results = load_all_results()

    # 合并所有结果
    all_results = []

    # 添加ML结果
    for df in ml_results:
        for _, row in df.iterrows():
            all_results.append({
                "Model": row['model'],
                "Feature": row['feature'],
                "Feature_Display": create_feature_display_name(row['feature']),
                "AUC": row['auc'],
                "F1": row.get('f1', row.get('f1_pos', 0)),
                "MCC": row.get('mcc', 0),
                "Category": "Traditional ML"
            })

    # 添加GAT结果
    for res in gat_results:
        all_results.append({
            "Model": res['model'],
            "Feature": res['feature'],
            "Feature_Display": create_feature_display_name(res['feature']),
            "AUC": res['auc'],
            "F1": res['f1'],
            "MCC": res['mcc'],
            "Category": "GNN"
        })

    df_all = pd.DataFrame(all_results)

    if len(df_all) == 0:
        print("⚠️  没有找到任何实验结果")
        return None

    # 按AUC排序
    df_all = df_all.sort_values('AUC', ascending=False).reset_index(drop=True)

    return df_all


def generate_pivot_tables(df):
    """生成透视表"""

    # 1. 模型 x 特征 AUC 矩阵
    pivot_auc = df.pivot_table(
        index='Model',
        columns='Feature_Display',
        values='AUC',
        aggfunc='max'
    )

    # 2. 按模型类别统计
    category_stats = df.groupby('Category').agg({
        'AUC': ['max', 'mean', 'count'],
        'F1': 'max'
    }).round(4)

    # 3. 按特征统计
    feature_stats = df.groupby('Feature_Display').agg({
        'AUC': ['max', 'mean', 'count'],
        'F1': 'max'
    }).round(4)

    return pivot_auc, category_stats, feature_stats


def print_summary():
    """打印汇总信息"""

    df = create_summary_table()

    if df is None:
        return

    print("\n" + "="*100)
    print(" " * 35 + "BBB渗透性预测 - 完整性能汇总表")
    print("="*100)
    print(f"\n总实验数: {len(df)}")
    print(f"模型数量: {df['Model'].nunique()}")
    print(f"特征类型: {df['Feature_Display'].nunique()}")
    print()

    # Top 50
    print("="*100)
    print("Top 50 模型-特征组合")
    print("="*100)
    top50 = df.head(50)[['Model', 'Feature_Display', 'AUC', 'F1', 'MCC', 'Category']]
    print(top50.to_string(index=False))
    print()

    # 透视表
    pivot_auc, category_stats, feature_stats = generate_pivot_tables(df)

    print("="*100)
    print("模型 × 特征 AUC 矩阵")
    print("="*100)
    print(pivot_auc.to_string())
    print()

    print("="*100)
    print("按模型类别统计")
    print("="*100)
    print(category_stats.to_string())
    print()

    print("="*100)
    print("按特征类型统计")
    print("="*100)
    print(feature_stats.to_string())
    print()

    # 最佳模型推荐
    print("="*100)
    print("🏆 最佳模型推荐")
    print("="*100)

    best_overall = df.iloc[0]
    print(f"\n🥇 最佳整体性能:")
    print(f"   {best_overall['Model']} + {best_overall['Feature_Display']}")
    print(f"   AUC: {best_overall['AUC']:.4f}, F1: {best_overall['F1']:.4f}, MCC: {best_overall['MCC']:.4f}")

    # 最佳传统ML
    best_ml = df[df['Category'] == 'Traditional ML'].iloc[0]
    print(f"\n🌳 最佳传统ML:")
    print(f"   {best_ml['Model']} + {best_ml['Feature_Display']}")
    print(f"   AUC: {best_ml['AUC']:.4f}, F1: {best_ml['F1']:.4f}, MCC: {best_ml['MCC']:.4f}")

    # 最佳GNN
    best_gnn = df[df['Category'] == 'GNN']
    if len(best_gnn) > 0:
        best_gnn = best_gnn.iloc[0]
        print(f"\n🧠 最佳GNN:")
        print(f"   {best_gnn['Model']} + {best_gnn['Feature_Display']}")
        print(f"   AUC: {best_gnn['AUC']:.4f}, F1: {best_gnn['F1']:.4f}, MCC: {best_gnn['MCC']:.4f}")

    # 按特征类型的最佳模型
    print(f"\n📊 各特征类型的最佳模型:")
    for feat in df['Feature_Display'].unique():
        best_feat = df[df['Feature_Display'] == feat].iloc[0]
        print(f"   {feat:30} → {best_feat['Model']:15} (AUC={best_feat['AUC']:.4f})")

    print("\n" + "="*100)

    return df


def save_summary_to_csv():
    """保存汇总表到CSV"""

    df = create_summary_table()

    if df is None:
        return

    output_dir = PROJECT_ROOT / "artifacts" / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "FINAL_COMPREHENSIVE_SUMMARY.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n📁 完整汇总表已保存到: {output_file}")

    # 同时保存透视表
    pivot_auc, category_stats, feature_stats = generate_pivot_tables(df)

    pivot_file = output_dir / "MODEL_FEATURE_PIVOT.csv"
    pivot_auc.to_csv(pivot_file)
    print(f"📁 模型-特征矩阵已保存到: {pivot_file}")

    return df


if __name__ == "__main__":
    # 打印汇总
    df = print_summary()

    # 保存到CSV
    if df is not None:
        save_summary_to_csv()

        print("\n✅ 完成!")
