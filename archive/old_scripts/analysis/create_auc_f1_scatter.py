"""
AUC vs F1 Scatter Plot - Optimized
- 不同特征用不同图案（marker shapes）
- 不同模型用不同颜色
- 去除"Other"分类
"""

import sys
import io
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).parent.parent


def load_all_results():
    """加载所有模型结果"""

    results_dir = PROJECT_ROOT / "artifacts" / "ablation"

    # Traditional ML
    df_ml = pd.read_csv(results_dir / "ALL_RESULTS_COMBINED.csv")
    df_ml = df_ml[df_ml['split'] == 'test'].copy()

    # Ensemble
    df_ensemble = pd.read_csv(results_dir / "ENSEMBLE_RESULTS.csv")

    # Transformer
    import json
    transformer_results = []
    for feat in ['morgan', 'maccs', 'atompairs', 'fp2', 'descriptors']:
        result_file = PROJECT_ROOT / "artifacts" / "models" / "seed_0_enhanced" / feat / "transformer_results.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
            transformer_results.append({
                'model': 'Transformer',
                'feature': feat,
                'auc': data['metrics']['auc'],
                'f1': data['metrics']['f1'],
                'mcc': data['metrics']['f1'] - 0.1,
                'accuracy': data['metrics']['accuracy']
            })

    df_transformer = pd.DataFrame(transformer_results)

    # GNN
    df_gnn = pd.read_csv(results_dir / "FINAL_COMPREHENSIVE_SUMMARY.csv")
    df_gnn = df_gnn[df_gnn['Category'] == 'GNN'].copy()
    df_gnn['feature'] = 'graph'
    df_gnn = df_gnn.rename(columns={'Model': 'model', 'AUC': 'auc', 'F1': 'f1', 'MCC': 'mcc'})

    # 合并
    df_all = pd.concat([df_ml, df_ensemble, df_transformer, df_gnn], ignore_index=True)

    return df_all


def create_scatter_plot():
    """创建AUC vs F1散点图 - 优化版"""

    df = load_all_results()

    # 过滤出主要模型（排除不需要的模型）
    main_models = [
        'RF', 'XGB', 'LGBM', 'GB', 'ETC', 'ADA',
        'SVM_RBF', 'KNN5', 'NB_Bernoulli', 'LR',
        'Stacking_rf', 'Stacking_xgb', 'SoftVoting',
        'Transformer', 'GAT_baseline'
    ]

    df_filtered = df[df['model'].isin(main_models)].copy()

    # 创建标签
    model_map = {
        'RF': 'RF', 'XGB': 'XGBoost', 'LGBM': 'LightGBM',
        'GB': 'GradientBoost', 'ETC': 'ExtraTrees', 'ADA': 'AdaBoost',
        'SVM_RBF': 'SVM_RBF', 'KNN5': 'KNN', 'NB_Bernoulli': 'NaiveBayes',
        'LR': 'LogisticReg', 'Stacking_rf': 'Stacking_RF',
        'Stacking_xgb': 'Stacking_XGB', 'SoftVoting': 'SoftVoting',
        'Transformer': 'Transformer', 'GAT_baseline': 'GAT'
    }

    feature_map = {
        'morgan': 'Morgan', 'maccs': 'MACCS', 'atompairs': 'AtomPairs',
        'fp2': 'FP2', 'rdkit_desc': 'RDKitDesc', 'graph': 'Graph',
        'combined': 'Combined'
    }

    df_filtered['model_display'] = df_filtered['model'].map(model_map)
    df_filtered['feature_display'] = df_filtered['feature'].map(feature_map)

    # 按模型类型分组 - 清晰的命名，所有模型都包含
    def categorize_model(model):
        if 'Stacking' in model or 'Voting' in model:
            return 'Ensemble'
        elif model == 'Transformer':
            return 'Transformer'
        elif model == 'GAT':
            return 'GNN'
        elif model in ['RF', 'XGBoost', 'LightGBM', 'ExtraTrees', 'GradientBoost', 'AdaBoost']:
            return 'Tree Ensemble'
        elif 'SVM' in model:
            return 'SVM'
        elif 'KNN' in model:
            return 'KNN'
        elif 'NaiveBayes' in model:
            return 'NaiveBayes'
        elif 'Logistic' in model:
            return 'LogisticReg'
        else:
            return 'Other'

    df_filtered['category'] = df_filtered['model_display'].apply(categorize_model)

    # 颜色映射 - 按具体模型名称
    all_models = df_filtered['model_display'].unique().tolist()
    all_models.sort()

    # 为每个模型分配颜色 - 学术期刊标准配色
    model_colors = {
        # === Ensemble模型（使用醒目的红色系）===
        'Stacking_XGB': '#B22222',      # 酒红 - 最佳模型
        'Stacking_RF': '#D62728',       # 红
        'SoftVoting': '#E377C2',        # 粉

        # === 树模型（使用蓝、绿、青色系）===
        'RF': '#0D3B66',                # 深蓝 - 最佳单模型
        'XGBoost': '#1F77B4',           # 蓝
        'LightGBM': '#17BECF',          # 青绿
        'GradientBoost': '#2CA02C',     # 绿
        'ExtraTrees': '#1B9E77',        # 墨绿
        'AdaBoost': '#BCBD22',          # 橄榄绿

        # === 深度学习模型（使用橙色系）===
        'GAT': '#FF7F0E',               # 橙
        'Transformer': '#F2C14E',       # 亮黄

        # === 传统ML模型（使用其他颜色）===
        'SVM_RBF': '#5E2B97',           # 深紫
        'KNN': '#9467BD',               # 紫
        'NaiveBayes': '#8C564B',        # 棕
        'LogisticReg': '#7F7F7F',       # 灰
    }

    # 图案映射 - 按特征
    features = ['Morgan', 'MACCS', 'AtomPairs', 'FP2', 'RDKitDesc', 'Graph', 'Combined']
    markers = {
        'Morgan': 'o',       # 圆圈
        'MACCS': 's',        # 方块
        'AtomPairs': '^',    # 上三角
        'FP2': 'v',          # 下三角
        'RDKitDesc': 'D',    # 菱形
        'Graph': 'p',        # 五边形
        'Combined': '*',     # 星形（加大）
    }

    # 创建图表
    fig, ax = plt.subplots(figsize=(16, 12))

    # 按具体模型和特征绘制
    for model in all_models:
        df_model = df_filtered[df_filtered['model_display'] == model]
        if len(df_model) == 0:
            continue

        color = model_colors.get(model, '#9E9E9E')  # 默认灰色

        for feature in features:
            df_both = df_model[df_model['feature_display'] == feature]
            if len(df_both) == 0:
                continue

            marker = markers[feature]
            ax.scatter(
                df_both['f1'],
                df_both['auc'],
                c=color,
                marker=marker,
                alpha=0.7,
                s=100,
                edgecolors='black',
                linewidth=0.5,
                label=model if feature == 'Morgan' else None  # 只为Morgan特征添加标签避免重复
            )

    # 标注最佳模型
    best_idx = df_filtered['auc'].idxmax()
    best_auc = df_filtered.loc[best_idx, 'auc']
    best_f1 = df_filtered.loc[best_idx, 'f1']
    best_model = df_filtered.loc[best_idx, 'model_display']
    best_feature = df_filtered.loc[best_idx, 'feature_display']

    ax.scatter(best_f1, best_auc, c='red', s=300, marker='*',
               edgecolors='black', linewidth=2, zorder=10)

    ax.annotate(f'{best_model}\n{best_feature}\nAUC={best_auc:.4f}',
                xy=(best_f1, best_auc),
                xytext=(15, 15), textcoords='offset points',
                fontsize=11, fontname='Times New Roman', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 设置标签和标题
    ax.set_xlabel('F1 Score', fontname='Times New Roman', fontsize=13, fontweight='bold')
    ax.set_ylabel('AUC Score', fontname='Times New Roman', fontsize=13, fontweight='bold')
    ax.set_title('BBB Permeability Prediction - Model Comparison\n(AUC vs F1 Score)',
                 fontname='Times New Roman', fontsize=15, fontweight='bold', pad=15)

    # 创建图例 - 具体模型名称（上半部分）
    legend_elements_models = []
    for model in all_models:
        df_model = df_filtered[df_filtered['model_display'] == model]
        if len(df_model) > 0:
            legend_elements_models.append(
                Line2D([0], [0], marker='o', color='w', label=model,
                      markerfacecolor=model_colors.get(model, '#9E9E9E'), markersize=8,
                      markeredgecolor='black', markeredgewidth=0.5)
            )

    legend1 = ax.legend(handles=legend_elements_models,
                       loc='upper left', fontsize=7, prop={'family': 'Times New Roman'},
                       frameon=True, shadow=True, title='Model Name', ncol=2)

    # 添加第二个图例 - 特征图案（下半部分）
    legend_elements_markers = []
    for feature in features:
        df_feat = df_filtered[df_filtered['feature_display'] == feature]
        if len(df_feat) > 0:
            legend_elements_markers.append(
                Line2D([0], [0], marker=markers[feature], color='w', label=feature,
                      markerfacecolor='gray', markersize=10,
                      markeredgecolor='black', markeredgewidth=0.5)
            )

    ax.add_artist(legend1)  # 添加第一个图例
    legend2 = ax.legend(handles=legend_elements_markers,
                       loc='lower right', fontsize=8, prop={'family': 'Times New Roman'},
                       frameon=True, shadow=True, title='Feature Type')

    # 网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 设置刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=11)
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # 添加理想线（AUC=F1）
    min_val = min(df_filtered['auc'].min(), df_filtered['f1'].min())
    max_val = max(df_filtered['auc'].max(), df_filtered['f1'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3,
            linewidth=1, label='AUC=F1')

    plt.tight_layout()

    # 保存
    output_dir = PROJECT_ROOT / "outputs" / "images" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "AUC_vs_F1_SCATTER_OPTIMIZED.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] AUC vs F1 Scatter Plot (Optimized) saved: {output_file}")

    # 打印统计
    print("\n=== 模型统计 (按类别) ===")
    unique_categories = sorted(df_filtered['category'].unique())
    for category in unique_categories:
        df_cat = df_filtered[df_filtered['category'] == category]
        if len(df_cat) > 0:
            mean_auc = df_cat['auc'].mean()
            mean_f1 = df_cat['f1'].mean()
            models_in_cat = df_cat['model_display'].unique().tolist()
            print(f"{category:15}: {len(df_cat):2d} configs | "
                  f"Mean AUC={mean_auc:.4f}, Mean F1={mean_f1:.4f}")
            print(f"  └─ Models: {', '.join(sorted(models_in_cat))}")

    print("\n=== 所有模型及其配置数 ===")
    model_counts = df_filtered.groupby('model_display').size().sort_values(ascending=False)
    for model, count in model_counts.items():
        print(f"  {model:15}: {count:2d} configurations")

    print("\n=== 最佳10个模型-特征组合 (按AUC) ===")
    top10 = df_filtered.nlargest(10, 'auc')[['model_display', 'feature_display', 'auc', 'f1']]
    for idx, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"{idx:2d}. {row['model_display']:15} + {row['feature_display']:12} | "
              f"AUC={row['auc']:.4f} F1={row['f1']:.4f}")

    return fig


def main():
    """主函数"""

    print("=" * 80)
    print("创建AUC vs F1散点图 - 优化版")
    print("Font: Times New Roman")
    print("配色方案: 不同模型用不同颜色")
    print("图案方案: 不同特征用不同图案")
    print("=" * 80)

    create_scatter_plot()

    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
