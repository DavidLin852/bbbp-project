"""
Generate Complete Model Performance Heatmap
生成所有模型性能对比的热图
"""

import sys
import io
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_all_results():
    """加载所有模型结果"""

    results_dir = PROJECT_ROOT / "artifacts" / "ablation"

    # 传统ML模型
    df_ml = pd.read_csv(results_dir / "COMPLETE_MATRIX_RESULTS.csv")

    # 创建透视表
    pivot = df_ml.pivot_table(
        index='model',
        columns='feature',
        values='auc'
    )

    # 确保列顺序
    feature_order = ['morgan', 'maccs', 'atompairs', 'fp2', 'rdkit_desc', 'combined']
    pivot = pivot.reindex(columns=feature_order)

    # 添加GAT模型（单独特征）
    pivot['graph_gat'] = np.nan
    pivot.loc['GAT_baseline', 'graph_gat'] = 0.9343
    pivot.loc['GAT+SMARTS', 'graph_gat'] = 0.9560
    pivot.loc['GAT+SMARTS(full)', 'graph_gat'] = 0.9479

    # 重命名列和行
    pivot.columns = [
        'Morgan\n(2048D)',
        'MACCS\n(167D)',
        'AtomPairs\n(1024D)',
        'FP2\n(2048D)',
        'RDKitDesc\n(98D)',
        'Combined\n(5287D)',
        'Graph\n(GAT)'
    ]

    # 模型显示名称
    model_names = {
        'RF': 'Random Forest',
        'XGB': 'XGBoost',
        'LGBM': 'LightGBM',
        'ETC': 'Extra Trees',
        'GB': 'Gradient Boosting',
        'ADA': 'AdaBoost',
        'SVM_RBF': 'SVM (RBF)',
        'SVM_LINEAR': 'SVM (Linear)',
        'SVM_POLY': 'SVM (Poly)',
        'KNN3': 'KNN (K=3)',
        'KNN5': 'KNN (K=5)',
        'KNN7': 'KNN (K=7)',
        'NB_Gaussian': 'NB (Gaussian)',
        'NB_Bernoulli': 'NB (Bernoulli)',
        'LR': 'Logistic Regression',
        'MLP': 'MLP',
        'MLP_Small': 'MLP (Small)',
        'GAT_baseline': 'GAT (Baseline)',
        'GAT+SMARTS': 'GAT+SMARTS',
        'GAT+SMARTS(full)': 'GAT+SMARTS(full)'
    }

    pivot = pivot.rename(index=model_names)

    # 排序 - 按平均性能
    pivot['mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean', ascending=False)
    pivot = pivot.drop('mean', axis=1)

    return pivot


def create_heatmap():
    """创建性能热图"""

    df = load_all_results()

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 10))

    # 绘制热图
    sns.heatmap(
        df,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.85,
        vmax=0.98,
        cbar_kws={'label': 'AUC Score'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    ax.set_title(
        'BBB Permeability Prediction - Complete Model Performance Comparison\n' +
        f'All Models × All Features Heatmap',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Models', fontsize=12, fontweight='bold')

    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0, ha='right')

    plt.tight_layout()

    # 保存图表
    output_dir = PROJECT_ROOT / "outputs" / "images" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "complete_model_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved to: {output_file}")

    return fig, df


def create_category_heatmap():
    """创建按模型类别分组的热图"""

    df = load_all_results()

    # 定义模型类别
    categories = {
        'Tree Ensemble': [
            'Random Forest', 'XGBoost', 'LightGBM', 'Extra Trees',
            'Gradient Boosting', 'AdaBoost'
        ],
        'SVM': ['SVM (RBF)', 'SVM (Linear)', 'SVM (Poly)'],
        'KNN': ['KNN (K=3)', 'KNN (K=5)', 'KNN (K=7)'],
        'Neural Network': ['MLP', 'MLP (Small)'],
        'Probabilistic': ['NB (Gaussian)', 'NB (Bernoulli)', 'Logistic Regression'],
        'GNN': ['GAT (Baseline)', 'GAT+SMARTS', 'GAT+SMARTS(full)']
    }

    # 为每个模型添加类别标签
    df_with_category = df.reset_index().copy()
    df_with_category['Category'] = 'Other'
    for category, models in categories.items():
        for model in models:
            if model in df_with_category['Model'].values:
                df_with_category.loc[df_with_category['Model'] == model, 'Category'] = category

    # 创建分面热图
    categories_order = ['Tree Ensemble', 'Neural Network', 'GNN', 'SVM', 'KNN', 'Probabilistic']
    df_with_category['Category'] = pd.Categorical(
        df_with_category['Category'],
        categories=categories_order
    )
    df_with_category = df_with_category.sort_values('Category')

    # 绘图
    g = sns.FacetGrid(
        df_with_category.melt(
            id_vars=['Model', 'Category'],
            var_name='Feature',
            value_name='AUC'
        ),
        col='Category',
        col_wrap=3,
        height=4,
        aspect=1.2,
        sharey=False
    )

    g.map_dataframe(
        lambda data, **kwargs: sns.heatmap(
            data.pivot(index='Model', columns='Feature', values='AUC'),
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0.85,
            vmax=0.98,
            linewidths=0.5,
            cbar=True,
            **kwargs
        )
    )

    g.fig.suptitle(
        'BBB Permeability Prediction - Model Performance by Category',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )

    # 调整布局
    plt.tight_layout()

    # 保存
    output_dir = PROJECT_ROOT / "outputs" / "images" / "model_comparison"
    output_file = output_dir / "model_heatmap_by_category.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Category heatmap saved to: {output_file}")

    return g


def print_summary_statistics():
    """打印统计摘要"""

    df = load_all_results()

    print("\n" + "="*80)
    print("模型性能统计摘要")
    print("="*80)

    # 按模型统计
    print("\n📊 按模型统计 (平均AUC):")
    model_stats = df.mean(axis=1).sort_values(ascending=False)
    for model, auc in model_stats.head(10).items():
        print(f"  {model:25} {auc:.4f}")

    print("\n📊 按特征统计 (平均AUC):")
    feature_stats = df.mean(axis=0).sort_values(ascending=False)
    for feat, auc in feature_stats.items():
        print(f"  {feat:20} {auc:.4f}")

    print("\n🏆 最佳模型-特征组合:")
    # 找到最大值（忽略NaN）
    max_val = df.stack().idxmax()
    model, feature = max_val
    auc = df.loc[model, feature]
    print(f"  {model:25} + {feature:20} = {auc:.4f}")

    print("\n📈 性能分布:")
    all_values = df.stack().dropna()
    print(f"  最高AUC: {all_values.max():.4f}")
    print(f"  最低AUC: {all_values.min():.4f}")
    print(f"  平均AUC: {all_values.mean():.4f}")
    print(f"  中位数AUC: {all_values.median():.4f}")


def main():
    """主函数"""

    print("="*80)
    print("生成完整模型性能热图")
    print("="*80)
    print()

    # 生成主热图
    fig, df = create_heatmap()

    # 生成分类热图
    # create_category_heatmap()

    # 打印统计
    print_summary_statistics()

    print("\n" + "="*80)
    print("完成！")
    print("="*80)


if __name__ == "__main__":
    main()
