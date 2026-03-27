"""
使用2026-01-27T02-03_export.csv数据生成模型对比图表
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

PROJECT_ROOT = Path.cwd()
DATA_FILE = PROJECT_ROOT / "2026-01-27T02-03_export.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 颜色方案
MODEL_COLORS = {
    'RF': '#3498db',
    'RF+SMARTS': '#1a5276',
    'LGBM': '#2ecc71',
    'LGBM+SMARTS': '#1e8449',
    'XGB': '#f39c12',
    'XGB+SMARTS': '#a04000',
    'GAT': '#9b59b6',
    'GAT+SMARTS': '#6c3483'
}

MODEL_ORDER = ['RF', 'RF+SMARTS', 'LGBM', 'LGBM+SMARTS',
               'XGB', 'XGB+SMARTS', 'GAT', 'GAT+SMARTS']

DATASET_ORDER = ['A', 'A,B', 'A,B,C', 'A,B,C,D']
DATASET_SIZES = {'A': 106, 'A,B': 468, 'A,B,C': 776, 'A,B,C,D': 781}


def load_data():
    """加载导出的CSV数据"""
    print(f"Loading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)

    # 清理列名
    df.columns = df.columns.str.strip()

    # 重命名列
    if 'Dataset_Display' in df.columns:
        df = df.rename(columns={'Dataset_Display': 'Dataset'})

    # 标准化数据集名称
    df['Dataset'] = df['Dataset'].str.strip()

    # 打印数据信息
    print(f"\nData Overview:")
    print(f"Total records: {len(df)}")
    print(f"Datasets: {sorted(df['Dataset'].unique())}")
    print(f"Models: {sorted(df['Model'].unique())}")

    # 检查每个数据集的模型数量
    print("\nModels per dataset:")
    for ds in DATASET_ORDER:
        count = len(df[df['Dataset'] == ds])
        models = sorted(df[df['Dataset'] == ds]['Model'].unique())
        print(f"  {ds}: {count} models - {models}")

    return df


def plot_heatmap(df):
    """图1: 性能指标热图"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    metrics = ['AUC', 'Precision', 'Recall', 'F1']

    # 只使用数据中存在的模型
    available_models = [m for m in MODEL_ORDER if m in df['Model'].values]

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # 创建透视表
        pivot = df.pivot_table(values=metric, index='Dataset', columns='Model', aggfunc='mean')

        # 确保所有模型都在
        for model in available_models:
            if model not in pivot.columns:
                pivot[model] = np.nan

        pivot = pivot[available_models]

        # 绘制热图
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0.80, vmax=1.0,
                   ax=ax, linewidths=0.5,
                   cbar_kws={'label': metric})

        ax.set_title(f'{metric} Comparison Across Datasets',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.suptitle('Performance Metrics Heatmap: All 8 Models',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'fig1_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {output_path}")
    return output_path


def plot_barplot(df):
    """图2: AUC柱状图"""
    datasets = DATASET_ORDER
    available_models = [m for m in MODEL_ORDER if m in df['Model'].values]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // 2, idx % 2]

        df_subset = df[df['Dataset'] == dataset]

        if len(df_subset) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            continue

        x = np.arange(len(available_models))
        auc_values = []
        bar_colors = []

        for model in available_models:
            model_data = df_subset[df_subset['Model'] == model]
            if len(model_data) > 0:
                auc_values.append(model_data['AUC'].values[0])
                bar_colors.append(MODEL_COLORS.get(model, '#999'))
            else:
                auc_values.append(0)
                bar_colors.append('gray')

        bars = ax.bar(x, auc_values, color=bar_colors, alpha=0.8,
                     edgecolor='black', linewidth=1.5)

        # 添加数值标签
        for bar, val in zip(bars, auc_values):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom',
                       fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(available_models, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('AUC Score', fontsize=11)
        ax.set_title(f'Dataset: {dataset}', fontsize=13, fontweight='bold')
        ax.set_ylim(0.75, 1.0)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.suptitle('AUC Score Comparison: All 8 Models',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'fig2_barplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {output_path}")
    return output_path


def plot_scatter(df):
    """图3: AUC vs F1散点图 - 颜色=模型，形状=数据集"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # 数据集对应的形状
    dataset_markers = {
        'A': 'o',      # 圆形
        'A,B': 's',    # 方形
        'A,B,C': '^',  # 三角形
        'A,B,C,D': 'D' # 菱形
    }

    # 为每个数据集创建图例
    from matplotlib.lines import Line2D

    # 绘制所有点
    for model in MODEL_ORDER:
        if model not in df['Model'].values:
            continue

        model_data = df[df['Model'] == model]
        if len(model_data) > 0:
            # 按数据集分组绘制
            for dataset in DATASET_ORDER:
                ds_data = model_data[model_data['Dataset'] == dataset]
                if len(ds_data) > 0:
                    ax.scatter(ds_data['AUC'], ds_data['F1'],
                              s=150,
                              c=MODEL_COLORS.get(model, '#999'),
                              marker=dataset_markers.get(dataset, 'o'),
                              edgecolors='black', linewidth=1.5,
                              alpha=0.8, zorder=3)

    # 添加对角线（参考线）
    ax.plot([0.80, 1.0], [0.80, 1.0], 'k--', alpha=0.3, linewidth=1.5,
           label='Perfect Balance (AUC=F1)')

    ax.set_xlabel('AUC Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax.set_title('AUC vs F1: Performance Trade-off (Color=Model, Shape=Dataset)',
                fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0.80, 1.0)
    ax.set_ylim(0.80, 1.0)
    ax.grid(True, linestyle='--', alpha=0.3)

    # 创建双图例
    # 图例1: 模型（颜色）
    legend_elements_models = []
    for model in MODEL_ORDER:
        if model in df['Model'].values:
            legend_elements_models.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=MODEL_COLORS.get(model, '#999'),
                       markersize=10, markeredgecolor='black',
                       markeredgewidth=1.5, label=model)
            )

    # 图例2: 数据集（形状）
    legend_elements_datasets = []
    for dataset in DATASET_ORDER:
        legend_elements_datasets.append(
            Line2D([0], [0], marker=dataset_markers.get(dataset, 'o'),
                   color='w', markerfacecolor='gray',
                   markersize=10, markeredgecolor='black',
                   markeredgewidth=1.5, label=dataset)
        )

    # 添加两个图例
    legend1 = ax.legend(handles=legend_elements_models,
                       loc='lower left', fontsize=10,
                       framealpha=0.9, ncol=2, title='Model')

    legend2 = ax.legend(handles=legend_elements_datasets,
                       loc='upper right', fontsize=10,
                       framealpha=0.9, ncol=1, title='Dataset')

    ax.add_artist(legend1)  # 重新添加第一个图例

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'fig3_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {output_path}")
    return output_path


def plot_complexity(df):
    """图4: 数据集复杂度影响"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    datasets = [ds for ds in DATASET_ORDER if ds in df['Dataset'].values]
    available_models = [m for m in MODEL_ORDER if m in df['Model'].values]

    metrics = ['AUC', 'Precision', 'Recall', 'F1']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        for model in available_models:
            scores = []
            for dataset in datasets:
                df_subset = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
                if len(df_subset) > 0:
                    scores.append(df_subset[metric].values[0])
                else:
                    scores.append(np.nan)

            # 选择线型
            if '+SMARTS' in model:
                linestyle = '--'
                linewidth = 2.5
            elif model == 'GAT':
                linestyle = '-.'
                linewidth = 2.5
            else:
                linestyle = '-'
                linewidth = 2.5

            ax.plot(range(len(datasets)), scores, marker='o',
                   color=MODEL_COLORS.get(model, '#999'),
                   linestyle=linestyle, linewidth=linewidth,
                   markersize=7, label=model, alpha=0.8)

        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([f'{ds}\nn={DATASET_SIZES.get(ds, "?")}' for ds in datasets])
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} vs Dataset Size', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(0.80, 1.0)

    plt.suptitle('Dataset Complexity Impact: Performance vs Sample Size',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'fig4_complexity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {output_path}")
    return output_path


def plot_fixed_model(df):
    """图5: 固定模型，看数据集影响 - 选择4个代表性模型"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 选择4个代表性模型
    selected_models = ['XGB', 'LGBM', 'RF+SMARTS', 'GAT+SMARTS']

    metrics = ['AUC', 'Precision', 'Recall', 'F1']
    datasets = [ds for ds in DATASET_ORDER if ds in df['Dataset'].values]

    for idx, model in enumerate(selected_models):
        ax = axes[idx // 2, idx % 2]

        if model not in df['Model'].values:
            ax.text(0.5, 0.5, f'{model} not available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            continue

        # 为每个指标绘制柱状图
        x = np.arange(len(metrics))
        metric_values = []

        for metric in metrics:
            values = []
            for dataset in datasets:
                df_subset = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
                if len(df_subset) > 0:
                    values.append(df_subset[metric].values[0])
                else:
                    values.append(np.nan)
            metric_values.append(values)

        # 绘制分组柱状图
        width = 0.2
        for i, dataset in enumerate(datasets):
            values = [mv[i] for mv in metric_values]
            ax.bar(x + i * width, values, width,
                   label=dataset, alpha=0.8,
                   color=['#3498db', '#2ecc71', '#f39c12', '#9b59b6'][i],
                   edgecolor='black', linewidth=1.0)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'Model: {model} - Performance Across Datasets',
                    fontsize=13, fontweight='bold',
                    color=MODEL_COLORS.get(model, '#333'))
        ax.set_ylim(0.75, 1.0)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(loc='best', fontsize=9)

    plt.suptitle('Fixed Model: Performance Comparison Across Datasets',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'fig5_fixed_model.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {output_path}")
    return output_path


def plot_fixed_dataset(df):
    """图6: 固定数据集，看模型对比 - 4个数据集"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    metrics = ['AUC', 'Precision', 'Recall', 'F1']
    available_models = [m for m in MODEL_ORDER if m in df['Model'].values]

    for idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[idx // 2, idx % 2]

        df_subset = df[df['Dataset'] == dataset]

        if len(df_subset) == 0:
            ax.text(0.5, 0.5, f'{dataset} not available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            continue

        # 为每个指标绘制柱状图
        x = np.arange(len(available_models))
        metric_values = []

        for metric in metrics:
            values = []
            for model in available_models:
                model_data = df_subset[df_subset['Model'] == model]
                if len(model_data) > 0:
                    values.append(model_data[metric].values[0])
                else:
                    values.append(np.nan)
            metric_values.append(values)

        # 绘制分组柱状图
        width = 0.2
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

        for i, metric in enumerate(metrics):
            values = metric_values[i]
            ax.bar(x + i * width, values, width,
                   label=metric, alpha=0.8,
                   color=colors[i],
                   edgecolor='black', linewidth=0.8)

        # 设置颜色
        bar_colors = []
        for model in available_models:
            bar_colors.append(MODEL_COLORS.get(model, '#999'))

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(available_models, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'Dataset: {dataset} (n={DATASET_SIZES.get(dataset, "?")})',
                    fontsize=13, fontweight='bold')
        ax.set_ylim(0.75, 1.0)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(loc='best', fontsize=9, ncol=2)

    plt.suptitle('Fixed Dataset: Model Comparison Across All Datasets',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'fig6_fixed_dataset.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {output_path}")
    return output_path


def generate_summary_table(df):
    """生成性能摘要表格"""
    summary = df.groupby('Model')[['AUC', 'Precision', 'Recall', 'F1']].mean()
    summary['Avg_Score'] = summary.mean(axis=1)
    summary = summary.sort_values('Avg_Score', ascending=False)

    # 保存到CSV
    summary_path = OUTPUT_DIR / 'performance_summary.csv'
    summary.to_csv(summary_path)
    print(f"[OK] Saved summary: {summary_path}")

    # 打印到控制台
    print("\n" + "="*70)
    print("Performance Ranking (Averaged Across Datasets)")
    print("="*70)
    print(summary.to_string())
    print("="*70 + "\n")

    return summary


def main():
    """Main function"""
    print("="*70)
    print("Generating Model Comparison Visualizations")
    print("Using data: 2026-01-27T02-03_export.csv")
    print("="*70)

    # 加载数据
    df = load_data()

    # 生成图表
    print("\nGenerating charts...")
    print("-" * 70)

    plot_heatmap(df)
    plot_barplot(df)
    plot_scatter(df)
    plot_complexity(df)
    plot_fixed_model(df)
    plot_fixed_dataset(df)

    # 生成摘要
    summary = generate_summary_table(df)

    print("-" * 70)
    print(f"\nAll charts saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - fig1_heatmap.png (Performance metrics heatmap)")
    print("  - fig2_barplot.png (AUC comparison by dataset)")
    print("  - fig3_scatter.png (AUC vs F1 scatter, color=model, shape=dataset)")
    print("  - fig4_complexity.png (Dataset complexity impact)")
    print("  - fig5_fixed_model.png (Fixed model across datasets)")
    print("  - fig6_fixed_dataset.png (Fixed dataset across models)")
    print("  - performance_summary.csv")

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
