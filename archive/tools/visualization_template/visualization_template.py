"""
BBB Model Visualization Template - Python Version

Generate interactive model comparison visualizations from CSV data.

Usage:
    # Use existing data file
    python visualization_template.py --data complete_model_performance.csv

    # Use preset 1 (SMARTS importance)
    python visualization_template.py --preset 1

    # Use preset 2 (Dataset impact)
    python visualization_template.py --preset 2

    # Custom data and output
    python visualization_template.py --data my_data.csv --output-dir my_charts/

    # Generate specific chart only
    python visualization_template.py --preset 1 --chart heatmap
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys

# 设置样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

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
DATASET_SIZES = {'A': 106, 'A,B': 496, 'A,B,C': 1060, 'A,B,C,D': 6296}


def load_csv_data(csv_path):
    """从CSV文件加载模型性能数据"""
    df = pd.read_csv(csv_path)

    # 标准化列名
    df.columns = [col.strip().lower() for col in df.columns]

    # 重命名列为标准格式
    column_map = {}
    for col in df.columns:
        if 'dataset' in col.lower():
            column_map[col] = 'Dataset'
        elif 'model' in col.lower():
            column_map[col] = 'Model'
        elif 'auc' in col.lower():
            column_map[col] = 'AUC'
        elif 'precision' in col.lower():
            column_map[col] = 'Precision'
        elif 'recall' in col.lower():
            column_map[col] = 'Recall'
        elif 'f1' in col.lower():
            column_map[col] = 'F1'

    df = df.rename(columns=column_map)

    # 确保所有必需列都存在
    required_cols = ['Dataset', 'Model', 'AUC', 'Precision', 'Recall', 'F1']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"⚠️ Warning: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return None

    return df


def create_preset_1():
    """创建预设1: SMARTS预训练重要性 (A,B数据集)"""
    data = {
        'Dataset': ['A,B'] * 8,
        'Model': ['RF', 'RF+SMARTS', 'LGBM', 'LGBM+SMARTS',
                  'XGB', 'XGB+SMARTS', 'GAT', 'GAT+SMARTS'],
        'AUC': [0.984, 0.989, 0.964, 0.984, 0.950, 0.986, 0.921, 0.965],
        'Precision': [0.938, 0.946, 0.944, 0.968, 0.920, 0.968, 0.952, 0.950],
        'Recall': [0.980, 0.980, 0.941, 0.959, 0.940, 0.959, 0.977, 0.975],
        'F1': [0.958, 0.963, 0.943, 0.963, 0.930, 0.963, 0.964, 0.962]
    }
    return pd.DataFrame(data)


def create_preset_2():
    """创建预设2: 数据集影响分析 (SMARTS模型跨数据集)"""
    data = {
        'Dataset': ['A', 'A', 'A', 'A',
                    'A,B', 'A,B', 'A,B', 'A,B',
                    'A,B,C', 'A,B,C', 'A,B,C', 'A,B,C',
                    'A,B,C,D', 'A,B,C,D', 'A,B,C,D', 'A,B,C,D'],
        'Model': ['RF+SMARTS', 'LGBM+SMARTS', 'XGB+SMARTS', 'GAT+SMARTS',
                 'RF+SMARTS', 'LGBM+SMARTS', 'XGB+SMARTS', 'GAT+SMARTS',
                 'RF+SMARTS', 'LGBM+SMARTS', 'XGB+SMARTS', 'GAT+SMARTS',
                 'RF+SMARTS', 'LGBM+SMARTS', 'XGB+SMARTS', 'GAT+SMARTS'],
        'AUC': [0.927, 0.882, 0.910, 0.950,
                0.989, 0.984, 0.986, 0.965,
                0.954, 0.943, 0.943, 0.920,
                0.958, 0.944, 0.946, 0.935],
        'Precision': [0.903, 0.937, 0.935, 0.940,
                     0.946, 0.968, 0.968, 0.950,
                     0.877, 0.907, 0.894, 0.900,
                     0.874, 0.882, 0.888, 0.880],
        'Recall': [1.000, 0.957, 0.935, 0.970,
                  0.980, 0.959, 0.959, 0.975,
                  0.925, 0.887, 0.889, 0.930,
                  0.938, 0.905, 0.895, 0.910],
        'F1': [0.949, 0.947, 0.935, 0.955,
               0.963, 0.963, 0.963, 0.962,
               0.901, 0.897, 0.892, 0.915,
               0.905, 0.894, 0.892, 0.895]
    }
    return pd.DataFrame(data)


def print_data_summary(df):
    """打印数据摘要"""
    print("\n" + "="*70)
    print("📊 Data Summary")
    print("="*70)
    print(f"Total records: {len(df)}")
    print(f"Datasets: {sorted(df['Dataset'].unique())}")
    print(f"Models: {sorted(df['Model'].unique())}")
    print(f"Date ranges: AUC [{df['AUC'].min():.3f}, {df['AUC'].max():.3f}]")
    print("="*70 + "\n")


def plot_heatmap(df, output_dir):
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

    plt.suptitle('Performance Metrics Heatmap: All Models',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = output_dir / 'fig1_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}")
    return output_path


def plot_barplot(df, output_dir):
    """图2: AUC柱状图"""
    datasets = sorted(df['Dataset'].unique())
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
        ax.set_ylim(0.80, 1.0)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.suptitle('AUC Score Comparison: All Models',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = output_dir / 'fig2_barplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}")
    return output_path


def plot_scatter(df, output_dir):
    """图3: AUC vs F1散点图"""
    fig, ax = plt.subplots(figsize=(14, 10))

    markers = {
        'RF': 'o', 'RF+SMARTS': 'o',
        'LGBM': 's', 'LGBM+SMARTS': 's',
        'XGB': '^', 'XGB+SMARTS': '^',
        'GAT': 'D', 'GAT+SMARTS': 'D'
    }

    sizes = {
        'RF': 120, 'RF+SMARTS': 120,
        'LGBM': 120, 'LGBM+SMARTS': 120,
        'XGB': 120, 'XGB+SMARTS': 120,
        'GAT': 150, 'GAT+SMARTS': 150
    }

    for model in MODEL_ORDER:
        if model not in df['Model'].values:
            continue

        model_data = df[df['Model'] == model]
        if len(model_data) > 0:
            ax.scatter(model_data['AUC'], model_data['F1'],
                      s=sizes.get(model, 120),
                      c=MODEL_COLORS.get(model, '#999'),
                      marker=markers.get(model, 'o'),
                      edgecolors='black', linewidth=1.5,
                      label=model, alpha=0.8, zorder=3)

    # 添加对角线（参考线）
    ax.plot([0.80, 1.0], [0.80, 1.0], 'k--', alpha=0.3, linewidth=1.5,
           label='Perfect Balance (AUC=F1)')

    ax.set_xlabel('AUC Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax.set_title('AUC vs F1: Performance Trade-off',
                fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0.80, 1.0)
    ax.set_ylim(0.80, 1.0)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9, ncol=3)

    plt.tight_layout()

    output_path = output_dir / 'fig3_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}")
    return output_path


def plot_complexity(df, output_dir):
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

    output_path = output_dir / 'fig4_complexity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}")
    return output_path


def generate_summary_table(df, output_dir):
    """生成性能摘要表格"""
    summary = df.groupby('Model')[['AUC', 'Precision', 'Recall', 'F1']].mean()
    summary['Avg_Score'] = summary.mean(axis=1)
    summary = summary.sort_values('Avg_Score', ascending=False)

    # 保存到CSV
    summary_path = output_dir / 'performance_summary.csv'
    summary.to_csv(summary_path)
    print(f"✅ Saved summary: {summary_path}")

    # 打印到控制台
    print("\n" + "="*70)
    print("📊 Performance Ranking (Averaged Across Datasets)")
    print("="*70)
    print(summary.to_string())
    print("="*70 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Generate BBB model comparison visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use existing CSV data
  python visualization_template.py --data complete_model_performance.csv

  # Use preset 1 (SMARTS importance comparison)
  python visualization_template.py --preset 1

  # Use preset 2 (Dataset impact analysis)
  python visualization_template.py --preset 2

  # Generate specific chart only
  python visualization_template.py --preset 1 --chart heatmap

  # Custom output directory
  python visualization_template.py --preset 1 --output-dir my_charts/
        """
    )

    parser.add_argument('--data', type=str, help='Path to CSV data file')
    parser.add_argument('--preset', type=int, choices=[1, 2],
                       help='Use preset data (1=SMARTS importance, 2=Dataset impact)')
    parser.add_argument('--chart', type=str,
                       choices=['heatmap', 'barplot', 'scatter', 'complexity', 'all'],
                       default='all', help='Which chart(s) to generate (default: all)')
    parser.add_argument('--output-dir', type=str, default='outputs/visualization_template',
                       help='Output directory for charts (default: outputs/visualization_template)')

    args = parser.parse_args()

    # 加载数据
    df = None

    if args.preset:
        print(f"\n🎯 Using Preset {args.preset}")
        if args.preset == 1:
            df = create_preset_1()
            print("📋 Preset 1: SMARTS Pretraining Importance (A,B dataset)")
        elif args.preset == 2:
            df = create_preset_2()
            print("📋 Preset 2: Dataset Impact Analysis (SMARTS models)")

    elif args.data:
        print(f"\n📂 Loading data from: {args.data}")
        df = load_csv_data(args.data)
        if df is None:
            print("❌ Failed to load data. Please check CSV format.")
            sys.exit(1)

    else:
        print("\n❌ Error: Please specify either --data or --preset")
        parser.print_help()
        sys.exit(1)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # 打印数据摘要
    print_data_summary(df)

    # 生成图表
    print("\n🎨 Generating charts...")
    print("-" * 70)

    generated = []

    if args.chart in ['all', 'heatmap']:
        generated.append(plot_heatmap(df, output_dir))

    if args.chart in ['all', 'barplot']:
        generated.append(plot_barplot(df, output_dir))

    if args.chart in ['all', 'scatter']:
        generated.append(plot_scatter(df, output_dir))

    if args.chart in ['all', 'complexity']:
        generated.append(plot_complexity(df, output_dir))

    # 生成摘要表格
    summary = generate_summary_table(df, output_dir)

    print("-" * 70)
    print(f"\n✅ Successfully generated {len(generated)} charts!")
    print(f"📁 All files saved to: {output_dir}")
    print("\nGenerated files:")
    for path in generated:
        print(f"  - {path.name}")
    print(f"  - performance_summary.csv")

    print("\n🎉 Visualization complete!")
    print("\nNext steps:")
    print("  1. View charts in the output directory")
    print("  2. Use performance_summary.csv for detailed analysis")
    print("  3. Integrate charts into your presentation or paper")


if __name__ == "__main__":
    main()
