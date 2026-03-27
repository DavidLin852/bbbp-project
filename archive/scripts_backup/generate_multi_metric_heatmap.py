"""
Multi-Metric Heatmap Generator
生成多指标性能热图 - SE, SP, MCC, ACC, Precision, F1, BA, AUC
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

# Set plotting parameters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 50

PROJECT_ROOT = Path(__file__).parent.parent


def load_all_results():
    """加载所有实验结果"""

    results_dir = PROJECT_ROOT / "artifacts" / "ablation"

    # 传统ML结果
    df_ml = pd.read_csv(results_dir / "COMPLETE_MATRIX_RESULTS.csv")

    # 只保留测试集结果
    df_ml = df_ml[df_ml['split'] == 'test'].copy()

    # 计算额外指标
    # SE = Recall = TP / (TP + FN)
    df_ml['SE'] = df_ml['recall']  # 已经有recall列
    # SP = Specificity = TN / (TN + FP) = 1 - FPR
    df_ml['SP'] = df_ml['specificity']
    # BA = Balanced Accuracy = (SE + SP) / 2
    df_ml['BA'] = (df_ml['SE'] + df_ml['SP']) / 2

    return df_ml


def create_multi_metric_heatmap():
    """创建多指标热图"""

    df = load_all_results()

    # 定义指标及其显示名称
    metrics = [
        ('auc', 'AUC'),
        ('f1', 'F1 Score'),
        ('mcc', 'MCC'),
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall (SE)'),
        ('specificity', 'Specificity (SP)'),
        ('BA', 'Balanced Acc')
    ]

    # 特征显示名称
    feature_names = {
        'morgan': 'Morgan\n(2048D)',
        'maccs': 'MACCS\n(167D)',
        'atompairs': 'AtomPairs\n(1024D)',
        'fp2': 'FP2\n(2048D)',
        'rdkit_desc': 'RDKitDesc\n(98D)',
        'combined': 'Combined\n(5287D)'
    }

    # 创建图表 - 每个指标一个子图
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for idx, (metric_col, metric_name) in enumerate(metrics):
        ax = axes[idx]

        # 创建透视表
        pivot = df.pivot_table(
            index='model',
            columns='feature',
            values=metric_col
        )

        # 移除combined和graph特征（如果存在）
        if 'combined' in pivot.columns:
            pivot = pivot.drop(columns=['combined'])
        if 'graph' in pivot.columns:
            pivot = pivot.drop(columns=['graph'])

        # 重命名列
        pivot.columns = [feature_names.get(col, col) for col in pivot.columns]

        # 排序
        pivot_mean = pivot.mean(axis=1)
        pivot = pivot.loc[pivot_mean.sort_values(ascending=False).index]

        # 绘制热图
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',  # 红蓝色系，红色=高，蓝色=低
            vmin=0.7,
            vmax=1.0 if metric_col == 'auc' else 0.95,
            cbar_kws={'label': metric_name},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )

        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.suptitle(
        'BBB Permeability Prediction - Multi-Metric Performance Heatmap\n' +
        f'All Models × Individual Features',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout()

    # 保存图表
    output_dir = PROJECT_ROOT / "outputs" / "images" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "multi_metric_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 多指标热图已保存: {output_file}")

    return fig


def create_single_metric_detail():
    """创建单个指标的详细热图"""

    df = load_all_results()

    metrics_info = [
        ('auc', 'AUC', 0.85, 0.98, 'RdYlGn'),
        ('f1', 'F1 Score', 0.85, 0.98, 'RdYlGn'),
        ('mcc', 'MCC', 0.5, 0.9, 'RdYlGn'),
        ('BA', 'Balanced Accuracy', 0.7, 0.98, 'RdYlGn')
    ]

    feature_names = {
        'morgan': 'Morgan (2048D)',
        'maccs': 'MACCS (167D)',
        'atompairs': 'AtomPairs (1024D)',
        'fp2': 'FP2 (2048D)',
        'rdkit_desc': 'RDKitDesc (98D)',
        'combined': 'Combined (5287D)'
    }

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    for idx, (metric_col, metric_name, vmin, vmax, cmap) in enumerate(metrics_info):
        ax = axes[idx // 2, idx % 2]

        # 创建透视表
        pivot = df.pivot_table(
            index='model',
            columns='feature',
            values=metric_col
        )

        # 移除combined和graph特征（如果存在）
        if 'combined' in pivot.columns:
            pivot = pivot.drop(columns=['combined'])
        if 'graph' in pivot.columns:
            pivot = pivot.drop(columns=['graph'])

        # 重命名列
        pivot.columns = [feature_names.get(col, col) for col in pivot.columns]

        # 排序
        pivot_mean = pivot.mean(axis=1)
        pivot = pivot.loc[pivot_mean.sort_values(ascending=False).index]

        # 绘制热图
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': metric_name},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )

        ax.set_title(f'{metric_name}', fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Features', fontsize=11)
        ax.set_ylabel('Models', fontsize=11)

    plt.suptitle(
        'BBB Permeability Prediction - Key Performance Metrics Heatmap\n' +
        f'Color Scale: Red (High) → Blue (Low)',
        fontsize=15,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout()

    # 保存
    output_dir = PROJECT_ROOT / "outputs" / "images" / "model_comparison"
    output_file = output_dir / "key_metrics_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 关键指标热图已保存: {output_file}")

    return fig


def print_metric_summary():
    """打印指标统计摘要"""

    df = load_all_results()

    print("\n" + "="*80)
    print("各性能指标统计摘要")
    print("="*80)

    metrics = ['auc', 'f1', 'mcc', 'accuracy', 'precision', 'recall', 'specificity', 'BA']

    for metric in metrics:
        print(f"\n📊 {metric.upper()} 统计:")

        # 最佳值
        best_val = df[metric].max()
        best_model = df.loc[df[metric].idxmax(), 'model']
        best_feat = df.loc[df[metric].idxmax(), 'feature']

        print(f"  最佳值: {best_val:.4f}")
        print(f"  最佳组合: {best_model} + {best_feat}")

        # 平均值
        mean_val = df[metric].mean()
        print(f"  平均值: {mean_val:.4f}")

        # Top 3 模型
        top3 = df.nlargest(3, metric)[['model', 'feature', metric]]
        print(f"  Top 3:")
        for idx, row in top3.iterrows():
            print(f"    {row['model']:12} + {row['feature']:15} = {row[metric]:.4f}")


def main():
    """主函数"""

    print("="*80)
    print("生成多指标性能热图")
    print("="*80)
    print()

    # 生成多指标热图
    create_multi_metric_heatmap()

    # 生成关键指标详细热图
    create_single_metric_detail()

    # 打印统计摘要
    print_metric_summary()

    print("\n" + "="*80)
    print("完成！")
    print("="*80)


if __name__ == "__main__":
    main()
