#!/usr/bin/env python
"""
Generate benchmark summary from experimental results.

This script creates a stable benchmark summary table
aggregated across seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def create_benchmark_summary(
    results_csv: Path,
    output_csv: Path,
    min_seed_count: int = 3,
) -> pd.DataFrame:
    """
    Create benchmark summary table.

    Args:
        results_csv: Path to baseline_results_master.csv
        output_csv: Path to save benchmark summary
        min_seed_count: Minimum number of seeds required for stability

    Returns:
        Benchmark summary DataFrame
    """
    # Load results
    df = pd.read_csv(results_csv)

    # Filter to scaffold split only (our main split)
    df = df[df['split_type'] == 'scaffold'].copy()

    # Group by feature and model
    grouped = df.groupby(['feature', 'model_name']).agg({
        'test_auc': ['mean', 'std', 'count'],
        'test_f1': ['mean', 'std'],
        'train_auc': 'mean',
        'val_auc': 'mean',
    }).reset_index()

    # Flatten column names
    grouped.columns = [
        'feature',
        'model_name',
        'test_auc_mean',
        'test_auc_std',
        'seed_count',
        'test_f1_mean',
        'test_f1_std',
        'train_auc_mean',
        'val_auc_mean',
    ]

    # Round to 4 decimal places
    grouped = grouped.round({
        'test_auc_mean': 4,
        'test_auc_std': 4,
        'test_f1_mean': 4,
        'test_f1_std': 4,
        'train_auc_mean': 4,
        'val_auc_mean': 4,
    })

    # Filter by minimum seed count
    grouped = grouped[grouped['seed_count'] >= min_seed_count]

    # Sort by test_auc_mean descending
    grouped = grouped.sort_values('test_auc_mean', ascending=False)

    # Add rank
    grouped['rank'] = range(1, len(grouped) + 1)

    # Reorder columns
    grouped = grouped[[
        'rank',
        'feature',
        'model_name',
        'test_auc_mean',
        'test_auc_std',
        'test_f1_mean',
        'test_f1_std',
        'train_auc_mean',
        'val_auc_mean',
        'seed_count',
    ]]

    # Save to CSV
    grouped.to_csv(output_csv, index=False)

    return grouped


def identify_strongest_baseline(df: pd.DataFrame) -> dict:
    """
    Identify the strongest baseline from summary table.

    Args:
        df: Benchmark summary DataFrame

    Returns:
        Dict with baseline info
    """
    # Get top result
    top = df.iloc[0]

    baseline = {
        'feature': top['feature'],
        'model': top['model_name'],
        'test_auc_mean': float(top['test_auc_mean']),
        'test_auc_std': float(top['test_auc_std']),
        'test_f1_mean': float(top['test_f1_mean']),
        'test_f1_std': 0.0,  # Will be added if present
        'seed_count': int(top['seed_count']),
        'rank': int(top['rank']),
    }

    return baseline


def main():
    # Setup paths
    results_csv = Path("artifacts/reports/baseline_results_master.csv")
    output_csv = Path("artifacts/reports/benchmark_summary.csv")
    report_txt = Path("artifacts/reports/benchmark_report.txt")

    if not results_csv.exists():
        print(f"ERROR: Results file not found: {results_csv}")
        print("Please run experiments first:")
        print("  python scripts/analysis/run_baseline_matrix.py")
        return

    print("Creating benchmark summary...")

    # Create summary
    df = create_benchmark_summary(results_csv, output_csv, min_seed_count=3)

    print(f"\nSaved benchmark summary to: {output_csv}")

    # Identify strongest baseline
    baseline = identify_strongest_baseline(df)

    # Generate report
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("BBB PERMEABILITY PREDICTION - BASELINE BENCHMARK")
    report_lines.append("="*70)
    report_lines.append("")
    report_lines.append("DATASET: B3DB (Groups A+B)")
    report_lines.append("SPLIT: Scaffold split (80:10:10)")
    report_lines.append("SEEDS: 0, 1, 2")
    report_lines.append("METRIC: Test ROC AUC (primary)")
    report_lines.append("")
    report_lines.append("="*70)
    report_lines.append("MAIN CONCLUSION (Stable Baseline)")
    report_lines.append("="*70)
    report_lines.append("")
    report_lines.append(f"Strongest Classical Baseline:")
    report_lines.append(f"  Feature:  {baseline['feature']}")
    report_lines.append(f"  Model:    {baseline['model']}")
    report_lines.append(f"  Test AUC: {baseline['test_auc_mean']:.4f} ± {baseline['test_auc_std']:.4f}")
    report_lines.append(f"  Test F1:  {baseline['test_f1_mean']:.4f} ± {baseline['test_f1_std']:.4f}")
    report_lines.append(f"  Seeds:    {baseline['seed_count']}")
    report_lines.append("")
    report_lines.append("This baseline should be used as:")
    report_lines.append("  - Reference point for new models")
    report_lines.append("  - Minimum performance threshold")
    report_lines.append("  - Standard for scaffold-split evaluation")
    report_lines.append("")
    report_lines.append("="*70)
    report_lines.append("TEMPORARY OBSERVATIONS")
    report_lines.append("="*70)
    report_lines.append("")
    report_lines.append("Based on current experiments (may change with more data):")
    report_lines.append("")
    report_lines.append("1. Feature Performance:")

    # Feature performance
    feature_perf = df.groupby('feature')['test_auc_mean'].agg(['mean', 'std'])
    for feature, row in feature_perf.sort_values('mean', ascending=False).iterrows():
        report_lines.append(f"   - {feature}: {row['mean']:.4f} (avg across models)")

    report_lines.append("")
    report_lines.append("2. Model Performance:")

    # Model performance
    model_perf = df.groupby('model_name')['test_auc_mean'].agg(['mean', 'std'])
    for model, row in model_perf.sort_values('mean', ascending=False).iterrows():
        report_lines.append(f"   - {model}: {row['mean']:.4f} (avg across features)")

    report_lines.append("")
    report_lines.append("3. Stability (low std = more stable):")

    # Most stable (lowest std)
    most_stable = df.nsmallest(3, 'test_auc_std')[['feature', 'model_name', 'test_auc_std']]
    for _, row in most_stable.iterrows():
        report_lines.append(f"   - {row['feature']} + {row['model_name']}: std={row['test_auc_std']:.4f}")

    report_lines.append("")
    report_lines.append("="*70)
    report_lines.append("COMPLETE RESULTS TABLE")
    report_lines.append("="*70)
    report_lines.append("")

    # Add table
    report_lines.append(df.to_string(index=False))

    report_lines.append("")
    report_lines.append("="*70)
    report_lines.append("NEXT STEPS")
    report_lines.append("="*70)
    report_lines.append("")
    report_lines.append("Recommended follow-up experiments:")
    report_lines.append("")
    report_lines.append("1. Baseline Refinement:")
    report_lines.append("   - Hyperparameter tuning on best baseline")
    report_lines.append("   - Test additional features (maccs, fp2)")
    report_lines.append("   - Test additional models (svm, lr, knn)")
    report_lines.append("")
    report_lines.append("2. Ablation Studies:")
    report_lines.append("   - Compare scaffold vs random split")
    report_lines.append("   - Test different group filters (A,B,C)")
    report_lines.append("   - Analyze overfitting (train vs val vs test gap)")
    report_lines.append("")
    report_lines.append("3. Advanced Models (future):")
    report_lines.append("   - GNN models (GAT, GCN)")
    report_lines.append("   - Transformer models")
    report_lines.append("   - Ensemble methods")
    report_lines.append("")
    report_lines.append("NOTE: Do not proceed to advanced models until baseline")
    report_lines.append("      is fully understood and stable.")
    report_lines.append("")
    report_lines.append("="*70)

    # Write report
    report_txt.write_text('\n'.join(report_lines))

    print(f"\nSaved benchmark report to: {report_txt}")

    # Print summary to console
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"\nStrongest Baseline: {baseline['feature']} + {baseline['model']}")
    print(f"Test AUC: {baseline['test_auc_mean']:.4f} ± {baseline['test_auc_std']:.4f}")
    print(f"Test F1:  {baseline['test_f1_mean']:.4f} ± {baseline['test_f1_std']:.4f}")
    print(f"\nTop 5 Results:")
    print(df[['rank', 'feature', 'model_name', 'test_auc_mean', 'test_auc_std']].head().to_string(index=False))
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
