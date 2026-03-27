"""
Generate evaluation reports.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .comparison import ModelComparison


def generate_report(
    comparison: ModelComparison,
    output_dir: Path | str = "artifacts/reports",
):
    """
    Generate evaluation report.

    Creates:
    - results_summary.csv: Main results table
    - best_model.txt: Info about best model

    Args:
        comparison: Model comparison results
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary table
    df = comparison.to_dataframe()
    summary_path = output_dir / "results_summary.csv"
    df.to_csv(summary_path, index=False)

    # Sorted by test AUC
    df_sorted = comparison.sort_by_test_auc(ascending=False)
    sorted_path = output_dir / "results_sorted_by_auc.csv"
    df_sorted.to_csv(sorted_path, index=False)

    # Best model info
    best = comparison.get_best_model()
    best_path = output_dir / "best_model.txt"
    best_path.write_text(
        f"Best Model: {best.model_name}\n"
        f"Feature Type: {best.feature_type}\n"
        f"Test AUC: {best.test_metrics.auc:.4f}\n"
        f"Test Accuracy: {best.test_metrics.accuracy:.4f}\n"
        f"Test F1: {best.test_metrics.f1_pos:.4f}\n",
        encoding="utf-8",
    )

    # Summary statistics
    summary = comparison.summary()
    summary_path = output_dir / "summary.json"
    import json
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
