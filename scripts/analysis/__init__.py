"""
Analysis scripts for BBB permeability prediction project.

This directory contains scripts for analyzing and reporting baseline experiment results.

STRUCTURE:
    scripts/analysis/
    ├── aggregate_results.py              # Core: Aggregate experiment results
    ├── generate_benchmark_summary.py     # Core: Generate benchmark summary
    ├── run_baseline_matrix.py            # Core: Run baseline experiment matrix
    └── exploratory/                      # Exploratory/research analysis scripts
        ├── complete_analysis.py
        ├── comprehensive_cornelissen_analysis.py
        ├── comprehensive_cornelissen_analysis_v2.py
        ├── final_figures.py
        ├── improved_figures.py
        └── predict_molecules_mechanism.py

CORE BASELINE WORKFLOW:
    1. Run experiments: python scripts/analysis/run_baseline_matrix.py
    2. Aggregate results: python scripts/analysis/aggregate_results.py
    3. Generate summary: python scripts/analysis/generate_benchmark_summary.py

    Output: artifacts/reports/benchmark_summary.csv

EXPLORATORY SCRIPTS:
    - Cornelissen 2022 transport mechanism analysis
    - Figure generation and visualization
    - Mechanism prediction for custom molecules
    - NOT part of official baseline workflow

Usage:
    # Core baseline workflow
    python scripts/analysis/run_baseline_matrix.py
    python scripts/analysis/aggregate_results.py
    python scripts/analysis/generate_benchmark_summary.py

    # Exploratory analysis (optional)
    python scripts/analysis/exploratory/predict_molecules_mechanism.py
"""
