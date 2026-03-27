"""
Analyze mechanism prediction results and compare with literature.

This script:
1. Analyzes feature importance for each mechanism
2. Compares physicochemical property distributions
3. Validates findings against Cornelissen et al. 2022
4. Generates visualization summaries
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Paths
from xgboost import XGBClassifier


def load_results():
    """Load training results."""
    results_file = Paths.artifacts / "models" / "cornelissen_2022" / "training_results.json"
    with open(results_file, 'r') as f:
        return json.load(f)


def load_data():
    """Load processed data."""
    data_dir = Paths.root / "data" / "transport_mechanisms" / "cornelissen_2022"
    data_file = data_dir / "cornelissen_2022_processed.csv"
    return pd.read_csv(data_file)


def analyze_physicochemical_properties(df):
    """Analyze physicochemical properties by mechanism."""
    physicochemical_cols = [
        'LogP', 'TPSA', 'MW', 'HBA', 'HBD',
        'RotatableBonds', 'RingCount', 'FractionCSP3'
    ]

    results = {}

    for mechanism in ['Influx', 'Efflux', 'PAMPA', 'BBB', 'CNS']:
        label_col = f'label_{mechanism}'

        # Get samples with labels
        mask = df[label_col].notna()
        df_mech = df[mask].copy()

        if len(df_mech) == 0:
            continue

        # Separate by class
        df_pos = df_mech[df_mech[label_col] == 1]
        df_neg = df_mech[df_mech[label_col] == 0]

        results[mechanism] = {
            'positive': {},
            'negative': {},
            'n_positive': len(df_pos),
            'n_negative': len(df_neg),
        }

        # Calculate statistics
        for col in physicochemical_cols:
            if col in df_mech.columns:
                results[mechanism]['positive'][col] = {
                    'mean': float(df_pos[col].mean()),
                    'std': float(df_pos[col].std()),
                    'median': float(df_pos[col].median()),
                }
                results[mechanism]['negative'][col] = {
                    'mean': float(df_neg[col].mean()),
                    'std': float(df_neg[col].std()),
                    'median': float(df_neg[col].median()),
                }

    return results


def print_literature_comparison():
    """Print comparison with Cornelissen et al. 2022 findings."""
    print("\n" + "="*80)
    print("Comparison with Cornelissen et al. 2022")
    print("="*80)

    print("\nKey Findings from Literature:")
    print("-" * 80)
    print("1. BBB Permeability:")
    print("   - Key Feature: TPSA (primary predictor)")
    print("   - Lower TPSA (<90 A^2) favors BBB penetration")
    print("   - Optimal MW: 200-500 Da")
    print("   - LogP: 1-3 (moderate lipophilicity)")

    print("\n2. PAMPA (Passive Diffusion):")
    print("   - Key Feature: LogD (lipophilicity)")
    print("   - High lipophilicity favors passive diffusion")
    print("   - MW <500 preferred")

    print("\n3. Influx (Active Transport):")
    print("   - Key Features: HBD, MACCS43, MACCS36")
    print("   - Higher TPSA and MW common")
    print("   - May utilize nutrient transporters")

    print("\n4. Efflux (P-gp substrates):")
    print("   - Key Features: MW, MACCS8 (beta-lactams)")
    print("   - Higher MW associated with efflux")
    print("   - Beta-lactam substructure common")


def print_our_findings(property_results, model_results):
    """Print our findings and compare with literature."""
    print("\n" + "="*80)
    print("Our Findings from Cornelissen 2022 Dataset")
    print("="*80)

    # BBB Analysis
    print("\n1. BBB Permeability:")
    print("-" * 40)
    bbb_pos = property_results['BBB']['positive']
    bbb_neg = property_results['BBB']['negative']

    print(f"   Samples: {property_results['BBB']['n_positive']} positive, {property_results['BBB']['n_negative']} negative")
    print(f"\n   Physicochemical Properties:")
    print(f"   {'Property':<15} {'BBB+':>10} {'BBB-':>10} {'Difference':>12}")
    print(f"   {'-'*50}")

    for prop in ['TPSA', 'MW', 'LogP', 'HBA', 'HBD']:
        if prop in bbb_pos:
            pos_val = bbb_pos[prop]['mean']
            neg_val = bbb_neg[prop]['mean']
            diff = pos_val - neg_val
            print(f"   {prop:<15} {pos_val:>10.2f} {neg_val:>10.2f} {diff:>+11.2f}")

    # Top features
    print(f"\n   Top 5 Important Features:")
    for i, (feat, imp) in enumerate(model_results['BBB']['feature_importance'][:5], 1):
        print(f"   {i}. {feat:<30} {imp:.4f}")

    print(f"\n   Interpretation:")
    if 'TPSA' in bbb_pos and bbb_pos['TPSA']['mean'] < bbb_neg['TPSA']['mean']:
        print(f"   [OK] BBB+ molecules have LOWER TPSA (consistent with literature)")
    if 'MW' in bbb_pos and bbb_pos['MW']['mean'] < bbb_neg['MW']['mean']:
        print(f"   [OK] BBB+ molecules have LOWER MW (consistent with literature)")

    # PAMPA Analysis
    print("\n2. PAMPA (Passive Diffusion):")
    print("-" * 40)
    pampa_pos = property_results['PAMPA']['positive']
    pampa_neg = property_results['PAMPA']['negative']

    print(f"   Samples: {property_results['PAMPA']['n_positive']} positive, {property_results['PAMPA']['n_negative']} negative")
    print(f"\n   Physicochemical Properties:")
    print(f"   {'Property':<15} {'PAMPA+':>10} {'PAMPA-':>10} {'Difference':>12}")
    print(f"   {'-'*50}")

    for prop in ['TPSA', 'MW', 'LogP', 'HBA', 'HBD']:
        if prop in pampa_pos:
            pos_val = pampa_pos[prop]['mean']
            neg_val = pampa_neg[prop]['mean']
            diff = pos_val - neg_val
            print(f"   {prop:<15} {pos_val:>10.2f} {neg_val:>10.2f} {diff:>+11.2f}")

    # Top features
    print(f"\n   Top 5 Important Features:")
    for i, (feat, imp) in enumerate(model_results['PAMPA']['feature_importance'][:5], 1):
        print(f"   {i}. {feat:<30} {imp:.4f}")

    print(f"\n   Interpretation:")
    if 'LogP' in pampa_pos and pampa_pos['LogP']['mean'] > pampa_neg['LogP']['mean']:
        print(f"   [OK] PAMPA+ molecules have HIGHER LogP (consistent with literature)")

    # Influx Analysis
    print("\n3. Influx (Active Transport):")
    print("-" * 40)
    influx_pos = property_results['Influx']['positive']
    influx_neg = property_results['Influx']['negative']

    print(f"   Samples: {property_results['Influx']['n_positive']} positive, {property_results['Influx']['n_negative']} negative")
    print(f"\n   Physicochemical Properties:")
    print(f"   {'Property':<15} {'Influx+':>10} {'Influx-':>10} {'Difference':>12}")
    print(f"   {'-'*50}")

    for prop in ['TPSA', 'MW', 'LogP', 'HBA', 'HBD']:
        if prop in influx_pos:
            pos_val = influx_pos[prop]['mean']
            neg_val = influx_neg[prop]['mean']
            diff = pos_val - neg_val
            print(f"   {prop:<15} {pos_val:>10.2f} {neg_val:>10.2f} {diff:>+11.2f}")

    # Top features
    print(f"\n   Top 5 Important Features:")
    for i, (feat, imp) in enumerate(model_results['Influx']['feature_importance'][:5], 1):
        print(f"   {i}. {feat:<30} {imp:.4f}")

    print(f"\n   Interpretation:")
    if 'TPSA' in influx_pos and influx_pos['TPSA']['mean'] > influx_neg['TPSA']['mean']:
        print(f"   [OK] Influx+ molecules have HIGHER TPSA (consistent with literature)")
    if 'HBA' in influx_pos and influx_pos['HBA']['mean'] > influx_neg['HBA']['mean']:
        print(f"   [OK] Influx+ molecules have MORE HBA (consistent with literature)")

    # Efflux Analysis
    print("\n4. Efflux (Active Efflux):")
    print("-" * 40)
    efflux_pos = property_results['Efflux']['positive']
    efflux_neg = property_results['Efflux']['negative']

    print(f"   Samples: {property_results['Efflux']['n_positive']} positive, {property_results['Efflux']['n_negative']} negative")
    print(f"\n   Physicochemical Properties:")
    print(f"   {'Property':<15} {'Efflux+':>10} {'Efflux-':>10} {'Difference':>12}")
    print(f"   {'-'*50}")

    for prop in ['TPSA', 'MW', 'LogP', 'HBA', 'HBD']:
        if prop in efflux_pos:
            pos_val = efflux_pos[prop]['mean']
            neg_val = efflux_neg[prop]['mean']
            diff = pos_val - neg_val
            print(f"   {prop:<15} {pos_val:>10.2f} {neg_val:>10.2f} {diff:>+11.2f}")

    # Top features
    print(f"\n   Top 5 Important Features:")
    for i, (feat, imp) in enumerate(model_results['Efflux']['feature_importance'][:5], 1):
        print(f"   {i}. {feat:<30} {imp:.4f}")

    print(f"\n   Interpretation:")
    if 'MW' in efflux_pos and efflux_pos['MW']['mean'] > efflux_neg['MW']['mean']:
        print(f"   [OK] Efflux+ molecules have HIGHER MW (consistent with literature)")


def main():
    """Main analysis function."""
    print("="*80)
    print("Mechanism Prediction Analysis")
    print("Cornelissen et al. 2022 Dataset")
    print("="*80)

    # Load data
    print("\n1. Loading data...")
    model_results = load_results()
    df = load_data()

    # Analyze properties
    print("2. Analyzing physicochemical properties...")
    property_results = analyze_physicochemical_properties(df)

    # Save property analysis
    output_dir = Paths.artifacts / "analysis" / "cornelissen_2022"
    output_dir.mkdir(parents=True, exist_ok=True)

    property_file = output_dir / "physicochemical_analysis.json"
    with open(property_file, 'w') as f:
        json.dump(property_results, f, indent=2)
    print(f"   Saved to: {property_file}")

    # Print literature comparison
    print_literature_comparison()

    # Print our findings
    print_our_findings(property_results, model_results)

    # Model performance summary
    print("\n" + "="*80)
    print("Model Performance Summary")
    print("="*80)
    print(f"{'Mechanism':<12} {'AUC':>8} {'Accuracy':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 60)

    for mech, res in model_results.items():
        metrics = res['metrics']
        print(f"{mech:<12} {metrics['auc']:>8.4f} {metrics['accuracy']:>8.4f} {metrics['f1']:>8.4f} {metrics['precision']:>10.4f} {metrics['recall']:>8.4f}")

    print("\n" + "="*80)
    print("Key Insights")
    print("="*80)

    # Check for consistency with literature
    insights = []

    # BBB
    if 'BBB' in property_results:
        bbb_pos_tpsa = property_results['BBB']['positive'].get('TPSA', {}).get('mean', 0)
        bbb_neg_tpsa = property_results['BBB']['negative'].get('TPSA', {}).get('mean', 0)
        if bbb_pos_tpsa < bbb_neg_tpsa:
            insights.append("[OK] BBB penetration correlates with LOWER TPSA (consistent with literature)")

    # PAMPA
    if 'PAMPA' in property_results:
        pampa_pos_logp = property_results['PAMPA']['positive'].get('LogP', {}).get('mean', 0)
        pampa_neg_logp = property_results['PAMPA']['negative'].get('LogP', {}).get('mean', 0)
        if pampa_pos_logp > pampa_neg_logp:
            insights.append("[OK] PAMPA permeability correlates with HIGHER LogP (consistent with literature)")

    # Influx
    if 'Influx' in property_results:
        influx_pos_tpsa = property_results['Influx']['positive'].get('TPSA', {}).get('mean', 0)
        influx_neg_tpsa = property_results['Influx']['negative'].get('TPSA', {}).get('mean', 0)
        if influx_pos_tpsa > influx_neg_tpsa:
            insights.append("[OK] Influx transport correlates with HIGHER TPSA (consistent with literature)")

    for insight in insights:
        print(f"\n{insight}")

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
