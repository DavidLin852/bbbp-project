"""
Analyze mechanism distribution in B3DB dataset using integrated predictor.

This script:
1. Loads B3DB dataset
2. Predicts mechanisms for all molecules
3. Analyzes mechanism distribution
4. Creates visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Set Chinese font for matplotlib
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Paths
from src.path_prediction.integrated_mechanism_predictor import IntegratedMechanismPredictor


def analyze_b3db_mechanisms():
    """Analyze mechanism distribution in B3DB dataset."""
    print("="*80)
    print("Analyzing B3DB Dataset - Mechanism Distribution")
    print("="*80)

    # Load B3DB data
    print("\n1. Loading B3DB dataset...")
    b3db_file = Paths.root / "outputs" / "b3db_analysis" / "b3db_with_features.csv"
    if not b3db_file.exists():
        print(f"   Error: {b3db_file} not found. Please run explore_b3db_mechanisms.py first.")
        return None

    df = pd.read_csv(b3db_file)
    print(f"   Loaded {len(df)} molecules")

    # Initialize predictor
    print("\n2. Loading integrated predictor...")
    predictor = IntegratedMechanismPredictor()

    # Predict mechanisms for all molecules
    print("\n3. Predicting mechanisms for all molecules...")
    print("   (This may take a while...)")

    results = []
    for idx, row in df.head(1000).iterrows():  # Start with first 1000 for speed
        if idx % 100 == 0:
            print(f"   Processing {idx}/1000...")

        smiles = row['SMILES']
        try:
            result = predictor.predict_mechanisms(smiles)

            results.append({
                'SMILES': smiles,
                'BBB_binary': row['BBB_binary'],
                'group': row['group'],
                'BBB_prob': result['BBB']['probability'],
                'Passive_prob': result['Passive_Diffusion']['probability'],
                'Influx_prob': result['Active_Influx']['probability'],
                'Efflux_prob': result['Active_Efflux']['probability'],
                'Primary_mechanism': result['mechanism_summary']['primary_mechanism'],
                'Certainty': result['mechanism_summary']['certainty'],
                'MW': result['properties']['MW'],
                'TPSA': result['properties']['TPSA'],
                'LogP': result['properties']['LogP'],
            })
        except Exception as e:
            print(f"   Error processing {smiles}: {e}")

    results_df = pd.DataFrame(results)
    print(f"   Successfully predicted {len(results_df)} molecules")

    # Save results
    output_dir = Paths.root / "outputs" / "b3db_mechanism_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "b3db_mechanism_predictions.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n   Saved to: {output_file}")

    return results_df


def create_mechanism_visualization(df):
    """Create visualization of mechanism distribution."""
    print("\n4. Creating visualizations...")

    output_dir = Paths.root / "outputs" / "b3db_mechanism_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mechanism distribution pie chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Overall mechanism distribution
    mech_counts = df['Primary_mechanism'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    axes[0, 0].pie(mech_counts.values, labels=mech_counts.index, autopct='%1.1f%%',
                    colors=colors, startangle=90)
    axes[0, 0].set_title('B3DB Dataset - Overall Mechanism Distribution\n(n=1000)',
                        fontsize=14, fontweight='bold')

    # Mechanism distribution by BBB status
    bbb_plus = df[df['BBB_binary'] == 1]
    bbb_minus = df[df['BBB_binary'] == 0]

    mech_bbb_plus = bbb_plus['Primary_mechanism'].value_counts()
    mech_bbb_minus = bbb_minus['Primary_mechanism'].value_counts()

    x = np.arange(len(mech_bbb_plus.index))
    width = 0.35

    axes[0, 1].bar(x - width/2, mech_bbb_plus.values, width, label='BBB+', color='#66b3ff')
    axes[0, 1].bar(x + width/2, mech_bbb_minus[:len(mech_bbb_plus)].values, width, label='BBB-', color='#ff9999')
    axes[0, 1].set_xlabel('Mechanism', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].set_title('Mechanism Distribution by BBB Status', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(mech_bbb_plus.index, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Probability distributions
    axes[1, 0].hist(df['Passive_prob'], bins=30, alpha=0.7, label='Passive Diffusion', color='blue')
    axes[1, 0].hist(df['Influx_prob'], bins=30, alpha=0.7, label='Active Influx', color='green')
    axes[1, 0].hist(df['Efflux_prob'], bins=30, alpha=0.7, label='Active Efflux', color='red')
    axes[1, 0].set_xlabel('Probability', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Mechanism Probability Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Certainty levels
    certainty_counts = df['Certainty'].value_counts()
    colors_certainty = {'High': '#66b3ff', 'Medium': '#ffcc99', 'Low': '#ff9999'}

    axes[1, 1].bar(certainty_counts.index, certainty_counts.values,
                   color=[colors_certainty.get(x, '#cccccc') for x in certainty_counts.index])
    axes[1, 1].set_xlabel('Certainty Level', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('Prediction Certainty Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_file = output_dir / "b3db_mechanism_distribution.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"   Saved: {fig_file}")
    plt.close()

    # 2. Property vs Mechanism scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # TPSA vs Mechanism
    for mech in df['Primary_mechanism'].unique():
        data = df[df['Primary_mechanism'] == mech]
        axes[0].scatter(data['TPSA'], data['MW'], label=mech, alpha=0.6, s=30)

    axes[0].set_xlabel('TPSA (A²)', fontsize=12)
    axes[0].set_ylabel('MW (Da)', fontsize=12)
    axes[0].set_title('MW vs TPSA by Mechanism', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # LogP vs Mechanism
    for mech in df['Primary_mechanism'].unique():
        data = df[df['Primary_mechanism'] == mech]
        axes[1].scatter(data['LogP'], data['TPSA'], label=mech, alpha=0.6, s=30)

    axes[1].set_xlabel('LogP', fontsize=12)
    axes[1].set_ylabel('TPSA (A²)', fontsize=12)
    axes[1].set_title('TPSA vs LogP by Mechanism', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # BBB probability vs Mechanism probability
    scatter = axes[2].scatter(df['Passive_prob'], df['BBB_prob'],
                              c=df['BBB_binary'], cmap='RdYlGn', alpha=0.6, s=30)
    axes[2].set_xlabel('Passive Diffusion Probability', fontsize=12)
    axes[2].set_ylabel('BBB+ Probability', fontsize=12)
    axes[2].set_title('BBB vs Passive Diffusion Probability', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=axes[2], label='BBB Status')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig_file = output_dir / "b3db_property_vs_mechanism.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"   Saved: {fig_file}")
    plt.close()

    # 3. Summary statistics
    print("\n5. Mechanism Distribution Statistics:")
    print("-" * 80)

    print("\nOverall Mechanism Distribution:")
    for mech, count in df['Primary_mechanism'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {mech:25s}: {count:4d} ({pct:5.1f}%)")

    print("\nMechanism by BBB Status:")
    print(f"{'Mechanism':<25} {'BBB+':>10} {'BBB-':>10}")
    print("-" * 50)

    all_mechs = set(df['Primary_mechanism'].unique())
    for mech in sorted(all_mechs):
        bbb_plus_count = len(df[(df['Primary_mechanism'] == mech) & (df['BBB_binary'] == 1)])
        bbb_minus_count = len(df[(df['Primary_mechanism'] == mech) & (df['BBB_binary'] == 0)])
        print(f"{mech:<25} {bbb_plus_count:>10} {bbb_minus_count:>10}")

    print("\nAverage Probabilities by Primary Mechanism:")
    print(f"{'Mechanism':<25} {'BBB+':>10} {'Passive':>10} {'Influx':>10} {'Efflux':>10}")
    print("-" * 70)

    for mech in sorted(all_mechs):
        data = df[df['Primary_mechanism'] == mech]
        print(f"{mech:<25} {data['BBB_prob'].mean():>10.2%} "
              f"{data['Passive_prob'].mean():>10.2%} {data['Influx_prob'].mean():>10.2%} "
              f"{data['Efflux_prob'].mean():>10.2%}")

    return df


def main():
    """Main analysis function."""
    print("="*80)
    print("B3DB Mechanism Distribution Analysis")
    print("="*80)

    # Analyze B3DB
    df = analyze_b3db_mechanisms()

    if df is not None:
        # Create visualizations
        create_mechanism_visualization(df)

        print("\n" + "="*80)
        print("Analysis Complete!")
        print("="*80)
        print("\nGenerated files:")
        print("  - outputs/b3db_mechanism_analysis/b3db_mechanism_predictions.csv")
        print("  - outputs/b3db_mechanism_analysis/b3db_mechanism_distribution.png")
        print("  - outputs/b3db_mechanism_analysis/b3db_property_vs_mechanism.png")


if __name__ == "__main__":
    main()
