"""
Predict mechanisms for proof1 molecules.

Proof1 molecules:
- L-DOPA (Parkinson's disease treatment)
- Gabapentin (Antiepileptic)
- Pregabalin (Antiepileptic)
- Glucose (Energy source)
- Lactic acid (Metabolite)
- Beta-hydroxybutyrate (Ketone body)
- Caffeine (Stimulant)
- Nicotine (Stimulant)
- Loperamide (Antidiarrheal - does NOT cross BBB)
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


# Proof1 molecules with SMILES
PROOF1_MOLECULES = {
    'L-DOPA': {
        'smiles': 'O=C(O)[C@@H](c1cc(O)c(O)cc1)N',
        'description': 'Levodopa - Parkinson\'s disease treatment, crosses BBB via LAT1 transporter',
        'expected_bbb': True,
        'expected_mechanism': 'Active_Influx',
    },
    'Gabapentin': {
        'smiles': 'CC(C(=O)N)C[C@@H]1CCCO1',
        'description': 'Antiepileptic - crosses BBB via LAT1 transporter',
        'expected_bbb': True,
        'expected_mechanism': 'Active_Influx',
    },
    'Pregabalin': {
        'smiles': 'CC(C(=O)N)C[C@@H]1CCCO1',
        'description': 'Antiepileptic - similar to gabapentin, uses LAT1',
        'expected_bbb': True,
        'expected_mechanism': 'Active_Influx',
    },
    'Glucose': {
        'smiles': 'OCC1OC(O)C(O)C(O)C1O',
        'description': 'Energy source - crosses BBB via GLUT1 transporter',
        'expected_bbb': True,
        'expected_mechanism': 'Active_Influx',
    },
    'Lactic Acid': {
        'smiles': 'CC(=O)O',
        'description': 'Metabolite - crosses BBB via MCT transporters',
        'expected_bbb': True,
        'expected_mechanism': 'Active_Influx',
    },
    'Beta-Hydroxybutyrate': {
        'smiles': 'CCC(=O)CO',
        'description': 'Ketone body - crosses BBB via MCT transporters',
        'expected_bbb': True,
        'expected_mechanism': 'Active_Influx',
    },
    'Caffeine': {
        'smiles': 'Cn1cnc2c1c(=O)n(C)c(=O)n2C',
        'description': 'Stimulant - crosses BBB via passive diffusion',
        'expected_bbb': True,
        'expected_mechanism': 'Passive_Diffusion',
    },
    'Nicotine': {
        'smiles': 'CN1CCCC1c1cccnc1',
        'description': 'Stimulant - crosses BBB via passive diffusion',
        'expected_bbb': True,
        'expected_mechanism': 'Passive_Diffusion',
    },
    'Loperamide': {
        'smiles': 'CC(C)(C)NC(=O)C1CN(C)CCC1c2ccc(C(=O)O)cc2',
        'description': 'Antidiarrheal - P-gp substrate, does NOT cross BBB',
        'expected_bbb': False,
        'expected_mechanism': 'Active_Efflux',
    },
}


def predict_proof1_molecules():
    """Predict mechanisms for all proof1 molecules."""
    print("="*80)
    print("Predicting Mechanisms for Proof1 Molecules")
    print("="*80)

    # Initialize predictor
    print("\nLoading integrated predictor...")
    predictor = IntegratedMechanismPredictor()

    # Predict for each molecule
    print("\nPredicting mechanisms...")
    print("-" * 80)

    results = []

    for name, mol_data in PROOF1_MOLECULES.items():
        print(f"\n{name}:")
        print(f"  Description: {mol_data['description']}")
        print(f"  Expected: BBB+={mol_data['expected_bbb']}, Mechanism={mol_data['expected_mechanism']}")

        try:
            result = predictor.predict_mechanisms(mol_data['smiles'])

            # Store results
            results.append({
                'Name': name,
                'SMILES': mol_data['smiles'],
                'Description': mol_data['description'],
                'Expected_BBB': mol_data['expected_bbb'],
                'Expected_Mechanism': mol_data['expected_mechanism'],
                'Predicted_BBB': result['BBB']['prediction'],
                'BBB_Probability': result['BBB']['probability'],
                'BBB_Confidence': result['BBB']['confidence'],
                'Predicted_Mechanism': result['mechanism_summary']['primary_mechanism'],
                'Mechanism_Certainty': result['mechanism_summary']['certainty'],
                'Passive_Prob': result['Passive_Diffusion']['probability'],
                'Influx_Prob': result['Active_Influx']['probability'],
                'Efflux_Prob': result['Active_Efflux']['probability'],
                'MW': result['properties']['MW'],
                'TPSA': result['properties']['TPSA'],
                'LogP': result['properties']['LogP'],
                'HBA': result['properties']['HBA'],
                'HBD': result['properties']['HBD'],
            })

            # Print predictions
            bbb_pred = "+" if result['BBB']['prediction'] else "-"
            print(f"  Predicted: BBB{bbb_pred} ({result['BBB']['probability']:.2%}, {result['BBB']['confidence']})")

            primary_mech = result['mechanism_summary']['primary_mechanism']
            certainty = result['mechanism_summary']['certainty']
            print(f"  Primary Mechanism: {primary_mech} ({certainty})")

            print(f"  Mechanism Probabilities:")
            print(f"    Passive Diffusion: {result['Passive_Diffusion']['probability']:.2%}")
            print(f"    Active Influx:    {result['Active_Influx']['probability']:.2%}")
            print(f"    Active Efflux:    {result['Active_Efflux']['probability']:.2%}")

            # Check if prediction matches expectation
            bbb_match = "✓" if result['BBB']['prediction'] == mol_data['expected_bbb'] else "✗"
            mech_match = "✓" if primary_mech.replace('_', ' ') == mol_data['expected_mechanism'].replace('_', ' ') else "?"

            print(f"  Match with Expected: BBB={bbb_match}, Mechanism={mech_match}")

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'Name': name,
                'SMILES': mol_data['smiles'],
                'Error': str(e)
            })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    output_dir = Paths.root / "outputs" / "proof1_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "proof1_predictions.csv"
    df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")

    return df


def create_proof1_visualization(df):
    """Create visualization for proof1 molecules."""
    print("\nCreating visualizations...")

    output_dir = Paths.root / "outputs" / "proof1_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Comparison: Expected vs Predicted
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # BBB prediction comparison
    x = np.arange(len(df))
    width = 0.35

    axes[0, 0].bar(x - width/2, df['Expected_BBB'].astype(int), width, label='Expected', color='gray')
    axes[0, 0].bar(x + width/2, df['Predicted_BBB'].astype(int), width, label='Predicted', color='#66b3ff')
    axes[0, 0].set_xlabel('Molecule', fontsize=12)
    axes[0, 0].set_ylabel('BBB+ (1) / BBB- (0)', fontsize=12)
    axes[0, 0].set_title('BBB Permeability: Expected vs Predicted', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(df['Name'], rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Mechanism probability comparison
    mechanisms = df['Expected_Mechanism'].unique()
    x_pos = np.arange(len(df))
    width = 0.25

    axes[0, 1].bar(x_pos - width, df['Passive_Prob'], width, label='Passive', color='blue', alpha=0.7)
    axes[0, 1].bar(x_pos, df['Influx_Prob'], width, label='Influx', color='green', alpha=0.7)
    axes[0, 1].bar(x_pos + width, df['Efflux_Prob'], width, label='Efflux', color='red', alpha=0.7)

    # Mark expected mechanisms
    for i, (idx, row) in enumerate(df.iterrows()):
        if row['Expected_Mechanism'] == 'Passive_Diffusion':
            axes[0, 1].scatter(i, row['Passive_Prob'], s=200, marker='*', color='black', zorder=5)
        elif row['Expected_Mechanism'] == 'Active_Influx':
            axes[0, 1].scatter(i, row['Influx_Prob'], s=200, marker='*', color='black', zorder=5)
        elif row['Expected_Mechanism'] == 'Active_Efflux':
            axes[0, 1].scatter(i, row['Efflux_Prob'], s=200, marker='*', color='black', zorder=5)

    axes[0, 1].set_xlabel('Molecule', fontsize=12)
    axes[0, 1].set_ylabel('Probability', fontsize=12)
    axes[0, 1].set_title('Mechanism Probabilities (* = Expected)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(df['Name'], rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Property space: TPSA vs MW
    colors = {'Passive_Diffusion': 'blue', 'Active_Influx': 'green', 'Active_Efflux': 'red',
              'Mixed/Uncertain': 'gray'}

    for expected_mech in mechanisms:
        data = df[df['Expected_Mechanism'] == expected_mech]
        axes[1, 0].scatter(data['TPSA'], data['MW'], label=expected_mech, s=200, alpha=0.7)

        # Label points
        for idx, row in data.iterrows():
            axes[1, 0].annotate(row['Name'], (row['TPSA'], row['MW']),
                              fontsize=8, ha='center', va='bottom')

    axes[1, 0].set_xlabel('TPSA (A²)', fontsize=12)
    axes[1, 0].set_ylabel('MW (Da)', fontsize=12)
    axes[1, 0].set_title('Property Space: Expected Mechanisms', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # BBB probability bar chart
    bars = axes[1, 1].barh(df['Name'], df['BBB_Probability'],
                           color=['green' if x > 0.5 else 'red' for x in df['BBB_Probability']])
    axes[1, 1].set_xlabel('BBB+ Probability', fontsize=12)
    axes[1, 1].set_title('BBB Permeability Predictions', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].grid(axis='x', alpha=0.3)

    # Add probability values
    for i, (idx, row) in enumerate(df.iterrows()):
        axes[1, 1].text(row['BBB_Probability'], i, f" {row['BBB_Probability']:.2%}",
                       va='center', fontsize=10)

    plt.tight_layout()
    fig_file = output_dir / "proof1_predictions_visualization.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_file}")
    plt.close()

    # 2. Detailed comparison table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    for idx, row in df.iterrows():
        table_data.append([
            row['Name'],
            f"{row['BBB_Probability']:.1%}",
            row['BBB_Confidence'],
            f"{row['Passive_Prob']:.1%}",
            f"{row['Influx_Prob']:.1%}",
            f"{row['Efflux_Prob']:.1%}",
            row['Predicted_Mechanism'],
            row['Mechanism_Certainty'],
        ])

    # Add header
    columns = ['Molecule', 'BBB+ Prob', 'BBB Conf', 'Passive', 'Influx', 'Efflux', 'Predicted', 'Certainty']

    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows based on BBB prediction
    for i in range(len(df)):
        if df.iloc[i]['Predicted_BBB']:
            table[(i + 1, 1)].set_facecolor('#C6EFCE')
        else:
            table[(i + 1, 1)].set_facecolor('#FFC7CE')

    plt.title('Proof1 Molecules - Detailed Prediction Results',
             fontsize=16, fontweight='bold', pad=20)

    fig_file = output_dir / "proof1_predictions_table.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_file}")
    plt.close()


def main():
    """Main analysis function."""
    print("="*80)
    print("Proof1 Molecules - Mechanism Prediction")
    print("="*80)

    # Predict mechanisms
    df = predict_proof1_molecules()

    # Create visualizations
    if df is not None and 'Error' not in df.columns:
        create_proof1_visualization(df)

    print("\n" + "="*80)
    print("Proof1 Analysis Complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - outputs/proof1_analysis/proof1_predictions.csv")
    print("  - outputs/proof1_analysis/proof1_predictions_visualization.png")
    print("  - outputs/proof1_analysis/proof1_predictions_table.png")


if __name__ == "__main__":
    main()
