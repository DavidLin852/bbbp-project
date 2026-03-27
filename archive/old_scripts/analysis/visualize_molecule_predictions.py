"""
Molecule Prediction Probability Visualization
Name vs Prediction Probability Bar Chart
Font: Times New Roman
"""

import sys
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).parent.parent


def create_prediction_bar_chart():
    """Create bar chart for molecule predictions"""

    # Data from user - keep original order
    data = {
        'Name': [
            'Betaine', 'DMACA', 'GABA', 'GLUT', 'Tryptamine', 'taurine',
            'MPC', 'CBMA', 'CBMA-2', 'CBMA-3', 'SBMA', 'SBMA-2', 'MPSMA',
            'DMAEMA', 'ONMA', 'DSC6MA', 'TryC3MA', 'DMACAC5MA', 'GLUTC4MA'
        ],
        'SMILES': [
            'C[N+](C)(C)CC(=O)[O-]',
            'CN(C)c1ccc(/C=C/C(=O)O)cc1',
            'NCCCC(=O)O',
            'OCC1OC(O)C(O)C(O)C1O',
            'NCCc1c[nH]c2ccccc12',
            'NCCS(=O)(=O)[O-]',
            'C=C(C)C(=O)OCCOP(=O)([O-])OCC[N+](C)(C)C',
            'C=C(C)C(O)OCC[N+](C)(C)CC(=O)[O-]',
            'C=C(C)C(=O)OCC[N+](C)(C)CCC(=O)[O-]',
            'C=C(C)C(=O)OCC[N+](C)(C)CCCC(=O)[O-]',
            'C=C(C)C(=O)OCC[N+](C)(C)CCCS(=O)(=O)[O-]',
            'C=C(C)C(=O)OCC[N+](C)(C)CCS(=O)(=O)[O-]',
            'C=C(C)C(=O)OCC(O)C[N+]1(C)CCC(S(=O)(=O)[O-])CC1',
            'C=C(C)C(=O)OCCN(C)C',
            'C=C(C)C(=O)OCC[N+](C)(C)[O-]',
            'C=C(C)C(=O)NCCCCCCNC(=O)[C@H](N)CO',
            'C=C(C)C(=O)NCCCC(=O)NCCc1c[nH]c2ccccc12',
            'C=C(C)C(=O)NCCCCCCNC(=O)/C=C/c1ccc(N(C)C)cc1',
            'C=C(C)C(=O)NCCCCNC(=O)OCC1OC(O)C(O)C(O)C1O'
        ],
        'Probability': [
            0.9599, 0.5598, 0.9853, 0.5415, 0.9337, 0.9912,
            0.8928, 0.8562, 0.8810, 0.8762, 0.8909, 0.6716, 0.7614,
            0.9987, 0.9526, 0.8653, 0.9123, 0.4950, 0.6694
        ]
    }

    df = pd.DataFrame(data)

    # Find MPC index
    mpc_idx = df[df['Name'] == 'MPC'].index[0]

    # Create colors: before MPC (blue), from MPC onwards (red)
    colors = []
    for i in range(len(df)):
        if i < mpc_idx:
            colors.append('#3498DB')  # Blue - before MPC
        else:
            colors.append('#E74C3C')  # Red - from MPC onwards

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create horizontal bar chart
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['Probability'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add probability values on bars
    for i, (bar, prob) in enumerate(zip(bars, df['Probability'])):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob:.4f}',
                ha='left', va='center',
                fontname='Times New Roman', fontsize=10, fontweight='bold')

    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Name'], fontname='Times New Roman', fontsize=11)

    # Set labels and title
    ax.set_xlabel('BBB+ Probability', fontname='Times New Roman', fontsize=13, fontweight='bold')
    ax.set_ylabel('Molecule Name', fontname='Times New Roman', fontsize=13, fontweight='bold')
    ax.set_title('BBB Permeability Prediction - 19 Molecules\n(Stacking_XGB + Combined Features)',
                 fontname='Times New Roman', fontsize=15, fontweight='bold', pad=15)

    # Set x-axis range
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                       fontname='Times New Roman', fontsize=11)

    # Add grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # Save figure
    output_dir = PROJECT_ROOT / "outputs" / "images" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "molecule_predictions_bar_chart.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Bar chart saved: {output_file}")

    return fig, df


def main():
    """Main function"""

    print("=" * 70)
    print("Creating Molecule Prediction Probability Bar Chart")
    print("Font: Times New Roman")
    print("=" * 70)

    create_prediction_bar_chart()

    print("\nChart generated successfully!")
    print(f"\nOutput file: outputs/images/analysis/molecule_predictions_bar_chart.png")


if __name__ == "__main__":
    main()
