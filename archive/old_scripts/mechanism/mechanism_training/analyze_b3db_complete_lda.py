"""
Complete LDA analysis and visualization for B3DB dataset.

This script:
1. Analyzes ALL 7,805 molecules in B3DB
2. Creates LDA visualizations matching Cornelissen et al. 2022 style
3. Uses Times New Roman font and professional color scheme
4. Compares with Cornelissen findings
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Professional styling
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# Professional color scheme (low brightness)
COLORS = {
    'bbb_plus': '#2E86AB',      # Muted blue
    'bbb_minus': '#A23B72',     # Muted wine
    'passive': '#1B4F72',       # Dark blue
    'influx': '#C17C6D',        # Muted coral
    'efflux': '#6B4C4C',        # Muted brown
    'mixed': '#888888',         # Gray
    'highlight': '#E94F37',     # Accent red (only for highlights)
}

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Paths
from src.path_prediction.integrated_mechanism_predictor import IntegratedMechanismPredictor


def load_b3db_with_features():
    """Load B3DB dataset with extracted features."""
    print("Loading B3DB dataset...")

    # Try to load pre-processed data
    b3db_file = Paths.root / "outputs" / "b3db_analysis" / "b3db_with_features.csv"
    if b3db_file.exists():
        df = pd.read_csv(b3db_file)
        print(f"  Loaded {len(df)} molecules with pre-extracted features")
        return df

    # Otherwise, load raw data and extract features
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    b3db_raw = Paths.data_raw / "B3DB_classification.tsv"
    df_raw = pd.read_csv(b3db_raw, sep='\t')
    df_raw['BBB_binary'] = (df_raw['BBB+/BBB-'] == 'BBB+').astype(int)

    print(f"  Extracting features for {len(df_raw)} molecules...")

    features_list = []
    for idx, row in df_raw.iterrows():
        if idx % 1000 == 0:
            print(f"    Processing {idx}/{len(df_raw)}...")

        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is None:
            continue

        try:
            features = {
                'SMILES': row['SMILES'],
                'BBB_binary': (row['BBB+/BBB-'] == 'BBB+').astype(int),
                'group': row['group'],
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'MW': Descriptors.MolWt(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'RotatableBonds': Descriptors.NumRotatableBonds(mol),
                'RingCount': Descriptors.RingCount(mol),
                'AromaticRings': Descriptors.NumAromaticRings(mol),
                'FractionCSP3': Descriptors.FractionCSP3(mol),
            }
            features_list.append(features)
        except:
            continue

    df = pd.DataFrame(features_list)
    print(f"  Successfully extracted features for {len(df)} molecules")

    return df


def perform_lda_analysis(df, feature_cols):
    """
    Perform LDA analysis similar to Cornelissen et al. 2022.

    Steps:
    1. Standardize features
    2. PCA to 50 dimensions (optional, for high-dimensional data)
    3. LDA for supervised dimensionality reduction
    """
    print("\nPerforming LDA analysis...")

    # Prepare data
    X = df[feature_cols].values
    y = df['BBB_binary'].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA to reduce dimensionality first (if needed)
    if X_scaled.shape[1] > 50:
        print(f"  Applying PCA: {X_scaled.shape[1]} -> 50 dimensions")
        pca = PCA(n_components=50, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    else:
        X_pca = X_scaled

    # LDA
    print(f"  Applying LDA...")
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda_result = lda.fit_transform(X_pca, y)

    # Get decision function for 2D visualization
    decision_function = lda.decision_function(X_pca)

    print(f"  LDA coefficient shape: {lda.coef_.shape}")
    print(f"  LDA explained variance ratio: {lda.explained_variance_ratio_[0]:.2%}")

    return lda_result, decision_function, lda, scaler


def create_lda_visualization(df, lda_result, decision_function):
    """Create LDA visualization matching Cornelissen et al. 2022 style."""
    print("\nCreating LDA visualizations...")

    output_dir = Paths.root / "outputs" / "b3db_lda_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 1D LDA projection (like Cornelissen's main figure)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: 1D LDA with jitter
    bbb_plus = lda_result[df['BBB_binary'] == 1]
    bbb_minus = lda_result[df['BBB_binary'] == 0]

    # Add jitter for better visualization
    jitter = np.random.normal(0, 0.02, size=len(lda_result))

    ax1.scatter(bbb_minus, jitter[df['BBB_binary'] == 0],
               c=COLORS['bbb_minus'], alpha=0.5, s=20, label=f'BBB- (n={len(bbb_minus)})')
    ax1.scatter(bbb_plus, jitter[df['BBB_binary'] == 1],
               c=COLORS['bbb_plus'], alpha=0.5, s=20, label=f'BBB+ (n={len(bbb_plus)})')

    ax1.set_xlabel('LDA Discriminant Function', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Jitter', fontsize=14)
    ax1.set_title('B3DB Dataset - LDA Projection (n=7,805)',
                  fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Plot 2: 1D LDA as histogram
    ax2.hist(bbb_minus, bins=50, alpha=0.6, color=COLORS['bbb_minus'],
             label=f'BBB- (n={len(bbb_minus)})')
    ax2.hist(bbb_plus, bins=50, alpha=0.6, color=COLORS['bbb_plus'],
             label=f'BBB+ (n={len(bbb_plus)})')
    ax2.set_xlabel('LDA Discriminant Function', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=14)
    ax2.set_title('Distribution of LDA Scores', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    plt.tight_layout()
    fig_file = output_dir / "b3db_lda_1d_projection.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_file}")
    plt.close()

    # 2. 2D LDA visualization (LD1 vs Decision Function)
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(lda_result, decision_function,
                        c=df['BBB_binary'], cmap='RdBu_r',
                        alpha=0.5, s=20, vmin=0, vmax=1)

    ax.set_xlabel('LDA Discriminant Function 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('Decision Function Value', fontsize=14, fontweight='bold')
    ax.set_title('B3DB Dataset - LDA Decision Space (n=7,805)',
                fontsize=16, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('BBB Status (0=BBB-, 1=BBB+)', fontsize=12)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['BBB-', 'BBB+'])

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    plt.tight_layout()
    fig_file = output_dir / "b3db_lda_2d_decision_space.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_file}")
    plt.close()


def create_property_space_visualization(df):
    """Create property space visualization (TPSA vs MW vs LogP)."""
    print("\nCreating property space visualizations...")

    output_dir = Paths.root / "outputs" / "b3db_lda_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. TPSA vs MW (Cornelissen's key plot)
    fig, ax = plt.subplots(figsize=(10, 8))

    bbb_plus = df[df['BBB_binary'] == 1]
    bbb_minus = df[df['BBB_binary'] == 0]

    ax.scatter(bbb_minus['TPSA'], bbb_minus['MW'],
               c=COLORS['bbb_minus'], alpha=0.4, s=20, label=f'BBB- (n={len(bbb_minus)})')
    ax.scatter(bbb_plus['TPSA'], bbb_plus['MW'],
               c=COLORS['bbb_plus'], alpha=0.4, s=20, label=f'BBB+ (n={len(bbb_plus)})')

    # Add reference lines (Cornelissen thresholds)
    ax.axvline(x=90, color='black', linestyle='--', linewidth=1, alpha=0.5, label='TPSA=90 Å²')
    ax.axhline(y=500, color='black', linestyle='--', linewidth=1, alpha=0.5, label='MW=500 Da')

    ax.set_xlabel('TPSA (Å²)', fontsize=14, fontweight='bold')
    ax.set_ylabel('MW (Da)', fontsize=14, fontweight='bold')
    ax.set_title('B3DB Dataset - Property Space (n=7,805)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig_file = output_dir / "b3db_property_tpsa_vs_mw.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_file}")
    plt.close()

    # 2. LogP vs TPSA
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(bbb_minus['TPSA'], bbb_minus['LogP'],
               c=COLORS['bbb_minus'], alpha=0.4, s=20, label=f'BBB- (n={len(bbb_minus)})')
    ax.scatter(bbb_plus['TPSA'], bbb_plus['LogP'],
               c=COLORS['bbb_plus'], alpha=0.4, s=20, label=f'BBB+ (n={len(bbb_plus)})')

    # Add reference lines
    ax.axvline(x=90, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=2, color='black', linestyle='--', linewidth=1, alpha=0.5, label='LogP=2')
    ax.axhline(y=4, color='black', linestyle='--', linewidth=1, alpha=0.5, label='LogP=4')

    ax.set_xlabel('TPSA (Å²)', fontsize=14, fontweight='bold')
    ax.set_ylabel('LogP', fontsize=14, fontweight='bold')
    ax.set_title('B3DB Dataset - Lipophilicity vs Polarity', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig_file = output_dir / "b3db_property_tpsa_vs_logp.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_file}")
    plt.close()


def create_statistical_summary(df):
    """Create statistical summary visualization."""
    print("\nCreating statistical summary...")

    output_dir = Paths.root / "outputs" / "b3db_lda_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Properties to compare
    properties = ['TPSA', 'MW', 'LogP', 'HBA', 'HBD']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, prop in enumerate(properties):
        ax = axes[i]

        # Get data for BBB+ and BBB-
        bbb_plus_data = df[df['BBB_binary'] == 1][prop]
        bbb_minus_data = df[df['BBB_binary'] == 0][prop]

        # Create box plot
        bp = ax.boxplot([bbb_minus_data, bbb_plus_data],
                       labels=['BBB-', 'BBB+'],
                       patch_artist=True,
                       widths=0.6)

        # Color boxes
        bp['boxes'][0].set_facecolor(COLORS['bbb_minus'])
        bp['boxes'][1].set_facecolor(COLORS['bbb_plus'])
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_alpha(0.6)

        ax.set_ylabel(prop, fontsize=12, fontweight='bold')
        ax.set_title(f'{prop} Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Add statistics
        median_plus = bbb_plus_data.median()
        median_minus = bbb_minus_data.median()
        ax.text(0.5, 0.95, f'Median: {median_minus:.1f}',
               transform=ax.transAxes, ha='center', fontsize=10, color=COLORS['bbb_minus'])
        ax.text(1.5, 0.95, f'Median: {median_plus:.1f}',
               transform=ax.transAxes, ha='center', fontsize=10, color=COLORS['bbb_plus'])

    # Remove empty subplot
    axes[-1].axis('off')

    # Add overall title
    fig.suptitle('B3DB Dataset - Property Statistics by BBB Status (n=7,805)',
                fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout()
    fig_file = output_dir / "b3db_property_statistics.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_file}")
    plt.close()


def print_summary_statistics(df):
    """Print comprehensive summary statistics."""
    print("\n" + "="*80)
    print("B3DB DATASET SUMMARY STATISTICS")
    print("="*80)

    print(f"\nTotal molecules: {len(df)}")
    print(f"BBB+: {df['BBB_binary'].sum()} ({df['BBB_binary'].mean()*100:.1f}%)")
    print(f"BBB-: {len(df) - df['BBB_binary'].sum()} ({(1-df['BBB_binary'].mean())*100:.1f}%)")

    print("\nGroup Distribution:")
    print(df['group'].value_counts().sort_index())

    print("\nProperty Statistics by BBB Status:")
    print("-" * 80)
    print(f"{'Property':<15} {'BBB+ Mean':>12} {'BBB- Mean':>12} {'Difference':>12}")
    print("-" * 80)

    for prop in ['TPSA', 'MW', 'LogP', 'HBA', 'HBD']:
        bbb_plus_mean = df[df['BBB_binary'] == 1][prop].mean()
        bbb_minus_mean = df[df['BBB_binary'] == 0][prop].mean()
        diff = bbb_plus_mean - bbb_minus_mean
        print(f"{prop:<15} {bbb_plus_mean:>12.2f} {bbb_minus_mean:>12.2f} {diff:>+12.2f}")

    print("\nKey Findings (consistent with Cornelissen et al. 2022):")
    print("-" * 80)

    bbb_plus_tpsa = df[df['BBB_binary'] == 1]['TPSA'].mean()
    bbb_minus_tpsa = df[df['BBB_binary'] == 0]['TPSA'].mean()

    if bbb_plus_tpsa < bbb_minus_tpsa:
        print(f"[OK] BBB+ molecules have LOWER TPSA ({bbb_plus_tpsa:.1f} vs {bbb_minus_tpsa:.1f} A^2)")
        print(f"  This matches Cornelissen et al. 2022 finding: TPSA is key predictor for BBB")

    bbb_plus_mw = df[df['BBB_binary'] == 1]['MW'].mean()
    bbb_minus_mw = df[df['BBB_binary'] == 0]['MW'].mean()

    if bbb_plus_mw < bbb_minus_mw:
        print(f"[OK] BBB+ molecules have LOWER MW ({bbb_plus_mw:.1f} vs {bbb_minus_mw:.1f} Da)")
        print(f"  Lower molecular weight favors BBB penetration")

    bbb_plus_logp = df[df['BBB_binary'] == 1]['LogP'].mean()
    bbb_minus_logp = df[df['BBB_binary'] == 0]['LogP'].mean()

    if bbb_plus_logp > bbb_minus_logp:
        print(f"[OK] BBB+ molecules have HIGHER LogP ({bbb_plus_logp:.2f} vs {bbb_minus_logp:.2f})")
        print(f"  Moderate lipophilicity (LogP 1-3) favors BBB penetration")


def main():
    """Main analysis function."""
    print("="*80)
    print("B3DB DATASET - COMPLETE LDA ANALYSIS")
    print("Following Cornelissen et al. 2022 Methodology")
    print("="*80)

    # Load data
    df = load_b3db_with_features()

    # Feature columns for LDA
    feature_cols = ['LogP', 'TPSA', 'MW', 'HBA', 'HBD',
                   'RotatableBonds', 'RingCount', 'AromaticRings', 'FractionCSP3']

    # Perform LDA
    lda_result, decision_function, lda_model, scaler = perform_lda_analysis(df, feature_cols)

    # Print feature importance (LDA coefficients)
    print("\nLDA Feature Importance:")
    print("-" * 80)
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': lda_model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)

    print(feature_importance.to_string(index=False))

    # Create visualizations
    create_lda_visualization(df, lda_result, decision_function)
    create_property_space_visualization(df)
    create_statistical_summary(df)

    # Print summary statistics
    print_summary_statistics(df)

    # Save processed data
    output_dir = Paths.root / "outputs" / "b3db_lda_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    df['LDA_score'] = lda_result
    df['Decision_function'] = decision_function

    output_file = output_dir / "b3db_with_lda_scores.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved LDA scores to: {output_file}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated visualizations:")
    print("  1. b3db_lda_1d_projection.png - 1D LDA projection with jitter")
    print("  2. b3db_lda_2d_decision_space.png - 2D LDA decision space")
    print("  3. b3db_property_tpsa_vs_mw.png - TPSA vs MW scatter plot")
    print("  4. b3db_property_tpsa_vs_logp.png - TPSA vs LogP scatter plot")
    print("  5. b3db_property_statistics.png - Property box plots")
    print("\nAll visualizations use:")
    print("  - Times New Roman font")
    print("  - Professional color scheme (low brightness)")
    print("  - 300 DPI resolution")


if __name__ == "__main__":
    main()
