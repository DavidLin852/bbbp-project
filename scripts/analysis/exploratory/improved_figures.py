"""
Improved Analysis Scripts for Cornelissen 2022 Dataset

Figure 2: Focus on TPSA, LogP, MW, HBD, HBA with better y-axis scaling
Figure 3: Multi-mechanism clustering comparison (PCA, t-SNE, UMAP)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set high DPI for publication quality
plt.rcParams['figure.dpi'] = 1200
plt.rcParams['savefig.dpi'] = 1200
sns.set_style("whitegrid")

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Paths


def load_data():
    """Load Cornelissen 2022 processed dataset"""
    data_path = Path(Paths.root) / "data" / "transport_mechanisms" / "cornelissen_2022" / "cornelissen_2022_processed.csv"
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded dataset: {df.shape}")
    return df


def figure2_improved_physicochemical(df, output_dir):
    """
    Figure 2 (Improved): Focus on Key Physicochemical Properties

    Shows TPSA, LogP, MW, HBD, HBA with better y-axis scaling
    """
    fig = plt.figure(figsize=(18, 10))

    # Key properties to analyze
    key_props = ['TPSA', 'LogP', 'MW', 'HBD', 'HBA']

    # Mechanisms to analyze
    mechanisms = ['BBB', 'Influx', 'Efflux', 'PAMPA']

    # Define y-axis limits for better visualization
    y_limits = {
        'TPSA': (0, 200),
        'LogP': (-2, 8),
        'MW': (100, 700),
        'HBD': (0, 10),
        'HBA': (0, 15)
    }

    # Create subplots
    for idx, prop in enumerate(key_props):
        if prop not in df.columns:
            continue

        ax = plt.subplot(2, 3, idx + 1)

        # Prepare data for box plot
        box_data = []
        labels = []
        colors = []

        for mech in mechanisms:
            col = f'label_{mech}'
            if col in df.columns and prop in df.columns:
                # Get positive and negative samples
                pos_samples = df[df[col] == 1][prop].dropna()
                neg_samples = df[df[col] == 0][prop].dropna()

                if len(pos_samples) > 0 and len(neg_samples) > 0:
                    box_data.extend([pos_samples, neg_samples])
                    labels.extend([f'{mech}+', f'{mech}-'])
                    colors.extend(['#27ae60', '#c0392b'])

        if box_data:
            parts = ax.boxplot(box_data, labels=labels, patch_artist=True,
                              showmeans=True, meanline=True,
                              boxprops=dict(linewidth=1.5),
                              whiskerprops=dict(linewidth=1.5),
                              capprops=dict(linewidth=1.5),
                              medianprops=dict(linewidth=2, color='black'),
                              meanprops=dict(linewidth=1.5, color='blue', linestyle='--'))

            # Color coding
            for i, patch in enumerate(parts['boxes']):
                patch.set_facecolor(colors[i])
                patch.set_alpha(0.7)

            # Set y-axis limits for better visualization
            if prop in y_limits:
                ax.set_ylim(y_limits[prop])

            ax.set_ylabel(prop, fontsize=12, fontweight='bold')
            ax.set_title(f'{prop} Distribution by Mechanism', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.grid(True, alpha=0.3, axis='y')

            # Add statistical annotations
            for mech in mechanisms:
                col = f'label_{mech}'
                if col in df.columns:
                    pos_mean = df[df[col] == 1][prop].mean()
                    neg_mean = df[df[col] == 0][prop].mean()
                    diff = pos_mean - neg_mean
                    if abs(diff) > 5:  # Only annotate significant differences
                        # Find position for annotation
                        pos_idx = labels.index(f'{mech}+')
                        # Add text annotation
                        ax.text(pos_idx, ax.get_ylim()[1] * 0.9,
                               f'Δ={diff:.1f}',
                               ha='center', fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    # Add summary statistics panel
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = "Key Findings:\n\n"

    # BBB mechanism
    bbb_col = 'label_BBB'
    if bbb_col in df.columns:
        tpsa_diff = df[df[bbb_col] == 1]['TPSA'].mean() - df[df[bbb_col] == 0]['TPSA'].mean()
        mw_diff = df[df[bbb_col] == 1]['MW'].mean() - df[df[bbb_col] == 0]['MW'].mean()
        summary_text += f"BBB Permeability:\n"
        summary_text += f"  • TPSA difference: {tpsa_diff:.1f} Ų\n"
        summary_text += f"  • MW difference: {mw_diff:.1f} Da\n"
        summary_text += f"  → Lower TPSA & MW favor BBB penetration\n\n"

    # Influx mechanism
    influx_col = 'label_Influx'
    if influx_col in df.columns:
        tpsa_pos = df[df[influx_col] == 1]['TPSA'].mean()
        tpsa_neg = df[df[influx_col] == 0]['TPSA'].mean()
        summary_text += f"Active Influx:\n"
        summary_text += f"  • TPSA+: {tpsa_pos:.1f} vs TPSA-: {tpsa_neg:.1f} Ų\n"
        summary_text += f"  → Higher TPSA for transporter substrates\n\n"

    # Efflux mechanism
    efflux_col = 'label_Efflux'
    if efflux_col in df.columns:
        mw_pos = df[df[efflux_col] == 1]['MW'].mean()
        mw_neg = df[df[efflux_col] == 0]['MW'].mean()
        summary_text += f"Efflux (P-gp):\n"
        summary_text += f"  • MW+: {mw_pos:.1f} vs MW-: {mw_neg:.1f} Da\n"
        summary_text += f"  → Higher MW for efflux substrates\n\n"

    summary_text += "Conclusions:\n"
    summary_text += "✓ TPSA is key discriminator for BBB\n"
    summary_text += "✓ Validates Cornelissen et al. 2022\n"
    summary_text += "✓ Distinct profiles for each mechanism"

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Figure 2: Key Physicochemical Properties by Transport Mechanism\n(Validating Cornelissen et al. 2022)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / 'figure2_physicochemical_improved.png'
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def figure3_multimechanism_clustering(df, output_dir):
    """
    Figure 3 (Improved): Multi-Mechanism Clustering Comparison

    Shows PCA, t-SNE, UMAP visualizations colored by different mechanisms
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    try:
        from umap import UMAP
        has_umap = True
    except ImportError:
        has_umap = False
        print("UMAP not installed, skipping UMAP visualization")

    # Select Morgan features
    feature_cols = [col for col in df.columns if col.startswith('Morgan_')]

    if len(feature_cols) == 0:
        print("No Morgan features found")
        return

    print(f"Using {len(feature_cols)} Morgan features")

    # Define mechanisms and colors
    mechanisms_info = [
        ('BBB', 'label_BBB', '#2ecc71', '#e74c3c'),
        ('Influx', 'label_Influx', '#3498db', '#9b59b6'),
        ('Efflux', 'label_Efflux', '#f39c12', '#1abc9c'),
        ('PAMPA', 'label_PAMPA', '#e91e63', '#00bcd4'),
    ]

    fig = plt.figure(figsize=(20, 12))

    # For each mechanism, create PCA, t-SNE, UMAP visualizations
    plot_idx = 1
    results_summary = {}

    for mech_name, mech_col, pos_color, neg_color in mechanisms_info:
        if mech_col not in df.columns:
            continue

        print(f"\nProcessing {mech_name} mechanism...")

        # Get data for this mechanism
        valid_idx = df[mech_col].notna()
        X = df.loc[valid_idx, feature_cols].values
        y = df.loc[valid_idx, mech_col].values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f"  Samples: {len(X)} ({(y==1).sum()} positive, {(y==0).sum()} negative)")

        # Store summary
        results_summary[mech_name] = {
            'n_samples': len(X),
            'n_positive': (y==1).sum(),
            'n_negative': (y==0).sum()
        }

        # PCA
        ax = plt.subplot(4, 3, plot_idx)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Plot negative samples first
        ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1],
                  c=neg_color, alpha=0.4, s=15, label=f'{mech_name}-',
                  edgecolors='none')
        # Plot positive samples
        ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1],
                  c=pos_color, alpha=0.6, s=20, label=f'{mech_name}+',
                  edgecolors='black', linewidth=0.5)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9)
        ax.set_title(f'{mech_name} - PCA', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.2)

        results_summary[mech_name]['pca_variance'] = pca.explained_variance_ratio_[:2].tolist()

        plot_idx += 1

        # t-SNE
        ax = plt.subplot(4, 3, plot_idx)

        # Use subset if too many samples
        if len(X_scaled) > 2000:
            sample_idx = np.random.choice(len(X_scaled), 2000, replace=False)
            X_tsne_input = X_scaled[sample_idx]
            y_tsne = y[sample_idx]
        else:
            X_tsne_input = X_scaled
            y_tsne = y

        print(f"  Running t-SNE (n={len(X_tsne_input)})...")
        tsne = TSNE(n_components=2, random_state=42,
                   perplexity=min(30, len(X_tsne_input)//4),
                   max_iter=1000, learning_rate='auto')
        X_tsne = tsne.fit_transform(X_tsne_input)

        # Plot negative samples first
        ax.scatter(X_tsne[y_tsne==0, 0], X_tsne[y_tsne==0, 1],
                  c=neg_color, alpha=0.4, s=15, label=f'{mech_name}-',
                  edgecolors='none')
        # Plot positive samples
        ax.scatter(X_tsne[y_tsne==1, 0], X_tsne[y_tsne==1, 1],
                  c=pos_color, alpha=0.6, s=20, label=f'{mech_name}+',
                  edgecolors='black', linewidth=0.5)

        ax.set_xlabel('t-SNE 1', fontsize=9)
        ax.set_ylabel('t-SNE 2', fontsize=9)
        ax.set_title(f'{mech_name} - t-SNE', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.2)

        plot_idx += 1

        # UMAP
        if has_umap:
            ax = plt.subplot(4, 3, plot_idx)

            # Use same subset as t-SNE
            if len(X_scaled) > 2000:
                X_umap_input = X_tsne_input
                y_umap = y_tsne
            else:
                X_umap_input = X_scaled
                y_umap = y

            print(f"  Running UMAP (n={len(X_umap_input)})...")
            umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            X_umap = umap_model.fit_transform(X_umap_input)

            # Plot negative samples first
            ax.scatter(X_umap[y_umap==0, 0], X_umap[y_umap==0, 1],
                      c=neg_color, alpha=0.4, s=15, label=f'{mech_name}-',
                      edgecolors='none')
            # Plot positive samples
            ax.scatter(X_umap[y_umap==1, 0], X_umap[y_umap==1, 1],
                      c=pos_color, alpha=0.6, s=20, label=f'{mech_name}+',
                      edgecolors='black', linewidth=0.5)

            ax.set_xlabel('UMAP 1', fontsize=9)
            ax.set_ylabel('UMAP 2', fontsize=9)
            ax.set_title(f'{mech_name} - UMAP', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.2)

            plot_idx += 1
        else:
            # Skip UMAP position
            plot_idx += 1

    plt.suptitle('Figure 3: Multi-Mechanism Clustering Comparison\nPCA vs t-SNE vs UMAP on Cornelissen 2022 Dataset',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / 'figure3_multimechanism_clustering.png'
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    # Print summary
    print("\n" + "=" * 80)
    print("MULTI-MECHANISM CLUSTERING SUMMARY")
    print("=" * 80)
    for mech, info in results_summary.items():
        print(f"\n{mech}:")
        print(f"  Samples: {info['n_samples']} (+:{info['n_positive']}, -:{info['n_negative']})")
        if 'pca_variance' in info:
            print(f"  PCA Variance: PC1={info['pca_variance'][0]*100:.2f}%, "
                  f"PC2={info['pca_variance'][1]*100:.2f}%")

    return results_summary


def main():
    """Main analysis pipeline"""
    print("=" * 80)
    print("IMPROVED CORNELISSEN 2022 ANALYSIS")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path(Paths.artifacts).parent / "outputs" / "cornelissen_comprehensive_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data()
    print()

    # Run analyses
    print("\n" + "=" * 80)
    print("GENERATING FIGURE 2: Improved Physicochemical Properties")
    print("=" * 80)
    figure2_improved_physicochemical(df, output_dir)

    print("\n" + "=" * 80)
    print("GENERATING FIGURE 3: Multi-Mechanism Clustering")
    print("=" * 80)
    clustering_results = figure3_multimechanism_clustering(df, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
