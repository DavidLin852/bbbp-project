"""
Final Improved Figures for Cornelissen 2022 Dataset

Figure 2: Clean boxplots for TPSA, LogP, MW, HBD, HBA
Figure 3: All mechanisms in ONE plot for each dimensionality reduction method
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


def figure2_clean_boxplots(df, output_dir):
    """
    Figure 2 (Final): Clean Boxplots for Key Properties

    Shows TPSA, LogP, MW, HBD, HBA with proper y-axis ranges based on data
    No key findings panel, no delta annotations
    """
    fig = plt.figure(figsize=(18, 10))

    # Key properties to analyze
    key_props = ['TPSA', 'LogP', 'MW', 'HBD', 'HBA']

    # Mechanisms to analyze
    mechanisms = ['BBB', 'Influx', 'Efflux', 'PAMPA']

    # Define fixed y-axis limits
    y_limits = {
        'TPSA': (0, 1000),
        'LogP': (-5, 15),
        'MW': (0, 2000),
        'HBD': (0, 30),
        'HBA': (0, 40)
    }

    # Create subplots (2 rows, 3 columns)
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
            # Use fixed y-axis limits
            y_lower, y_upper = y_limits.get(prop, (0, 100))

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

            ax.set_ylim(y_lower, y_upper)
            ax.set_ylabel(prop, fontsize=12, fontweight='bold')
            ax.set_title(f'{prop} Distribution by Mechanism', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Figure 2: Physicochemical Properties by Transport Mechanism\n(Cornelissen et al. 2022 Dataset)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / 'figure2_clean_boxplots.png'
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def figure3_allmechanisms_oneplot(df, output_dir):
    """
    Figure 3 (Final): All Mechanisms in ONE Plot for Each Method

    Shows PCA, t-SNE, UMAP with all mechanisms (BBB, Influx, Efflux, PAMPA)
    in a single plot for each dimensionality reduction method
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler, LabelEncoder
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

    # Define mechanisms with colors and markers
    mechanisms_info = [
        ('BBB', 'label_BBB', '#2ecc71', 'o'),      # Green, circle
        ('Influx', 'label_Influx', '#3498db', 's'), # Blue, square
        ('Efflux', 'label_Efflux', '#f39c12', '^'), # Orange, triangle up
        ('PAMPA', 'label_PAMPA', '#e91e63', 'd'),   # Pink, diamond
    ]

    # Prepare combined dataset
    # We'll create a unified visualization where each compound is labeled
    # by its mechanism status (positive/negative for each mechanism)

    fig = plt.figure(figsize=(20, 6))

    # For visualization, we'll focus on BBB as the primary label
    # but show all mechanisms with different colors/markers

    # Create a combined label for visualization
    # Priority: BBB > Influx > Efflux > PAMPA
    df_vis = df.copy()

    # Add a priority column for visualization
    df_vis['viz_label'] = 'Other'
    df_vis['viz_color'] = '#95a5a6'  # Gray
    df_vis['viz_marker'] = 'o'

    for mech_name, mech_col, color, marker in mechanisms_info:
        if mech_col not in df.columns:
            continue
        # Mark positive samples for this mechanism
        df_vis.loc[df_vis[mech_col] == 1, 'viz_label'] = mech_name
        df_vis.loc[df_vis[mech_col] == 1, 'viz_color'] = color
        df_vis.loc[df_vis[mech_col] == 1, 'viz_marker'] = marker

    # Get data for compounds that have at least one mechanism label
    valid_idx = df_vis['viz_label'] != 'Other'
    X = df_vis.loc[valid_idx, feature_cols].values
    labels = df_vis.loc[valid_idx, 'viz_label'].values
    colors = df_vis.loc[valid_idx, 'viz_color'].values

    print(f"\nTotal samples with mechanism labels: {len(X)}")
    print("Label distribution:")
    for mech_name, _, color, _ in mechanisms_info:
        count = (labels == mech_name).sum()
        print(f"  {mech_name}: {count}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    ax1 = plt.subplot(1, 3, 1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot each mechanism
    for mech_name, _, color, marker in mechanisms_info:
        mask = labels == mech_name
        if mask.sum() > 0:
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=color, marker=marker, alpha=0.6, s=30,
                       label=mech_name, edgecolors='black', linewidth=0.5)

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                   fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                   fontsize=11, fontweight='bold')
    ax1.set_title('PCA', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # t-SNE
    ax2 = plt.subplot(1, 3, 2)

    # Use subset if too many samples
    if len(X_scaled) > 3000:
        sample_idx = np.random.choice(len(X_scaled), 3000, replace=False)
        X_tsne_input = X_scaled[sample_idx]
        labels_tsne = labels[sample_idx]
    else:
        X_tsne_input = X_scaled
        labels_tsne = labels

    print(f"\nRunning t-SNE (n={len(X_tsne_input)})...")
    tsne = TSNE(n_components=2, random_state=42,
               perplexity=min(30, len(X_tsne_input)//4),
               max_iter=1000, learning_rate='auto')
    X_tsne = tsne.fit_transform(X_tsne_input)

    # Plot each mechanism
    for mech_name, _, color, marker in mechanisms_info:
        mask = labels_tsne == mech_name
        if mask.sum() > 0:
            ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=color, marker=marker, alpha=0.6, s=30,
                       label=mech_name, edgecolors='black', linewidth=0.5)

    ax2.set_xlabel('t-SNE 1', fontsize=11, fontweight='bold')
    ax2.set_ylabel('t-SNE 2', fontsize=11, fontweight='bold')
    ax2.set_title('t-SNE', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # UMAP
    if has_umap:
        ax3 = plt.subplot(1, 3, 3)

        # Use same subset as t-SNE
        if len(X_scaled) > 3000:
            X_umap_input = X_tsne_input
            labels_umap = labels_tsne
        else:
            X_umap_input = X_scaled
            labels_umap = labels

        print(f"Running UMAP (n={len(X_umap_input)})...")
        umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap = umap_model.fit_transform(X_umap_input)

        # Plot each mechanism
        for mech_name, _, color, marker in mechanisms_info:
            mask = labels_umap == mech_name
            if mask.sum() > 0:
                ax3.scatter(X_umap[mask, 0], X_umap[mask, 1],
                           c=color, marker=marker, alpha=0.6, s=30,
                           label=mech_name, edgecolors='black', linewidth=0.5)

        ax3.set_xlabel('UMAP 1', fontsize=11, fontweight='bold')
        ax3.set_ylabel('UMAP 2', fontsize=11, fontweight='bold')
        ax3.set_title('UMAP', fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)

    plt.suptitle('Figure 3: Multi-Mechanism Clustering Comparison\nAll Transport Mechanisms in One Plot (Cornelissen et al. 2022 Dataset)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = output_dir / 'figure3_allmechanisms_oneplot.png'
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    # Print summary
    print("\n" + "=" * 80)
    print("VISUALIZATION SUMMARY")
    print("=" * 80)
    print(f"Total compounds with mechanism labels: {len(X)}")
    for mech_name, _, _, _ in mechanisms_info:
        count = (labels == mech_name).sum()
        print(f"  {mech_name}: {count} compounds")


def main():
    """Main analysis pipeline"""
    print("=" * 80)
    print("FINAL IMPROVED FIGURES FOR CORNELISSEN 2022")
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
    print("GENERATING FIGURE 2: Clean Boxplots")
    print("=" * 80)
    figure2_clean_boxplots(df, output_dir)

    print("\n" + "=" * 80)
    print("GENERATING FIGURE 3: All Mechanisms in One Plot")
    print("=" * 80)
    figure3_allmechanisms_oneplot(df, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
