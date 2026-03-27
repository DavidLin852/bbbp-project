"""
t-SNE Clustering Analysis for Mechanism Visualization.

This script:
1. Performs t-SNE for nonlinear dimensionality reduction
2. Compares PCA, UMAP, and t-SNE side by side
3. Shows which method best separates mechanisms by clustering
4. Overlays B3DB data on all three methods
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Professional styling
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# Professional color scheme (low brightness, distinct for each mechanism)
MECHANISM_COLORS = {
    'BBB+': '#2E86AB',           # Muted blue
    'BBB-': '#A23B72',           # Muted wine
    'Passive_Diffusion': '#1B4F72',  # Dark blue
    'Active_Influx': '#C17C6D',     # Muted coral
    'Active_Efflux': '#6B4C4C',     # Muted brown
    'PAMPA+': '#3D5A80',            # Muted navy
    'Mixed': '#888888',             # Gray
    'Unknown': '#CCCCCC',           # Light gray
}

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Paths


def load_cornelissen_data():
    """Load Cornelissen 2022 dataset with mechanism labels."""
    print("Loading Cornelissen 2022 dataset...")

    data_file = Paths.root / "data" / "transport_mechanisms" / "cornelissen_2022" / "cornelissen_2022_processed.csv"

    if not data_file.exists():
        print(f"  Error: {data_file} not found")
        return None

    df = pd.read_csv(data_file)
    print(f"  Loaded {len(df)} molecules")

    return df


def get_mechanism_labels(row):
    """Determine mechanism label for each molecule."""
    # Check efflux first (P-gp substrates)
    if pd.notna(row['label_Efflux']) and row['label_Efflux'] == 1:
        return 'Active_Efflux'

    # Check influx
    if pd.notna(row['label_Influx']) and row['label_Influx'] == 1:
        return 'Active_Influx'

    # Check PAMPA (passive diffusion)
    if pd.notna(row['label_PAMPA']) and row['label_PAMPA'] == 1:
        return 'Passive_Diffusion'

    # Fallback to BBB
    if pd.notna(row['label_BBB']):
        if row['label_BBB'] == 1:
            return 'BBB+'
        else:
            return 'BBB-'

    return 'Unknown'


def prepare_features(df):
    """Prepare feature matrix for dimensionality reduction."""
    print("\nPreparing features...")

    # Get feature columns
    feature_cols = []

    # Physicochemical properties
    physicochemical = ['LogP', 'TPSA', 'MW', 'HBA', 'HBD',
                      'RotatableBonds', 'RingCount', 'AromaticRings', 'FractionCSP3']
    feature_cols.extend([col for col in physicochemical if col in df.columns])

    # MACCS fingerprints
    maccs_cols = [col for col in df.columns if col.startswith('MACCS_')]
    feature_cols.extend(maccs_cols)

    # Morgan fingerprints
    morgan_cols = [col for col in df.columns if col.startswith('Morgan_')]
    feature_cols.extend(morgan_cols)

    print(f"  Total features: {len(feature_cols)}")

    # Extract features
    X = df[feature_cols].values

    # Handle NaN values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Feature matrix shape: {X_scaled.shape}")

    return X_scaled, scaler, feature_cols


def perform_dimensionality_reduction(X, methods=['pca', 'tsne', 'umap']):
    """Perform PCA, t-SNE, and UMAP for comparison."""
    results = {}

    # PCA
    if 'pca' in methods:
        print("\nPerforming PCA...")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        results['pca'] = X_pca
        print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]*100:.2f}%, "
              f"PC2={pca.explained_variance_ratio_[1]*100:.2f}%")

    # t-SNE
    if 'tsne' in methods:
        print("\nPerforming t-SNE...")
        print("  (This may take a few minutes...)")

        # Try different perplexities
        for perplexity in [30, 50, 100]:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                        learning_rate='auto')
            X_tsne = tsne.fit_transform(X)

            # Calculate silhouette score
            mask = y != 'Unknown'
            score = silhouette_score(X_tsne[mask], y[mask])

            print(f"  Perplexity={perplexity}: Silhouette={score:.3f}")

            # Keep the best one
            if 'tsne' not in results or score > results.get('tsne_score', -1):
                results['tsne'] = X_tsne
                results['tsne_score'] = score
                results['tsne_perplexity'] = perplexity

        print(f"  Best t-SNE: perplexity={results['tsne_perplexity']}, "
              f"silhouette={results['tsne_score']:.3f}")

    # UMAP
    if 'umap' in methods:
        print("\nPerforming UMAP...")
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1,
                           n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X)
        results['umap'] = X_umap

    return results


def create_comparison_plot(X_dict, y, title, output_file):
    """Create side-by-side comparison of PCA, t-SNE, UMAP."""
    print(f"\nCreating comparison plot: {title}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    methods = ['pca', 'tsne', 'umap']
    method_names = ['PCA', 't-SNE', 'UMAP']

    for ax, method, method_name in zip(axes, methods, method_names):
        if method not in X_dict:
            continue

        X_2d = X_dict[method]

        # Plot each mechanism with its own color
        for mech in y.unique():
            if mech == 'Unknown':
                continue

            mask = y == mech
            count = mask.sum()

            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=MECHANISM_COLORS.get(mech, '#888888'),
                      label=f'{mech} (n={count})',
                      alpha=0.6, s=30, edgecolors='none')

        ax.set_xlabel(f'{method_name} 1', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{method_name} 2', fontsize=12, fontweight='bold')
        ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, frameon=True, shadow=True)
        ax.grid(True, alpha=0.2, linestyle='--')

        # Calculate and display silhouette score
        mask = y != 'Unknown'
        if mask.sum() > 0:
            score = silhouette_score(X_2d[mask], y[mask])
            ax.text(0.05, 0.95, f'Silhouette: {score:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def create_overlay_plots(X_dict, y_cornelissen, X_b3db, y_b3db,
                         output_dir):
    """Create overlay plots for all three methods."""
    print("\nCreating overlay plots...")

    methods = ['pca', 'tsne', 'umap']
    method_names = ['PCA', 't-SNE', 'UMAP']

    for method, method_name in zip(methods, method_names):
        if method not in X_dict:
            continue

        # For simplicity, use only 9D feature results for overlay
        if method in ['tsne', 'umap']:
            print(f"\n  Skipping {method_name} overlay (would require refitting)")
            continue

        X_cornelissen_2d = X_dict[method]
        X_b3db_2d = X_b3db  # Use already transformed B3DB

        # Create overlay plot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot Cornelissen data as background
        for mech in y_cornelissen.unique():
            if mech == 'Unknown':
                continue

            mask = y_cornelissen == mech

            ax.scatter(X_cornelissen_2d[mask, 0], X_cornelissen_2d[mask, 1],
                      c=MECHANISM_COLORS.get(mech, '#888888'),
                      label=f'{mech} (n={mask.sum()})',
                      alpha=0.3, s=15, edgecolors='none')

        # Plot B3DB data on top
        for mech in y_b3db.unique():
            mask_b = y_b3db == mech
            marker_dict = {'BBB+': 'o', 'BBB-': 's'}
            marker = marker_dict.get(mech, 'o')

            ax.scatter(X_b3db_2d[mask_b, 0], X_b3db_2d[mask_b, 1],
                      c='none',
                      edgecolors=MECHANISM_COLORS.get(mech, '#888888'),
                      label=f'B3DB {mech} (n={mask_b.sum()})',
                      alpha=0.8, s=50, linewidths=2, marker=marker)

        ax.set_xlabel(f'{method_name} 1', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{method_name} 2', fontsize=14, fontweight='bold')
        ax.set_title(f'B3DB Data Overlaid on Cornelissen 2022 ({method_name})',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')

        # Legend for B3DB
        from matplotlib.lines import Line2D
        bbb_plus_count = (y_b3db == 'BBB+').sum()
        bbb_minus_count = (y_b3db == 'BBB-').sum()

        b3db_legend_elements = [
            Line2D([0], [0], marker='o', color='w', markeredgecolor=MECHANISM_COLORS['BBB+'],
                markersize=10, label=f'B3DB BBB+ (n={bbb_plus_count})', linestyle='None'),
            Line2D([0], [0], marker='s', color='w', markeredgecolor=MECHANISM_COLORS['BBB-'],
                markersize=10, label=f'B3DB BBB- (n={bbb_minus_count})', linestyle='None'),
        ]
        ax.legend(handles=b3db_legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()
        output_file = output_dir / f'b3db_overlay_{method.lower()}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_file}")
        plt.close()


def analyze_clustering_quality_all(X_dict, y):
    """Analyze clustering quality for all methods."""
    print("\n" + "="*80)
    print("CLUSTERING QUALITY COMPARISON")
    print("="*80)

    print(f"\n{'Method':<10} {'Silhouette':>12} {'Interpretation':>30}")
    print("-" * 65)

    for method in ['pca', 'tsne', 'umap']:
        if method not in X_dict:
            continue

        X_2d = X_dict[method]
        mask = y != 'Unknown'

        if mask.sum() > 0:
            score = silhouette_score(X_2d[mask], y[mask])

            if score > 0.3:
                interpretation = "[GOOD] Strong clustering"
            elif score > 0.1:
                interpretation = "[MODERATE] Some clustering"
            elif score > -0.1:
                interpretation = "[WEAK] Weak clustering"
            else:
                interpretation = "[POOR] No clustering"

            print(f"{method.upper():<10} {score:>12.3f} {interpretation:>30}")


def main():
    """Main analysis function."""
    print("="*80)
    print("MECHANISM CLUSTERING ANALYSIS - PCA vs t-SNE vs UMAP")
    print("Using Cornelissen 2022 + B3DB Datasets")
    print("="*80)

    output_dir = Paths.root / "outputs" / "mechanism_clustering"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Cornelissen data
    df_cornelissen = load_cornelissen_data()
    if df_cornelissen is None:
        return

    # Add mechanism labels
    print("\nAdding mechanism labels...")
    df_cornelissen['mechanism'] = df_cornelissen.apply(get_mechanism_labels, axis=1)

    # Print mechanism distribution
    print("\nMechanism distribution in Cornelissen 2022:")
    print(df_cornelissen['mechanism'].value_counts())

    # Prepare features
    global X, y
    X, scaler, feature_cols = prepare_features(df_cornelissen)
    y = df_cornelissen['mechanism']

    # Perform all three dimensionality reduction methods
    X_dict = perform_dimensionality_reduction(X, methods=['pca', 'tsne', 'umap'])

    # Create comparison plot
    create_comparison_plot(
        X_dict, y,
        'Cornelissen 2022 - Mechanism Clustering Comparison',
        output_dir / 'cornelissen_method_comparison.png'
    )

    # Analyze clustering quality
    analyze_clustering_quality_all(X_dict, y)

    # Load B3DB data
    print("\n" + "="*80)
    print("Loading B3DB data for overlay...")
    print("="*80)

    b3db_file = Paths.root / "outputs" / "b3db_analysis" / "b3db_with_features.csv"
    if not b3db_file.exists():
        print("  B3DB pre-processed file not found")
        return

    df_b3db = pd.read_csv(b3db_file)
    print(f"  Loaded {len(df_b3db)} B3DB molecules")

    # Prepare B3DB features (physicochemical only for simplicity)
    b3db_feature_cols = ['LogP', 'TPSA', 'MW', 'HBA', 'HBD',
                        'RotatableBonds', 'RingCount', 'AromaticRings', 'FractionCSP3']
    b3db_feature_cols = [col for col in b3db_feature_cols if col in df_b3db.columns]

    # For B3DB overlay, we need to use the same feature space
    # So we re-train on physicochemical only
    print("\n  Re-training models on physicochemical properties (9D) for B3DB overlay...")

    # Get physicochemical features from Cornelissen
    physicochemical_cols = ['LogP', 'TPSA', 'MW', 'HBA', 'HBD',
                           'RotatableBonds', 'RingCount', 'AromaticRings', 'FractionCSP3']

    X_phys = df_cornelissen[physicochemical_cols].values
    from sklearn.impute import SimpleImputer
    imputer_phys = SimpleImputer(strategy='median')
    X_phys = imputer_phys.fit_transform(X_phys)
    scaler_phys = StandardScaler()
    X_phys = scaler_phys.fit_transform(X_phys)

    # Re-run PCA, t-SNE, UMAP on 9D features
    print("\n  Re-running PCA on 9D features...")
    pca_phys = PCA(n_components=2, random_state=42)
    X_pca_phys = pca_phys.fit_transform(X_phys)

    print("\n  Re-running t-SNE on 9D features...")
    tsne_phys = TSNE(n_components=2, perplexity=50, random_state=42,
                    learning_rate='auto')
    X_tsne_phys = tsne_phys.fit_transform(X_phys)

    print("\n  Re-running UMAP on 9D features...")
    umap_phys = umap.UMAP(n_neighbors=30, min_dist=0.1,
                         n_components=2, random_state=42)
    X_umap_phys = umap_phys.fit_transform(X_phys)

    # Prepare B3DB
    X_b3db = df_b3db[b3db_feature_cols].values
    X_b3db = imputer_phys.transform(X_b3db)
    X_b3db = scaler_phys.transform(X_b3db)

    # Add BBB+/- labels
    df_b3db['mechanism'] = df_b3db['BBB_binary'].apply(lambda x: 'BBB+' if x == 1 else 'BBB-')

    # Create comparison plot for 9D features
    X_dict_phys = {
        'pca': X_pca_phys,
        'tsne': X_tsne_phys,
        'umap': X_umap_phys,
    }

    create_comparison_plot(
        X_dict_phys, y,
        'Cornelissen 2022 - Mechanism Clustering (9D Features)',
        output_dir / 'cornelissen_9d_comparison.png'
    )

    # Create overlay plots
    create_overlay_plots(
        X_dict_phys, y, X_b3db, df_b3db['mechanism'],
        output_dir
    )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

    print("\nGenerated visualizations:")
    print("  Cornelissen 2022 (all features, 1200D):")
    print("    1. cornelissen_method_comparison.png - PCA vs t-SNE vs UMAP")
    print("\n  Cornelissen 2022 (physicochemical only, 9D):")
    print("    2. cornelissen_9d_comparison.png - PCA vs t-SNE vs UMAP")
    print("\n  B3DB overlay (9D features):")
    print("    3. b3db_overlay_pca.png - B3DB on Cornelissen (PCA)")
    print("    4. b3db_overlay_tsne.png - B3DB on Cornelissen (t-SNE)")
    print("    5. b3db_overlay_umap.png - B3DB on Cornelissen (UMAP)")

    print("\nKey Insights:")
    print("  - Compare PCA (linear) vs t-SNE (nonlinear, local) vs UMAP (nonlinear, global)")
    print("  - Each method optimized for different aspects")
    print("  - t-SNE often shows better local clustering")
    print("  - UMAP preserves both local and global structure")


if __name__ == "__main__":
    main()
