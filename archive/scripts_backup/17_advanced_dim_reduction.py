"""
Advanced dimensionality reduction analysis for BBB permeability.

Supports multiple methods (PCA, t-SNE, UMAP) and feature types (Morgan, Descriptors, SMARTS, Combined).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz, hstack
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem

from src.config import Paths, DatasetConfig

# Try to import UMAP (optional)
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not installed. Install with: pip install umap-learn")


def load_smarts_patterns(smarts_json: Path):
    """Load SMARTS patterns from JSON file.

    Returns:
        List of (name, smarts_pattern) tuples
    """
    with open(smarts_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    patterns = []
    for item in data:
        if isinstance(item, dict) and "name" in item and "smarts" in item:
            patterns.append((item["name"], item["smarts"]))

    print(f"Loaded {len(patterns)} SMARTS patterns from {smarts_json.name}")
    return patterns


def compute_smarts_features(smiles_list, smarts_patterns):
    """Compute SMARTS binary features for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings
        smarts_patterns: List of (name, smarts_pattern) tuples

    Returns:
        numpy array of shape (n_molecules, n_patterns) with binary features
    """
    print(f"Computing SMARTS features for {len(smiles_list)} molecules...")

    n_patterns = len(smarts_patterns)
    features = np.zeros((len(smiles_list), n_patterns), dtype=np.int8)

    for i, smiles in enumerate(smiles_list):
        if i % 500 == 0:
            print(f"  Processing molecule {i}/{len(smiles_list)}...")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        for j, (name, pattern) in enumerate(smarts_patterns):
            try:
                smarts_mol = Chem.MolFromSmarts(pattern)
                if smarts_mol and mol.HasSubstructMatch(smarts_mol):
                    features[i, j] = 1
            except:
                pass

    print(f"  Completed. Average SMARTS per molecule: {features.mean(axis=1).mean():.2f}")
    return features


def load_features(feature_dir: Path, feature_type: str, smiles_list=None, smarts_json=None):
    """Load molecular features and metadata.

    Args:
        feature_dir: Directory containing feature files
        feature_type: Type of features ('morgan', 'descriptors', 'smarts', 'combined', 'all')
        smiles_list: List of SMILES (required for SMARTS features)
        smarts_json: Path to SMARTS JSON file (required for SMARTS features)

    Returns:
        X: Feature matrix (sparse or dense)
        meta: Metadata DataFrame
    """
    # Load metadata
    meta_path = feature_dir / "meta.csv"
    meta = pd.read_csv(meta_path)

    if smiles_list is None:
        smiles_list = meta['SMILES'].tolist()

    features_list = []
    feature_names = []

    # Load Morgan fingerprints
    if feature_type in ['morgan', 'combined', 'all']:
        X_morgan = load_npz(feature_dir / "morgan_2048.npz")
        features_list.append(X_morgan)
        feature_names.append('Morgan')

    # Load descriptors
    if feature_type in ['descriptors', 'combined', 'all']:
        X_desc = pd.read_csv(feature_dir / "descriptors.csv", index_col=0).values
        features_list.append(X_desc)
        feature_names.append('Descriptors')

    # Load SMARTS features
    if feature_type in ['smarts', 'all']:
        if smarts_json is None:
            smarts_json = Paths().data_external / "smarts" / "bbb_smarts_v1.json"
        smarts_patterns = load_smarts_patterns(smarts_json)
        X_smarts = compute_smarts_features(smiles_list, smarts_patterns)
        features_list.append(X_smarts)
        feature_names.append('SMARTS')

    # Combine features
    if len(features_list) == 1:
        X = features_list[0]
    else:
        # Convert sparse matrices to dense for stacking with SMARTS
        from scipy.sparse import issparse
        dense_features = []
        for feat in features_list:
            if issparse(feat):
                dense_features.append(feat.toarray())
            else:
                dense_features.append(feat)
        X = np.hstack(dense_features)

    print(f"Loaded features: {' + '.join(feature_names)}")
    print(f"  Total dimensions: {X.shape[1]}")

    return X, meta


def run_dimensionality_reduction(X, method='tsne', **kwargs):
    """Run dimensionality reduction.

    Args:
        X: Feature matrix
        method: Reduction method ('pca', 'tsne', 'umap')
        **kwargs: Method-specific parameters

    Returns:
        X_embedded: 2D embedded coordinates
    """
    # Convert sparse to dense if needed
    from scipy.sparse import issparse
    if issparse(X):
        X = X.toarray()

    method = method.lower()
    print(f"\nRunning {method.upper()}...")

    if method == 'pca':
        reducer = PCA(
            n_components=2,
            random_state=kwargs.get('random_state', 42)
        )
        X_embedded = reducer.fit_transform(X)
        print(f"  Explained variance ratio: {reducer.explained_variance_ratio_}")
        print(f"  Total explained variance: {reducer.explained_variance_ratio_.sum():.3f}")

    elif method == 'tsne':
        reducer = TSNE(
            n_components=2,
            perplexity=kwargs.get('perplexity', 30),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42),
            verbose=1,
            n_jobs=-1
        )
        X_embedded = reducer.fit_transform(X)

    elif method == 'umap':
        if not HAS_UMAP:
            raise ImportError("UMAP is not installed. Run: pip install umap-learn")
        reducer = UMAP(
            n_components=2,
            n_neighbors=kwargs.get('n_neighbors', 15),
            min_dist=kwargs.get('min_dist', 0.1),
            random_state=kwargs.get('random_state', 42),
            verbose=True
        )
        X_embedded = reducer.fit_transform(X)

    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'pca', 'tsne', 'umap'")

    return X_embedded


def plot_embedding_2d(X_embedded, meta, output_path: Path, split: str = None,
                      method_name: str = "t-SNE", title_prefix: str = ""):
    """Create 2D scatter plot.

    Args:
        X_embedded: 2D embedded coordinates
        meta: Metadata DataFrame
        output_path: Path to save the plot
        split: If specified, only plot this split ('train', 'val', 'test')
        method_name: Name of the reduction method for title
        title_prefix: Prefix for plot title
    """
    # Filter by split if specified
    if split:
        mask = meta['split'] == split
        X_embedded = X_embedded[mask]
        meta_filtered = meta[mask].copy()
    else:
        meta_filtered = meta.copy()

    # Create labels
    labels = meta_filtered['y_cls'].map({1: 'BBB+', 0: 'BBB-'})

    # Set up the plot style
    plt.style.use('default')
    plt.figure(figsize=(10, 8))

    # Define colors
    colors = {'BBB+': '#2ecc71', 'BBB-': '#e74c3c'}

    # Plot each class separately for better legend
    for label in ['BBB+', 'BBB-']:
        mask = labels == label
        plt.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            c=colors[label],
            label=label,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )

    plt.xlabel(f'{method_name} Dimension 1', fontsize=12, fontname='Times New Roman')
    plt.ylabel(f'{method_name} Dimension 2', fontsize=12, fontname='Times New Roman')
    title = f"{title_prefix}BBB Permeability - {method_name} Visualization"
    if split:
        title += f" ({split.upper()} set)"
    plt.title(title, fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.legend(fontsize=11, prop={'family': 'Times New Roman'})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to {output_path}")
    plt.close()


def plot_embedding_by_split(X_embedded, meta, output_path: Path,
                            method_name: str = "t-SNE", title_prefix: str = ""):
    """Create 2D scatter plot showing all splits together.

    Args:
        X_embedded: 2D embedded coordinates
        meta: Metadata DataFrame
        output_path: Path to save the plot
        method_name: Name of the reduction method for title
        title_prefix: Prefix for plot title
    """
    from itertools import product

    # Create labels
    labels = meta['y_cls'].map({1: 'BBB+', 0: 'BBB-'})
    splits = meta['split']

    # Create plot
    plt.figure(figsize=(12, 10))

    # Define colors, markers, and alphas
    colors = {'BBB+': '#2ecc71', 'BBB-': '#e74c3c'}
    markers = {'train': 'o', 'val': 's', 'test': '^'}
    alphas = {'train': 0.4, 'val': 0.6, 'test': 0.8}

    # Plot each combination
    for (bbb_label, split_name) in product(['BBB+', 'BBB-'], ['train', 'val', 'test']):
        mask = (labels == bbb_label) & (splits == split_name)
        if mask.sum() == 0:
            continue

        plt.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            c=colors[bbb_label],
            label=f'{bbb_label} ({split_name})',
            alpha=alphas[split_name],
            s=60,
            marker=markers[split_name],
            edgecolors='black',
            linewidth=0.5
        )

    plt.xlabel(f'{method_name} Dimension 1', fontsize=12, fontname='Times New Roman')
    plt.ylabel(f'{method_name} Dimension 2', fontsize=12, fontname='Times New Roman')
    plt.title(f"{title_prefix}BBB Permeability - {method_name} (All Splits)",
              fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.legend(fontsize=10, prop={'family': 'Times New Roman'}, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to {output_path}")
    plt.close()


def plot_separation_metrics(X_embedded, meta, output_path: Path, method_name: str = "t-SNE"):
    """Calculate and visualize separation metrics between BBB+ and BBB-.

    Args:
        X_embedded: 2D embedded coordinates
        meta: Metadata DataFrame
        output_path: Path to save the plot
        method_name: Name of the reduction method
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from scipy.spatial.distance import euclidean

    # Filter for test set only
    test_mask = meta['split'] == 'test'
    X_test = X_embedded[test_mask]
    y_test = meta[test_mask]['y_cls'].values

    # Calculate centroids
    bbb_plus_mask = y_test == 1
    bbb_minus_mask = y_test == 0

    centroid_plus = X_test[bbb_plus_mask].mean(axis=0)
    centroid_minus = X_test[bbb_minus_mask].mean(axis=0)

    # Calculate distance between centroids
    centroid_distance = euclidean(centroid_plus, centroid_minus)

    # Calculate spread within each class
    spread_plus = np.mean([euclidean(X_test[i], centroid_plus)
                          for i in np.where(bbb_plus_mask)[0]])
    spread_minus = np.mean([euclidean(X_test[i], centroid_minus)
                           for i in np.where(bbb_minus_mask)[0]])

    # Calculate silhouette score (higher is better, max 1)
    silhouette = silhouette_score(X_test, y_test)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Scatter with centroids
    ax1 = axes[0]
    ax1.scatter(X_test[bbb_plus_mask, 0], X_test[bbb_plus_mask, 1],
               c='#2ecc71', label='BBB+', alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
    ax1.scatter(X_test[bbb_minus_mask, 0], X_test[bbb_minus_mask, 1],
               c='#e74c3c', label='BBB-', alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
    ax1.scatter([centroid_plus[0]], [centroid_plus[1]],
               c='darkgreen', marker='X', s=300, label='BBB+ Centroid', edgecolors='black', linewidth=2)
    ax1.scatter([centroid_minus[0]], [centroid_minus[1]],
               c='darkred', marker='X', s=300, label='BBB- Centroid', edgecolors='black', linewidth=2)

    # Draw line between centroids
    ax1.plot([centroid_plus[0], centroid_minus[0]],
            [centroid_plus[1], centroid_minus[1]],
            'k--', linewidth=2, alpha=0.5)

    ax1.set_xlabel(f'{method_name} Dimension 1', fontsize=11, fontname='Times New Roman')
    ax1.set_ylabel(f'{method_name} Dimension 2', fontsize=11, fontname='Times New Roman')
    ax1.set_title('Test Set Distribution with Centroids', fontsize=12, fontname='Times New Roman', fontweight='bold')
    ax1.legend(fontsize=9, prop={'family': 'Times New Roman'})
    ax1.grid(True, alpha=0.3)

    # Plot 2: Metrics bar chart
    ax2 = axes[1]
    metrics = {
        'Centroid\nDistance': centroid_distance,
        'BBB+\nSpread': spread_plus,
        'BBB-\nSpread': spread_minus,
        'Silhouette\nScore': silhouette
    }

    x = list(metrics.keys())
    y = list(metrics.values())
    colors_bars = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    bars = ax2.bar(x, y, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, y):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=11, fontname='Times New Roman', fontweight='bold')

    ax2.set_ylabel('Value', fontsize=11, fontname='Times New Roman')
    ax2.set_title('Separation Metrics (Test Set)', fontsize=12, fontname='Times New Roman', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved separation metrics to {output_path}")
    plt.close()

    # Print metrics
    print("\n" + "="*60)
    print(f"SEPARATION METRICS ({method_name} - Test Set)")
    print("="*60)
    print(f"Centroid Distance: {centroid_distance:.3f}")
    print(f"BBB+ Spread:       {spread_plus:.3f}")
    print(f"BBB- Spread:       {spread_minus:.3f}")
    print(f"Silhouette Score:  {silhouette:.3f} (higher is better)")
    print("="*60)


def print_statistics(meta):
    """Print dataset statistics."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    for split_name in ['train', 'val', 'test']:
        split_data = meta[meta['split'] == split_name]
        n_total = len(split_data)
        n_bbb_plus = (split_data['y_cls'] == 1).sum()
        n_bbb_minus = (split_data['y_cls'] == 0).sum()
        pct_bbb_plus = 100 * n_bbb_plus / n_total if n_total > 0 else 0

        print(f"\n{split_name.upper()} set:")
        print(f"  Total: {n_total}")
        print(f"  BBB+: {n_bbb_plus} ({pct_bbb_plus:.1f}%)")
        print(f"  BBB-: {n_bbb_minus} ({100-pct_bbb_plus:.1f}%)")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Advanced dimensionality reduction analysis for BBB permeability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PCA with Morgan fingerprints
  python scripts/17_advanced_dim_reduction.py --method pca --feature_type morgan

  # t-SNE with SMARTS features only
  python scripts/17_advanced_dim_reduction.py --method tsne --feature_type smarts

  # UMAP with combined features (Morgan + Descriptors + SMARTS)
  python scripts/17_advanced_dim_reduction.py --method umap --feature_type all

  # PCA with all features on test set only
  python scripts/17_advanced_dim_reduction.py --method pca --feature_type all --split test

  # Compare multiple methods
  python scripts/17_advanced_dim_reduction.py --method tsne --feature_type all
  python scripts/17_advanced_dim_reduction.py --method umap --feature_type all
  python scripts/17_advanced_dim_reduction.py --method pca --feature_type all
        """
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed for data split")
    parser.add_argument("--method", type=str, default="tsne",
                       choices=["pca", "tsne", "umap"],
                       help="Dimensionality reduction method")
    parser.add_argument("--feature_type", type=str, default="all",
                       choices=["morgan", "descriptors", "smarts", "combined", "all"],
                       help="Type of features to use")
    parser.add_argument("--smarts_json", type=str, default=None,
                       help="Path to SMARTS JSON file (default: assets/smarts/bbb_smarts_v1.json)")

    # Method-specific parameters
    parser.add_argument("--perplexity", type=int, default=30,
                       help="t-SNE perplexity (default: 30, recommended: 5-50)")
    parser.add_argument("--max_iter", type=int, default=1000,
                       help="t-SNE/UMAP number of iterations (default: 1000)")
    parser.add_argument("--n_neighbors", type=int, default=15,
                       help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--min_dist", type=float, default=0.1,
                       help="UMAP min_dist (default: 0.1)")

    # Visualization parameters
    parser.add_argument("--split", type=str, default=None,
                       choices=["train", "val", "test"],
                       help="Only visualize specific split (default: all)")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reduction method")

    args = parser.parse_args()

    # Setup paths
    paths = Paths()
    feature_dir = paths.features / f"seed_{args.seed}"
    output_dir = paths.figures / "dim_reduction"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if features exist
    if not feature_dir.exists():
        print(f"Error: Feature directory not found: {feature_dir}")
        print("Please run featurization first: python scripts/02_featurize_all.py --seed {args.seed}")
        return

    # Set SMARTS path
    smarts_json = None
    if args.feature_type in ['smarts', 'all']:
        if args.smarts_json:
            smarts_json = Path(args.smarts_json)
        else:
            smarts_json = paths.root / "assets" / "smarts" / "bbb_smarts_v1.json"
        if not smarts_json.exists():
            print(f"Error: SMARTS file not found: {smarts_json}")
            return

    # Load features and metadata
    print(f"Loading features from {feature_dir}...")
    print(f"Method: {args.method.upper()}")
    print(f"Feature type: {args.feature_type}")
    print()

    X, meta = load_features(feature_dir, args.feature_type, smarts_json=smarts_json)
    print(f"Total samples: {X.shape[0]}, Feature dimensions: {X.shape[1]}")

    # Print statistics
    print_statistics(meta)

    # Run dimensionality reduction
    X_embedded = run_dimensionality_reduction(
        X,
        method=args.method,
        perplexity=args.perplexity,
        max_iter=args.max_iter,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state
    )

    # Prepare output paths
    method_upper = args.method.upper()
    feature_suffix = f"_{args.feature_type}"
    output_base = output_dir / f"{args.method}_seed{args.seed}{feature_suffix}"

    # Generate plots
    if args.split:
        # Single split plot
        output_path = Path(f"{output_base}_{args.split}.png")
        plot_embedding_2d(X_embedded, meta, output_path, split=args.split,
                          method_name=method_upper,
                          title_prefix=f"{args.feature_type.upper()} - ")
    else:
        # All splits - individual plots
        for split_name in ['train', 'val', 'test']:
            output_path = Path(f"{output_base}_{split_name}.png")
            plot_embedding_2d(X_embedded, meta, output_path, split=split_name,
                              method_name=method_upper,
                              title_prefix=f"{args.feature_type.upper()} - ")

        # Combined plot with all splits
        output_path = Path(f"{output_base}_all.png")
        plot_embedding_by_split(X_embedded, meta, output_path, method_name=method_upper,
                               title_prefix=f"{args.feature_type.upper()} - ")

    # Separation metrics (only for test set)
    output_metrics = Path(f"{output_base}_metrics.png")
    plot_separation_metrics(X_embedded, meta, output_metrics, method_name=method_upper)

    print("\n" + "="*60)
    print(f"{args.method.upper()} analysis completed!")
    print(f"Plots saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
