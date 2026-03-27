"""
t-SNE visualization of BBB+ vs BBB- molecules in feature space.

This script performs t-SNE dimensionality reduction on molecular features
and visualizes the clustering of BBB permeable vs non-permeable compounds.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

from src.config import Paths, DatasetConfig


def load_features(feature_dir: Path, feature_type: str = "morgan"):
    """Load molecular features and metadata.

    Args:
        feature_dir: Directory containing feature files
        feature_type: Type of features to load ('morgan', 'descriptors', 'combined')

    Returns:
        X: Feature matrix (sparse or dense)
        meta: Metadata DataFrame with SMILES, labels, split info
    """
    # Load metadata
    meta_path = feature_dir / "meta.csv"
    meta = pd.read_csv(meta_path)

    # Load features
    if feature_type == "morgan":
        from scipy.sparse import load_npz
        X = load_npz(feature_dir / "morgan_2048.npz")
    elif feature_type == "descriptors":
        X = pd.read_csv(feature_dir / "descriptors.csv", index_col=0).values
    elif feature_type == "combined":
        from scipy.sparse import load_npz, hstack
        X_morgan = load_npz(feature_dir / "morgan_2048.npz")
        X_desc = pd.read_csv(feature_dir / "descriptors.csv", index_col=0).values
        X = hstack([X_morgan, X_desc])
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    return X, meta


def run_tsne(X, perplexity: int = 30, n_iter: int = 1000, random_state: int = 42):
    """Run t-SNE dimensionality reduction.

    Args:
        X: Feature matrix
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        random_state: Random seed

    Returns:
        X_embedded: 2D embedded coordinates
    """
    # Convert sparse to dense if needed (t-SNE doesn't support sparse)
    from scipy.sparse import issparse
    if issparse(X):
        X = X.toarray()

    # Run t-SNE
    print(f"Running t-SNE with perplexity={perplexity}, max_iter={n_iter}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        verbose=1,
        n_jobs=-1
    )
    X_embedded = tsne.fit_transform(X)

    return X_embedded


def plot_tsne(X_embedded, meta, output_path: Path, split: str = None,
              title_prefix: str = ""):
    """Create t-SNE scatter plot.

    Args:
        X_embedded: 2D t-SNE coordinates
        meta: Metadata DataFrame
        output_path: Path to save the plot
        split: If specified, only plot this split ('train', 'val', 'test')
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

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot each class separately for better legend
    for label, color, marker in [('BBB+', '#2ecc71', 'o'), ('BBB-', '#e74c3c', 's')]:
        mask = labels == label
        plt.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            c=color,
            label=label,
            alpha=0.6,
            s=50,
            marker=marker,
            edgecolors='black',
            linewidth=0.5
        )

    plt.xlabel('t-SNE Dimension 1', fontsize=12, fontname='Times New Roman')
    plt.ylabel('t-SNE Dimension 2', fontsize=12, fontname='Times New Roman')
    title = f"{title_prefix}BBB Permeability - t-SNE Visualization"
    if split:
        title += f" ({split.upper()} set)"
    plt.title(title, fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.legend(fontsize=11, prop={'family': 'Times New Roman'})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_tsne_by_split(X_embedded, meta, output_path: Path, title_prefix: str = ""):
    """Create t-SNE scatter plot with all splits shown together.

    Args:
        X_embedded: 2D t-SNE coordinates
        meta: Metadata DataFrame
        output_path: Path to save the plot
        title_prefix: Prefix for plot title
    """
    # Create labels
    labels = meta['y_cls'].map({1: 'BBB+', 0: 'BBB-'})
    splits = meta['split']

    # Create plot
    plt.figure(figsize=(12, 10))

    # Plot each combination of class and split
    from itertools import product

    colors = {'BBB+': '#2ecc71', 'BBB-': '#e74c3c'}
    markers = {'train': 'o', 'val': 's', 'test': '^'}
    alphas = {'train': 0.4, 'val': 0.6, 'test': 0.8}

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

    plt.xlabel('t-SNE Dimension 1', fontsize=12, fontname='Times New Roman')
    plt.ylabel('t-SNE Dimension 2', fontsize=12, fontname='Times New Roman')
    plt.title(f"{title_prefix}BBB Permeability - t-SNE Visualization (All Splits)",
              fontsize=14, fontname='Times New Roman', fontweight='bold')
    plt.legend(fontsize=10, prop={'family': 'Times New Roman'}, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def print_statistics(meta):
    """Print dataset statistics.

    Args:
        meta: Metadata DataFrame
    """
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
    parser = argparse.ArgumentParser(description="t-SNE analysis of BBB permeability")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data split")
    parser.add_argument("--feature_type", type=str, default="morgan",
                        choices=["morgan", "descriptors", "combined"],
                        help="Type of features to use for t-SNE")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="t-SNE perplexity (default: 30, recommended: 5-50)")
    parser.add_argument("--n_iter", type=int, default=1000,
                        help="t-SNE number of iterations (default: 1000)")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"],
                        help="Only visualize specific split (default: all)")
    parser.add_argument("--tsne_seed", type=int, default=42,
                        help="Random seed for t-SNE initialization")

    args = parser.parse_args()

    # Setup paths
    paths = Paths()
    feature_dir = paths.features / f"seed_{args.seed}"
    output_dir = paths.figures / "tsne"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if features exist
    if not feature_dir.exists():
        print(f"Error: Feature directory not found: {feature_dir}")
        print("Please run featurization first: python scripts/02_featurize_all.py --seed {args.seed}")
        return

    # Load features and metadata
    print(f"Loading features from {feature_dir}...")
    X, meta = load_features(feature_dir, args.feature_type)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")

    # Print statistics
    print_statistics(meta)

    # Run t-SNE
    X_embedded = run_tsne(X, perplexity=args.perplexity, n_iter=args.n_iter,
                         random_state=args.tsne_seed)

    # Generate plots
    feature_suffix = f"_{args.feature_type}" if args.feature_type != "morgan" else ""

    if args.split:
        # Single split plot
        output_path = output_dir / f"tsne_{args.split}_seed{args.seed}{feature_suffix}.png"
        plot_tsne(X_embedded, meta, output_path, split=args.split,
                 title_prefix=f"{args.feature_type.upper()} - ")
    else:
        # All splits - individual plots
        for split_name in ['train', 'val', 'test']:
            output_path = output_dir / f"tsne_{split_name}_seed{args.seed}{feature_suffix}.png"
            plot_tsne(X_embedded, meta, output_path, split=split_name,
                     title_prefix=f"{args.feature_type.upper()} - ")

        # Combined plot with all splits
        output_path = output_dir / f"tsne_all_seed{args.seed}{feature_suffix}.png"
        plot_tsne_by_split(X_embedded, meta, output_path, title_prefix=f"{args.feature_type.upper()} - ")

    print("\nt-SNE analysis completed!")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
