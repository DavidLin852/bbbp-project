"""
Compare dimensionality reduction methods and feature types.

This script creates a comprehensive comparison of different combinations
of reduction methods and feature types for BBB permeability analysis.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean
import json
from rdkit import Chem

from src.config import Paths, DatasetConfig


def load_smarts_patterns(smarts_json: Path):
    """Load SMARTS patterns from JSON file."""
    with open(smarts_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    patterns = []
    for item in data:
        if isinstance(item, dict) and "name" in item and "smarts" in item:
            patterns.append((item["name"], item["smarts"]))

    return patterns


def compute_smarts_features(smiles_list, smarts_patterns):
    """Compute SMARTS binary features."""
    n_patterns = len(smarts_patterns)
    features = np.zeros((len(smiles_list), n_patterns), dtype=np.int8)

    for i, smiles in enumerate(smiles_list):
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

    return features


def calculate_metrics(X_embedded, y_true, method_name, feature_type):
    """Calculate separation metrics.

    Returns:
        dict with metrics
    """
    # Centroid distance
    bbb_plus_mask = y_true == 1
    bbb_minus_mask = y_true == 0

    centroid_plus = X_embedded[bbb_plus_mask].mean(axis=0)
    centroid_minus = X_embedded[bbb_minus_mask].mean(axis=0)

    centroid_distance = euclidean(centroid_plus, centroid_minus)

    # Spread within each class
    spread_plus = np.mean([euclidean(X_embedded[i], centroid_plus)
                          for i in np.where(bbb_plus_mask)[0]])
    spread_minus = np.mean([euclidean(X_embedded[i], centroid_minus)
                           for i in np.where(bbb_minus_mask)[0]])

    # Silhouette score
    silhouette = silhouette_score(X_embedded, y_true)

    return {
        'method': method_name,
        'feature_type': feature_type,
        'centroid_distance': centroid_distance,
        'spread_plus': spread_plus,
        'spread_minus': spread_minus,
        'silhouette': silhouette
    }


def run_comparison(seed: int = 0):
    """Run comprehensive comparison of methods and features."""
    paths = Paths()
    feature_dir = paths.features / f"seed_{seed}"

    # Load metadata
    meta_path = feature_dir / "meta.csv"
    meta = pd.read_csv(meta_path)

    # Get test set
    test_mask = meta['split'] == 'test'
    meta_test = meta[test_mask].reset_index(drop=True)
    smiles_test = meta_test['SMILES'].tolist()
    y_test = meta_test['y_cls'].values

    print(f"Test set size: {len(y_test)} ({(y_test==1).sum()} BBB+, {(y_test==0).sum()} BBB-)")
    print()

    # Define feature types to test
    feature_configs = {
        'Morgan': 'morgan',
        'Descriptors': 'descriptors',
        'SMARTS': 'smarts',
        'Combined (Morgan+Desc)': 'combined',
        'All Features': 'all'
    }

    # Load features
    print("Loading features...")
    X_morgan = load_npz(feature_dir / "morgan_2048.npz").toarray()[test_mask]
    X_desc = pd.read_csv(feature_dir / "descriptors.csv", index_col=0).values[test_mask]

    # Compute SMARTS features
    print("Computing SMARTS features...")
    smarts_json = paths.root / "assets" / "smarts" / "bbb_smarts_v1.json"
    smarts_patterns = load_smarts_patterns(smarts_json)
    X_smarts = compute_smarts_features(smiles_test, smarts_patterns)

    # Combine features
    X_combined = np.hstack([X_morgan, X_desc])
    X_all = np.hstack([X_morgan, X_desc, X_smarts])

    features_dict = {
        'Morgan': X_morgan,
        'Descriptors': X_desc,
        'SMARTS': X_smarts,
        'Combined (Morgan+Desc)': X_combined,
        'All Features': X_all
    }

    # Test different reduction methods
    methods = ['PCA', 't-SNE']
    results = []

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running {method}...")
        print('='*60)

        for feat_name, X in features_dict.items():
            print(f"\n{feat_name} ({X.shape[1]} features)...")

            # Apply dimensionality reduction
            if method == 'PCA':
                reducer = PCA(n_components=2, random_state=42)
                X_embedded = reducer.fit_transform(X)
                var_explained = reducer.explained_variance_ratio_.sum()
                print(f"  Explained variance: {var_explained:.3f}")
            else:  # t-SNE
                reducer = TSNE(
                    n_components=2,
                    perplexity=30,
                    max_iter=1000,
                    random_state=42,
                    verbose=0
                )
                X_embedded = reducer.fit_transform(X)

            # Calculate metrics
            metrics = calculate_metrics(X_embedded, y_test, method, feat_name)
            results.append(metrics)

            print(f"  Centroid Distance: {metrics['centroid_distance']:.3f}")
            print(f"  Silhouette Score: {metrics['silhouette']:.3f}")

    # Create comparison visualization
    create_comparison_plots(results, seed)


def create_comparison_plots(results, seed):
    """Create comparison plots."""
    paths = Paths()
    output_dir = paths.figures / "dim_reduction"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Silhouette Score comparison
    ax1 = axes[0, 0]
    pivot = df.pivot(index='feature_type', columns='method', values='silhouette')
    pivot.plot(kind='bar', ax=ax1, color=['#3498db', '#e74c3c'], alpha=0.7,
              edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Silhouette Score', fontsize=11, fontname='Times New Roman')
    ax1.set_title('Silhouette Score (higher is better)', fontsize=12,
                  fontname='Times New Roman', fontweight='bold')
    ax1.set_xlabel('Feature Type', fontsize=11, fontname='Times New Roman')
    ax1.legend(fontsize=10, prop={'family': 'Times New Roman'})
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right',
             fontname='Times New Roman')

    # Plot 2: Centroid Distance comparison
    ax2 = axes[0, 1]
    pivot = df.pivot(index='feature_type', columns='method', values='centroid_distance')
    pivot.plot(kind='bar', ax=ax2, color=['#3498db', '#e74c3c'], alpha=0.7,
              edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Centroid Distance', fontsize=11, fontname='Times New Roman')
    ax2.set_title('Centroid Distance (higher is better)', fontsize=12,
                  fontname='Times New Roman', fontweight='bold')
    ax2.set_xlabel('Feature Type', fontsize=11, fontname='Times New Roman')
    ax2.legend(fontsize=10, prop={'family': 'Times New Roman'})
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right',
             fontname='Times New Roman')

    # Plot 3: Spread comparison
    ax3 = axes[1, 0]
    x = np.arange(len(df['feature_type'].unique()))
    width = 0.35

    for i, method in enumerate(['PCA', 't-SNE']):
        df_method = df[df['method'] == method]
        offset = width * i
        ax3.bar(x + offset, df_method['spread_plus'], width,
               label=f'{method} BBB+', alpha=0.7, edgecolor='black', linewidth=1)
        ax3.bar(x + offset, df_method['spread_minus'], width,
               label=f'{method} BBB-', alpha=0.7, edgecolor='black', linewidth=1)

    ax3.set_ylabel('Spread', fontsize=11, fontname='Times New Roman')
    ax3.set_title('Within-Class Spread (lower is better)', fontsize=12,
                  fontname='Times New Roman', fontweight='bold')
    ax3.set_xlabel('Feature Type', fontsize=11, fontname='Times New Roman')
    ax3.set_xticks(x + width / 2)
    ax3.set_xticklabels(df['feature_type'].unique(), rotation=45, ha='right',
                        fontname='Times New Roman')
    ax3.legend(fontsize=9, prop={'family': 'Times New Roman'}, ncol=2)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    # Create summary table
    summary = df.pivot(index='feature_type', columns='method', values='silhouette')
    table_data = []
    for idx in summary.index:
        row = [idx]
        for col in summary.columns:
            val = summary.loc[idx, col]
            row.append(f'{val:.3f}')
        table_data.append(row)

    # Add best performing row
    table = ax4.table(cellText=table_data,
                     colLabels=['Feature Type', 'PCA', 't-SNE'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.4, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#3498db')
                cell.set_text_props(weight='bold', color='white',
                                   fontname='Times New Roman')
            else:
                cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
                cell.set_text_props(fontname='Times New Roman')

    ax4.set_title('Silhouette Score Summary\n(Test Set)', fontsize=12,
                  fontname='Times New Roman', fontweight='bold', pad=20)

    plt.tight_layout()

    # Save plot
    output_path = output_dir / f"comparison_seed{seed}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_path}")
    plt.close()

    # Print summary table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY - Silhouette Scores (Test Set)")
    print("="*80)
    print(f"{'Feature Type':<30} {'PCA':>10} {'t-SNE':>10}")
    print("-"*80)

    for feat_type in df['feature_type'].unique():
        pca_score = df[(df['method'] == 'PCA') & (df['feature_type'] == feat_type)]['silhouette'].values[0]
        tsne_score = df[(df['method'] == 't-SNE') & (df['feature_type'] == feat_type)]['silhouette'].values[0]
        print(f"{feat_type:<30} {pca_score:>10.3f} {tsne_score:>10.3f}")

    print("="*80)

    # Find best performing combination
    best_idx = df['silhouette'].idxmax()
    best_row = df.loc[best_idx]
    print(f"\nBEST PERFORMING: {best_row['method']} with {best_row['feature_type']}")
    print(f"  Silhouette Score: {best_row['silhouette']:.3f}")
    print("="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare dimensionality reduction methods")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    run_comparison(seed=args.seed)
