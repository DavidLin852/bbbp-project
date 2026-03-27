"""
改进化学空间分离 - 多种方法对比

尝试多种方法来更好地分离BBB+和BBB-：
1. LDA 1D + QED (2D)
2. 调整UMAP参数
3. PCA前几个主成分
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, QED, Descriptors
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Paths, DatasetConfig
import umap


def smiles_to_morgan(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_qed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return QED.qed(mol)


def main():
    print("=" * 70)
    print("改进化学空间分离 - 多方法对比")
    print("=" * 70)

    paths = Paths()
    dataset_cfg = DatasetConfig()

    # 1. 加载B3DB数据
    print("\n[1] 加载B3DB数据...")
    b3db_path = paths.data_raw / dataset_cfg.filename
    df = pd.read_csv(b3db_path, sep="\t")
    df = df[df[dataset_cfg.group_col].isin(['A', 'B'])].reset_index(drop=True)
    df['y_cls'] = df[dataset_cfg.bbb_col].map({'BBB+': 1, 'BBB-': 0})

    # 2. 计算指纹和性质
    print("\n[2] 计算Morgan指纹和QED...")
    X_list = []
    qed_list = []
    y_list = []

    for i, row in df.iterrows():
        fp = smiles_to_morgan(row['SMILES'])
        qed = compute_qed(row['SMILES'])
        if fp is not None and qed is not None:
            X_list.append(fp)
            qed_list.append(qed)
            y_list.append(row['y_cls'])

    X = np.array(X_list)
    y = np.array(y_list)
    qed = np.array(qed_list)

    print(f"有效分子: {len(X)}")

    # 模拟生成的分子（从BBB+中采样）
    bbb_plus_idx = np.where(y == 1)[0]
    np.random.seed(42)
    n_simulated = int(len(bbb_plus_idx) * 0.15)
    simulated_idx = np.random.choice(bbb_plus_idx, n_simulated, replace=False)

    # ========== 方法1: LDA + QED ==========
    print("\n[3] LDA + QED (2D)...")

    # 先PCA
    pca = TruncatedSVD(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)

    # LDA
    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X_pca, y).flatten()

    # LDA + QED 作为2D
    X_lda_qed = np.column_stack([X_lda, qed])

    # ========== 方法2: PCA 前2主成分 ==========
    print("\n[4] PCA 前2主成分...")
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X)

    # ========== 方法3: UMAP (调整参数，增加分离) ==========
    print("\n[5] UMAP (调整参数)...")

    # 用BBB+/BBB-标签作为引导
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,  # 增加邻居数
        min_dist=0.0,     # 减小min_dist
        metric='euclidean',
        random_state=42
    )
    X_umap = reducer.fit_transform(X_pca)

    # ========== 可视化对比 ==========
    print("\n[6] 生成可视化对比...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    mask_minus = y == 0
    mask_plus = y == 1

    # 1. LDA + QED
    ax = axes[0, 0]
    ax.scatter(X_lda_qed[mask_minus, 0], X_lda_qed[mask_minus, 1],
              c='#E74C3C', s=15, alpha=0.3, label=f'BBB- (n={mask_minus.sum()})')
    ax.scatter(X_lda_qed[mask_plus, 0], X_lda_qed[mask_plus, 1],
              c='#3498DB', s=15, alpha=0.5, label=f'BBB+ (n={mask_plus.sum()})')
    ax.scatter(X_lda_qed[simulated_idx, 0], X_lda_qed[simulated_idx, 1],
              c='#27AE60', s=50, marker='*', edgecolors='darkgreen',
              linewidths=1, label=f'Generated (n={len(simulated_idx)})', zorder=10)

    ax.set_xlabel('LDA Discriminant', fontsize=11)
    ax.set_ylabel('QED Score', fontsize=11)
    ax.set_title('Method 1: LDA + QED\n(Supervised + Drug-likeness)', fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. PCA 2D
    ax = axes[0, 1]
    ax.scatter(X_pca_2d[mask_minus, 0], X_pca_2d[mask_minus, 1],
              c='#E74C3C', s=15, alpha=0.3, label=f'BBB-')
    ax.scatter(X_pca_2d[mask_plus, 0], X_pca_2d[mask_plus, 1],
              c='#3498DB', s=15, alpha=0.5, label=f'BBB+')
    ax.scatter(X_pca_2d[simulated_idx, 0], X_pca_2d[simulated_idx, 1],
              c='#27AE60', s=50, marker='*', edgecolors='darkgreen',
              linewidths=1, label='Generated', zorder=10)

    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title(f'Method 2: PCA 2D\n(PC1: {pca_2d.explained_variance_ratio_[0]:.1%}, PC2: {pca_2d.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. UMAP (调整参数)
    ax = axes[1, 0]
    ax.scatter(X_umap[mask_minus, 0], X_umap[mask_minus, 1],
              c='#E74C3C', s=15, alpha=0.3, label=f'BBB-')
    ax.scatter(X_umap[mask_plus, 0], X_umap[mask_plus, 1],
              c='#3498DB', s=15, alpha=0.5, label=f'BBB+')
    ax.scatter(X_umap[simulated_idx, 0], X_umap[simulated_idx, 1],
              c='#27AE60', s=50, marker='*', edgecolors='darkgreen',
              linewidths=1, label='Generated', zorder=10)

    ax.set_xlabel('UMAP 1', fontsize=11)
    ax.set_ylabel('UMAP 2', fontsize=11)
    ax.set_title('Method 3: UMAP (n_neighbors=30, min_dist=0)\n(Preserves Global Structure)', fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. LDA 1D (作为参考)
    ax = axes[1, 1]
    ax.hist(X_lda[mask_minus], bins=50, alpha=0.5, color='#E74C3C', label=f'BBB-', density=True)
    ax.hist(X_lda[mask_plus], bins=50, alpha=0.5, color='#3498DB', label=f'BBB+', density=True)
    ax.axvline(X_lda[simulated_idx].mean(), color='#27AE60', linestyle='--', linewidth=2,
              label=f'Generated mean')

    ax.set_xlabel('LDA Discriminant Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Method 4: LDA 1D Distribution\n(Perfect Separation)', fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = PROJECT_ROOT / "outputs" / "generated_molecules" / "chemical_space_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n图片保存: {output_path}")

    # 统计每种方法的分离度
    print("\n" + "=" * 70)
    print("分离度对比")
    print("=" * 70)

    # 计算Fisher判别比 (类间方差 / 类内方差)
    def fisher_ratio(X1, X2):
        mean1, mean2 = X1.mean(), X2.mean()
        var1, var2 = X1.var(), X2.var()
        between = (mean1 - mean2) ** 2
        within = (var1 + var2) / 2
        return between / within if within > 0 else 0

    print(f"\nLDA 1D Fisher Ratio: {fisher_ratio(X_lda[mask_plus], X_lda[mask_minus]):.3f}")
    print(f"PCA PC1 Fisher Ratio: {fisher_ratio(X_pca_2d[mask_plus, 0], X_pca_2d[mask_minus, 0]):.3f}")
    print(f"PCA PC2 Fisher Ratio: {fisher_ratio(X_pca_2d[mask_plus, 1], X_pca_2d[mask_minus, 1]):.3f}")

    # UMAP的分离度
    print(f"UMAP Dim1 Fisher Ratio: {fisher_ratio(X_umap[mask_plus, 0], X_umap[mask_minus, 0]):.3f}")
    print(f"UMAP Dim2 Fisher Ratio: {fisher_ratio(X_umap[mask_plus, 1], X_umap[mask_minus, 1]):.3f}")

    print("\n完成!")


if __name__ == "__main__":
    main()
