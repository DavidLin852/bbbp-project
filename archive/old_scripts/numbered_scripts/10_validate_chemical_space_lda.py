"""
化学空间可视化 - 使用LDA降维

LDA是有监督的降维方法，可以更好地区分BBB+和BBB-两类
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Paths, DatasetConfig
from src.multi_model_predictor import MultiModelPredictor, EnsembleStrategy


def smiles_to_morgan(smiles, n_bits=2048):
    """Convert SMILES to Morgan fingerprint"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main():
    print("=" * 70)
    print("化学空间可视化 - LDA降维")
    print("=" * 70)

    paths = Paths()
    dataset_cfg = DatasetConfig()

    # 1. 加载B3DB数据
    print("\n[1] 加载B3DB数据...")
    b3db_path = paths.data_raw / dataset_cfg.filename
    df = pd.read_csv(b3db_path, sep="\t")
    df = df[df[dataset_cfg.group_col].isin(['A', 'B'])].reset_index(drop=True)
    df['y_cls'] = df[dataset_cfg.bbb_col].map({'BBB+': 1, 'BBB-': 0})

    print(f"BBB+ 分子: {(df['y_cls'] == 1).sum()}")
    print(f"BBB- 分子: {(df['y_cls'] == 0).sum()}")

    # 2. 计算Morgan指纹
    print("\n[2] 计算Morgan指纹...")
    X_list = []
    y_list = []
    smiles_list = []

    for i, row in df.iterrows():
        smi = row['SMILES']
        fp = smiles_to_morgan(smi)
        if fp is not None:
            X_list.append(fp)
            y_list.append(row['y_cls'])
            smiles_list.append(smi)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"有效分子: {len(X)}")

    # 3. 加载生成的分子
    print("\n[3] 加载生成的BBB+分子...")
    gen_path = PROJECT_ROOT / "outputs" / "generated_molecules" / "bbb_positive_molecules.csv"
    if gen_path.exists():
        df_gen = pd.read_csv(gen_path)
    else:
        print("未找到生成分子文件")
        return

    X_gen_list = []
    gen_smiles = []
    for smi in df_gen['SMILES']:
        fp = smiles_to_morgan(smi)
        if fp is not None:
            X_gen_list.append(fp)
            gen_smiles.append(smi)

    X_gen = np.array(X_gen_list) if X_gen_list else None

    print(f"生成分子: {len(gen_smiles)}")

    # 4. 先PCA降维
    print("\n[4] PCA降维到50维...")
    pca = TruncatedSVD(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)

    X_gen_pca = pca.transform(X_gen) if X_gen is not None else None

    # 5. LDA降维 (有监督) - LDA最多只能降到1维（2类）
    print("\n[5] LDA降维到1维...")
    lda = LDA(n_components=1)
    X_lda_1d = lda.fit_transform(X_pca, y).flatten()

    # 用LDA的判别函数值作为第二维（就是预测的log-likelihood比）
    X_lda_2d = np.column_stack([X_lda_1d, lda.decision_function(X_pca)])

    if X_gen_pca is not None:
        X_gen_lda_1d = lda.transform(X_gen_pca).flatten()
        X_gen_lda = np.column_stack([X_gen_lda_1d, lda.decision_function(X_gen_pca)])
    else:
        X_gen_lda = None

    # 6. t-SNE (可选对比)
    print("\n[6] t-SNE降维 (对比)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_pca)

    if X_gen_pca is not None:
        X_gen_tsne = tsne.transform(X_gen_pca)
    else:
        X_gen_tsne = None

    # 7. 创建可视化
    print("\n[7] 生成可视化...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ===== LDA =====
    ax = axes[0]

    # BBB-
    mask_minus = y == 0
    ax.scatter(X_lda_2d[mask_minus, 0], X_lda_2d[mask_minus, 1],
              c='#CCCCCC', s=15, alpha=0.5, label=f'BBB- (n={mask_minus.sum()})')

    # BBB+
    mask_plus = y == 1
    ax.scatter(X_lda_2d[mask_plus, 0], X_lda_2d[mask_plus, 1],
              c='#2E86AB', s=20, alpha=0.6, label=f'BBB+ (n={mask_plus.sum()})')

    # 生成的分子
    if X_gen_lda is not None and len(X_gen_lda) > 0:
        ax.scatter(X_gen_lda[:, 0], X_gen_lda[:, 1],
                  c='#E94F37', s=200, marker='*', edgecolors='darkred',
                  linewidths=1.5, label=f'Generated (n={len(X_gen_lda)})',
                  zorder=10)

    ax.set_xlabel('LDA Discriminant Function', fontsize=12)
    ax.set_ylabel('Decision Function Value', fontsize=12)
    ax.set_title('LDA (Linear Discriminant Analysis)\nSupervised: Maximizes BBB+/BBB- Separation', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # ===== t-SNE =====
    ax = axes[1]

    # BBB-
    ax.scatter(X_tsne[mask_minus, 0], X_tsne[mask_minus, 1],
              c='#CCCCCC', s=15, alpha=0.5, label=f'BBB- (n={mask_minus.sum()})')

    # BBB+
    ax.scatter(X_tsne[mask_plus, 0], X_tsne[mask_plus, 1],
              c='#2E86AB', s=20, alpha=0.6, label=f'BBB+ (n={mask_plus.sum()})')

    # 生成的分子
    if X_gen_tsne is not None and len(X_gen_tsne) > 0:
        ax.scatter(X_gen_tsne[:, 0], X_gen_tsne[:, 1],
                  c='#E94F37', s=200, marker='*', edgecolors='darkred',
                  linewidths=1.5, label=f'Generated (n={len(X_gen_tsne)})',
                  zorder=10)

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE (t-Distributed Stochastic Neighbor Embedding)\nUnsupervised Dimensionality Reduction', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    output_path = PROJECT_ROOT / "outputs" / "generated_molecules" / "chemical_space_lda.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n图片已保存: {output_path}")

    # 8. 统计信息
    print("\n" + "=" * 70)
    print("统计信息")
    print("=" * 70)

    if X_gen_lda is not None:
        # LDA空间
        gen_centroid_lda = X_gen_lda.mean(axis=0)
        bbb_plus_centroid_lda = X_lda_2d[mask_plus].mean(axis=0)

        # 计算到BBB+中心的距离
        dist_lda = np.sqrt(((X_gen_lda - bbb_plus_centroid_lda) ** 2).sum(axis=1))
        bbb_plus_dists_lda = np.sqrt(((X_lda_2d[mask_plus] - bbb_plus_centroid_lda) ** 2).sum(axis=1))

        # t-SNE空间
        gen_centroid_tsne = X_gen_tsne.mean(axis=0)
        bbb_plus_centroid_tsne = X_tsne[mask_plus].mean(axis=0)
        dist_tsne = np.sqrt(((X_gen_tsne - bbb_plus_centroid_tsne) ** 2).sum(axis=1))
        bbb_plus_dists_tsne = np.sqrt(((X_tsne[mask_plus] - bbb_plus_centroid_tsne) ** 2).sum(axis=1))

        print(f"\nLDA空间:")
        print(f"  生成分子中心: ({gen_centroid_lda[0]:.2f}, {gen_centroid_lda[1]:.2f})")
        print(f"  BBB+中心: ({bbb_plus_centroid_lda[0]:.2f}, {bbb_plus_centroid_lda[1]:.2f})")
        print(f"  平均距离BBB+: {dist_lda.mean():.2f}")
        print(f"  BBB+分散度: {bbb_plus_dists_lda.mean():.2f}")

        print(f"\nt-SNE空间:")
        print(f"  生成分子中心: ({gen_centroid_tsne[0]:.1f}, {gen_centroid_tsne[1]:.1f})")
        print(f"  BBB+中心: ({bbb_plus_centroid_tsne[0]:.1f}, {bbb_plus_centroid_tsne[1]:.1f})")
        print(f"  平均距离BBB+: {dist_tsne.mean():.1f}")
        print(f"  BBB+分散度: {bbb_plus_dists_tsne.mean():.1f}")

    # 9. 列出生成的分子
    print("\n" + "=" * 70)
    print("生成的BBB+分子详情")
    print("=" * 70)

    df_gen['BBB_prob'] = df_gen['BBB_prob'].round(3)
    df_gen['QED'] = df_gen['QED'].round(3)
    df_gen['SA'] = df_gen['SA'].round(1)

    print(df_gen.to_string(index=False))

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
