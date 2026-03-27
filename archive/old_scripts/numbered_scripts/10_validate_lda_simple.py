"""
化学空间可视化 - 简化的LDA + t-SNE
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import TruncatedSVD
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Paths, DatasetConfig


def smiles_to_morgan(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main():
    print("=" * 70)
    print("化学空间可视化 - LDA + t-SNE")
    print("=" * 70)

    paths = Paths()
    dataset_cfg = DatasetConfig()

    # 1. 加载B3DB数据
    print("\n[1] 加载B3DB数据...")
    b3db_path = paths.data_raw / dataset_cfg.filename
    df = pd.read_csv(b3db_path, sep="\t")
    df = df[df[dataset_cfg.group_col].isin(['A', 'B'])].reset_index(drop=True)
    df['y_cls'] = df[dataset_cfg.bbb_col].map({'BBB+': 1, 'BBB-': 0})

    print(f"BBB+: {(df['y_cls'] == 1).sum()}, BBB-: {(df['y_cls'] == 0).sum()}")

    # 2. 计算指纹
    print("\n[2] 计算指纹...")
    X_list = []
    y_list = []

    for i, row in df.iterrows():
        fp = smiles_to_morgan(row['SMILES'])
        if fp is not None:
            X_list.append(fp)
            y_list.append(row['y_cls'])

    X = np.array(X_list)
    y = np.array(y_list)

    # 3. 加载生成的分子
    print("\n[3] 加载生成分子...")
    gen_path = PROJECT_ROOT / "outputs" / "generated_molecules" / "bbb_positive_molecules.csv"
    df_gen = pd.read_csv(gen_path)

    X_gen_list = []
    for smi in df_gen['SMILES']:
        fp = smiles_to_morgan(smi)
        if fp is not None:
            X_gen_list.append(fp)
    X_gen = np.array(X_gen_list)

    print(f"生成分子: {len(X_gen)}")

    # 4. 合并数据
    X_all = np.vstack([X, X_gen])
    labels = np.concatenate([y, np.ones(len(X_gen), dtype=int) * 2])

    print(f"\n总分子: {len(X_all)}")

    # 5. 先PCA
    print("\n[4] PCA降维...")
    pca = TruncatedSVD(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_all)

    # 6. t-SNE
    print("\n[5] t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_pca)

    # 7. LDA
    print("\n[6] LDA降维...")
    # 用BBB+/BBB-标签训练LDA
    lda = LDA(n_components=1)
    # 只用原始数据训练
    X_lda = lda.fit_transform(X_pca[:len(X)], y)
    # 预测所有数据（包括生成分子）
    X_lda_all = lda.decision_function(X_pca)

    # 8. 可视化
    print("\n[7] 生成图片...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # t-SNE
    ax = axes[0]
    mask_minus = labels == 0
    mask_plus = labels == 1
    mask_gen = labels == 2

    ax.scatter(X_tsne[mask_minus, 0], X_tsne[mask_minus, 1],
              c='#CCCCCC', s=15, alpha=0.5, label=f'BBB- (n={mask_minus.sum()})')
    ax.scatter(X_tsne[mask_plus, 0], X_tsne[mask_plus, 1],
              c='#2E86AB', s=20, alpha=0.6, label=f'BBB+ (n={mask_plus.sum()})')
    ax.scatter(X_tsne[mask_gen, 0], X_tsne[mask_gen, 1],
              c='#E94F37', s=200, marker='*', edgecolors='darkred',
              linewidths=1.5, label=f'Generated (n={mask_gen.sum()})', zorder=10)

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE (Unsupervised)\nPreserves Local Structure', fontsize=13)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # LDA
    ax = axes[1]
    ax.scatter(X_lda_all[mask_minus], np.zeros(mask_minus.sum()),
              c='#CCCCCC', s=15, alpha=0.5, label=f'BBB- (n={mask_minus.sum()})')
    ax.scatter(X_lda_all[mask_plus], np.zeros(mask_plus.sum()),
              c='#2E86AB', s=20, alpha=0.6, label=f'BBB+ (n={mask_plus.sum()})')
    ax.scatter(X_lda_all[mask_gen], np.zeros(mask_gen.sum()),
              c='#E94F37', s=200, marker='*', edgecolors='darkred',
              linewidths=1.5, label=f'Generated (n={mask_gen.sum()})', zorder=10)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('LDA Decision Function', fontsize=12)
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_title('LDA (Supervised)\nMaximizes BBB+/BBB- Separation', fontsize=13)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = PROJECT_ROOT / "outputs" / "generated_molecules" / "chemical_space_lda.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n图片保存: {output_path}")

    # 9. 统计
    print("\n" + "=" * 70)
    print("统计结果")
    print("=" * 70)

    # LDA决策函数值
    gen_lda = X_lda_all[mask_gen]
    bbb_plus_lda = X_lda_all[mask_plus]
    bbb_minus_lda = X_lda_all[mask_minus]

    print(f"\nLDA决策函数值 (越高越可能是BBB+):")
    print(f"  生成分子: {gen_lda.mean():.3f} ± {gen_lda.std():.3f}")
    print(f"  BBB+平均: {bbb_plus_lda.mean():.3f}")
    print(f"  BBB-平均: {bbb_minus_lda.mean():.3f}")

    # 判断
    in_range = (gen_lda > bbb_minus_lda.mean()) & (gen_lda < bbb_plus_lda.mean() * 1.5)
    print(f"\n落在BBB+/BBB-之间: {in_range.sum()}/{len(in_range)} ({in_range.sum()/len(in_range)*100:.0f}%)")

    # 生成的分子详情
    print("\n" + "=" * 70)
    print("生成分子详情")
    print("=" * 70)
    print(df_gen[['SMILES', 'BBB_prob', 'QED', 'SA']].to_string(index=False))

    print("\n完成!")


if __name__ == "__main__":
    main()
