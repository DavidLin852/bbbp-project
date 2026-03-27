"""
验证生成的分子是否落在BBB+分子的化学空间中

使用UMAP/t-SNE对Morgan指纹进行降维，将生成的分子投影到同一空间
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.sparse import vstack
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
    print("=" * 60)
    print("化学空间可视化 - 验证生成分子")
    print("=" * 60)

    paths = Paths()
    dataset_cfg = DatasetConfig()

    # 1. 加载训练数据
    print("\n[1] 加载B3DB数据...")
    b3db_path = paths.data_raw / dataset_cfg.filename
    df = pd.read_csv(b3db_path, sep="\t")

    # 过滤A,B组
    df = df[df[dataset_cfg.group_col].isin(['A', 'B'])].reset_index(drop=True)
    df['y_cls'] = df[dataset_cfg.bbb_col].map({'BBB+': 1, 'BBB-': 0})

    print(f"BBB+ 分子数: {(df['y_cls'] == 1).sum()}")
    print(f"BBB- 分子数: {(df['y_cls'] == 0).sum()}")

    # 2. 计算Morgan指纹
    print("\n[2] 计算Morgan指纹...")
    X_list = []
    valid_idx = []

    for i, smi in enumerate(df['SMILES']):
        fp = smiles_to_morgan(smi)
        if fp is not None:
            X_list.append(fp)
            valid_idx.append(i)

    X = np.array(X_list)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)

    print(f"有效分子数: {len(X)}")

    # 3. 加载生成的分子
    print("\n[3] 加载生成的BBB+分子...")
    gen_path = PROJECT_ROOT / "outputs" / "generated_molecules" / "bbb_positive_molecules.csv"
    if gen_path.exists():
        df_gen = pd.read_csv(gen_path)
        print(f"生成分子数: {len(df_gen)}")
    else:
        print("未找到生成的分子文件")
        return

    # 计算生成分子的指纹
    X_gen = []
    gen_valid = []
    for i, smi in enumerate(df_gen['SMILES']):
        fp = smiles_to_morgan(smi)
        if fp is not None:
            X_gen.append(fp)
            gen_valid.append(i)

    X_gen = np.array(X_gen)
    df_gen_valid = df_gen.iloc[gen_valid].reset_index(drop=True)
    print(f"有效生成分子数: {len(X_gen)}")

    # 4. 合并数据
    print("\n[4] 合并数据...")
    X_all = np.vstack([X, X_gen])
    labels = np.concatenate([
        df_valid['y_cls'].values,  # 0=BBB-, 1=BBB+
        np.ones(len(X_gen), dtype=int) * 2  # 2=生成分子
    ])

    print(f"总分子数: {len(X_all)}")
    print(f"  - BBB-: {(labels == 0).sum()}")
    print(f"  - BBB+: {(labels == 1).sum()}")
    print(f"  - 生成: {(labels == 2).sum()}")

    # 5. 降维 (t-SNE)
    print("\n[5] t-SNE降维...")
    # 先PCA降维加速
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_svd = svd.fit_transform(X_all)

    # 再用t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X_svd)

    print("降维完成!")

    # 6. 可视化
    print("\n[6] 生成可视化...")
    plt.figure(figsize=(12, 10))

    # 背景: BBB- (灰色)
    mask_minus = labels == 0
    plt.scatter(X_2d[mask_minus, 0], X_2d[mask_minus, 1],
                c='lightgray', s=20, alpha=0.5, label='BBB-')

    # 背景: BBB+ (蓝色)
    mask_plus = labels == 1
    plt.scatter(X_2d[mask_plus, 0], X_2d[mask_plus, 1],
                c='steelblue', s=30, alpha=0.6, label='BBB+ (训练)')

    # 前景: 生成分子 (红色星号)
    mask_gen = labels == 2
    plt.scatter(X_2d[mask_gen, 0], X_2d[mask_gen, 1],
                c='red', s=150, marker='*', edgecolors='darkred',
                linewidths=1, label='Generated')

    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.title('BBB+ 化学空间验证\nGenerated Molecules in BBB+ Chemical Space', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)

    # 保存图片
    output_path = PROJECT_ROOT / "outputs" / "generated_molecules" / "chemical_space_validation.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图片已保存: {output_path}")

    # 7. 计算统计信息
    print("\n[7] 统计信息...")

    # 计算生成分子与BBB+中心的距离
    bbb_plus_centroid = X_2d[mask_plus].mean(axis=0)
    gen_centroid = X_2d[mask_gen].mean(axis=0)

    # 计算每个生成分子到BBB+中心的距离
    distances = np.sqrt(((X_2d[mask_gen] - bbb_plus_centroid) ** 2).sum(axis=1))

    # 计算BBB+分子的平均距离作为参考
    bbb_plus_distances = np.sqrt(((X_2d[mask_plus] - bbb_plus_centroid) ** 2).sum(axis=1))

    print(f"\nBBB+分子区域:")
    print(f"  - 中心坐标: ({bbb_plus_centroid[0]:.1f}, {bbb_plus_centroid[1]:.1f})")
    print(f"  - 平均分散度: {bbb_plus_distances.mean():.1f} ± {bbb_plus_distances.std():.1f}")

    print(f"\n生成分子:")
    print(f"  - 中心坐标: ({gen_centroid[0]:.1f}, {gen_centroid[1]:.1f})")
    print(f"  - 到BBB+中心距离: {distances.mean():.1f} ± {distances.std():.1f}")
    print(f"  - 距离BBB+平均分散度: {distances.mean() / bbb_plus_distances.mean():.2f}x")

    # 判断是否在合理范围内
    threshold = bbb_plus_distances.mean() + 2 * bbb_plus_distances.std()
    in_range = distances < threshold
    print(f"\n落在BBB+化学空间内: {in_range.sum()}/{len(in_range)} ({in_range.sum()/len(in_range)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
