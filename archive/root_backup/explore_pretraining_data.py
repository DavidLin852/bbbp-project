"""
ZINC20预训练数���探索与分析

分析预训练数据的：
1. 基本统计（分子数量、大小等）
2. 物理化学性质分布
3. 分子多样性分析
4. 预训练方法说明
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Draw
from rdkit.Chem import AllChem

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pretrain.zinc20_loader import compute_zinc_properties, ZINC20Property


def analyze_smiles_file(smiles_file: Path, sample_size: int = 5000):
    """分析SMILES文件中的分子"""
    print("\n" + "=" * 80)
    print("ZINC20预训练数据分析")
    print("=" * 80)

    # 加载数据
    df = pd.read_csv(smiles_file)
    print(f"\n数据文件: {smiles_file}")
    print(f"总分子数: {len(df):,}")

    # 计算属性
    print("\n计算分子属性...")
    props_list = []
    valid_smiles = []

    for smi in df['SMILES']:
        props = compute_zinc_properties(smi)
        if props is not None:
            props_list.append(props)
            valid_smiles.append(smi)

    print(f"有效分子数: {len(valid_smiles):,}")

    if len(props_list) == 0:
        print("ERROR: 没有有效分子!")
        return None

    # 转换为DataFrame
    props_df = pd.DataFrame([
        {
            'MW': p.mw,
            'logP': p.logp,
            'TPSA': p.tpsa,
            'num_rotatable_bonds': p.num_rotatable_bonds,
            'num_hbd': p.num_hbd,
            'num_hba': p.num_hba,
            'num_rings': p.num_rings,
            'num_heavy_atoms': p.num_heavy_atoms,
            'fraction_csp3': p.fraction_csp3,
            'aromatic_proportion': p.aromatic_proportion,
        }
        for p in props_list
    ])

    return props_df, valid_smiles


def print_basic_statistics(props_df: pd.DataFrame):
    """打印基本统计信息"""
    print("\n" + "=" * 80)
    print("1. 基本统计信息")
    print("=" * 80)

    print(f"\n分子数量: {len(props_df):,}")

    print("\n分子量 (MW):")
    print(f"  均值: {props_df['MW'].mean():.2f} Da")
    print(f"  中位数: {props_df['MW'].median():.2f} Da")
    print(f"  范围: {props_df['MW'].min():.2f} - {props_df['MW'].max():.2f} Da")
    print(f"  标准差: {props_df['MW'].std():.2f} Da")

    print("\n脂溶性 (logP):")
    print(f"  均值: {props_df['logP'].mean():.2f}")
    print(f"  中位数: {props_df['logP'].median():.2f}")
    print(f"  范围: {props_df['logP'].min():.2f} - {props_df['logP'].max():.2f}")

    print("\n极性表面积 (TPSA):")
    print(f"  均值: {props_df['TPSA'].mean():.2f} A^2")
    print(f"  中位数: {props_df['TPSA'].median():.2f} A^2")
    print(f"  范围: {props_df['TPSA'].min():.2f} - {props_df['TPSA'].max():.2f} A^2")

    print("\n重原子数:")
    print(f"  均值: {props_df['num_heavy_atoms'].mean():.1f}")
    print(f"  范围: {props_df['num_heavy_atoms'].min():.0f} - {props_df['num_heavy_atoms'].max():.0f}")

    print("\n旋转键数:")
    print(f"  均值: {props_df['num_rotatable_bonds'].mean():.1f}")
    print(f"  范围: {props_df['num_rotatable_bonds'].min():.0f} - {props_df['num_rotatable_bonds'].max():.0f}")

    print("\n氢键供体 (HBD):")
    print(f"  均值: {props_df['num_hbd'].mean():.2f}")
    print(f"  范围: {props_df['num_hbd'].min():.0f} - {props_df['num_hbd'].max():.0f}")

    print("\n氢键受体 (HBA):")
    print(f"  均值: {props_df['num_hba'].mean():.2f}")
    print(f"  范围: {props_df['num_hba'].min():.0f} - {props_df['num_hba'].max():.0f}")

    print("\n环数:")
    print(f"  均值: {props_df['num_rings'].mean():.2f}")
    print(f"  范围: {props_df['num_rings'].min():.0f} - {props_df['num_rings'].max():.0f}")


def print_drug_likeness(props_df: pd.DataFrame):
    """药物相似性分析"""
    print("\n" + "=" * 80)
    print("2. 药物相似性分析 (Lipinski's Rule of 5)")
    print("=" * 80)

    # Lipinski's Rule of 5
    mw_ok = (props_df['MW'] >= 150) & (props_df['MW'] <= 500)
    logp_ok = (props_df['logP'] >= -2) & (props_df['logP'] <= 5)
    hbd_ok = props_df['num_hbd'] <= 5
    hba_ok = props_df['num_hba'] <= 10

    lipinski_compliant = mw_ok & logp_ok & hbd_ok & hba_ok

    print(f"\n完全符合Lipinski规则: {lipinski_compliant.sum():,} / {len(props_df):,} ({lipinski_compliant.mean()*100:.1f}%)")

    print("\n各项符合率:")
    print(f"  分子量 150-500 Da:     {mw_ok.sum():,} ({mw_ok.mean()*100:.1f}%)")
    print(f"  logP -2 到 5:          {logp_ok.sum():,} ({logp_ok.mean()*100:.1f}%)")
    print(f"  HBD ≤ 5:               {hbd_ok.sum():,} ({hbd_ok.mean()*100:.1f}%)")
    print(f"  HBA ≤ 10:              {hba_ok.sum():,} ({hba_ok.mean()*100:.1f}%)")

    # Veber规则
    rotatable_bonds_ok = props_df['num_rotatable_bonds'] <= 10
    tpsa_ok = props_df['TPSA'] <= 140
    veber_compliant = rotatable_bonds_ok & tpsa_ok

    print(f"\nVeber规则符合率: {veber_compliant.sum():,} / {len(props_df):,} ({veber_compliant.mean()*100:.1f}%)")
    print(f"  旋转键 ≤ 10:           {rotatable_bonds_ok.sum():,} ({rotatable_bonds_ok.mean()*100:.1f}%)")
    print(f"  TPSA <= 140 A^2:          {tpsa_ok.sum():,} ({tpsa_ok.mean()*100:.1f}%)")


def analyze_diversity(props_df: pd.DataFrame, smiles_list: List[str]):
    """分析分子多样性"""
    print("\n" + "=" * 80)
    print("3. 分子多样性分析")
    print("=" * 80)

    # MW分布
    print("\n分子量分布:")
    bins = [0, 200, 300, 400, 500, 1000]
    labels = ['<200', '200-300', '300-400', '400-500', '>500']
    mw_dist = pd.cut(props_df['MW'], bins=bins, labels=labels).value_counts().sort_index()

    for label, count in mw_dist.items():
        print(f"  {label} Da:  {count:,} ({count/len(props_df)*100:.1f}%)")

    # logP分布
    print("\n脂溶性分布:")
    bins = [-10, 0, 2, 3, 5, 10]
    labels = ['<0', '0-2', '2-3', '3-5', '>5']
    logp_dist = pd.cut(props_df['logP'], bins=bins, labels=labels).value_counts().sort_index()

    for label, count in logp_dist.items():
        print(f"  {label}:     {count:,} ({count/len(props_df)*100:.1f}%)")

    # TPSA分布
    print("\n极性表面积分布:")
    bins = [0, 40, 80, 120, 140, 300]
    labels = ['<40', '40-80', '80-120', '120-140', '>140']
    tpsa_dist = pd.cut(props_df['TPSA'], bins=bins, labels=labels).value_counts().sort_index()

    for label, count in tpsa_dist.items():
        print(f"  {label} A^2:   {count:,} ({count/len(props_df)*100:.1f}%)")

    # 环数分布
    print("\n环数分布:")
    ring_dist = props_df['num_rings'].value_counts().sort_index()
    for ring_num, count in ring_dist.head(10).items():
        print(f"  {int(ring_num)} 个环:  {count:,} ({count/len(props_df)*100:.1f}%)")


def print_pretraining_method():
    """打印预训练方法说明"""
    print("\n" + "=" * 80)
    print("4. 预训练方法说明")
    print("=" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ZINC20自监督预训练方法                                    │
└─────────────────────────────────────────────────────────────────────────────┘

1. 模型架构
   -----------
   骨干网络: GAT (Graph Attention Network)
   - 输入: 原子特征 (29维) + 边特征
   - 隐藏层: 128维
   - 注意力头: 4个
   - 层数: 3层
   - 输出: 图级别表示 (128维)

   总参数量: ~399K


2. 预训练任务 (Multi-task Learning)
   ----------------------------------

   Task 1: Context Prediction (上下文预测)
   ────────────────────────────────────────
   目标: 预测每个原子邻域中的原子类型
   方法: 多标签分类，预测邻域中是否存在 {C, N, O, F, P, S, Cl, Br, I}
   输入: 节点嵌入 [num_nodes, 128]
   输出: 9维logits (每个原子类型一个)
   损失: BCELoss
   权重: λ_context = 1.0

   原理: 学习局部化学环境和官能团模式


   Task 2: Property Prediction (属性预测)
   ────────────────────────────────────────
   目标: 预测分子物理化学性质
   预测属性 (9个):
     1. logP          - 脂溶性
     2. TPSA          - 拓扑极性表面积
     3. MW            - 分子量
     4. #RotBonds     - 旋转键数量
     5. #HBD          - 氢键供体数
     6. #HBA          - 氢键受体数
     7. #Rings        - 环数
     8. Fraction Csp3 - sp3碳比例
     9. Aromatic %    - 芳香原子比例

   输入: 图嵌入 [batch_size, 128]
   输出: 9维属性值
   损失: MSELoss (回归)
   权重: λ_property = 1.0

   原理: 学习全局分子性质与结构的关系


   Task 3: Masked Reconstruction (掩码重构) [可选]
   ───────────────────────────────────────────────────
   目标: 重构被随机掩码的节点特征
   方法: 类似BERT，随机掩码15%的节点
   输入: 掩码节点的上下文嵌入
   输出: 原始原子特征 (29维)
   损失: MSELoss
   权重: λ_mask = 0.5 (默认关闭)

   原理: 学习鲁棒的原子表示


3. 训练策略
   -----------
   优化器: AdamW
   学习率: 2e-3 (cosine annealing)
   Batch size: 256
   Epochs: 100 (或更多)
   Weight decay: 1e-4
   梯度裁剪: max_norm=5.0

   数据分割: 90% train / 5% val / 5% test


4. 预训练数据
   -----------
   当前规模: 6,244 分子 (从BBB数据采样)
   推荐规模: 1,000,000+ 分子 (从ZINC20下载)

   数据来源:
   - ZINC20: 10亿+可商业化分子
   - ZINC15: 稳定的公开子集
   - 或自定义SMILES文件


5. 微调 (Downstream Task)
   ------------------------
   步骤:
   1. 加载预训练的backbone权重
   2. 添加任务特定的分类头
   3. 冻结或微调backbone
   4. 在下游任务上训练

   BBB分类器:
     Input: 预训练backbone [128-dim]
     Head: MLP [128 -> 128 -> 1]
     Loss: BCEWithLogitsLoss
     LR: 1e-3 (比预训练更小)


6. 预期效果
   -----------
   小规模 (6K分子): AUC ~0.94
   中规模 (100K):   AUC ~0.95+
   大规模 (1M+):    AUC ~0.96+ (可能超越XGB)

   优势场景:
   - 下游数据量小 (<1K样本)
   - 需要迁移学习到新任务
   - 分子结构复杂


7. 关键超参数
   -----------
   lambda_context:  1.0   # Context loss权重
   lambda_property: 1.0   # Property loss权重
   lambda_mask:     0.5   # Mask loss权重
   mask_ratio:      0.15  # 掩码比例

   hidden:          128   # 隐藏层维度
   heads:           4     # 注意力头数
   num_layers:      3     # GNN层数
   dropout:         0.2   # Dropout率
""")


def visualize_data(props_df: pd.DataFrame, output_dir: str):
    """生成数据可视化"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n生成可视化图表...")

    # 设置风格
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 分子量分布
    axes[0, 0].hist(props_df['MW'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(props_df['MW'].mean(), color='red', linestyle='--', label=f"Mean: {props_df['MW'].mean():.0f}")
    axes[0, 0].set_xlabel('分子量 (Da)')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('分子量分布')
    axes[0, 0].legend()

    # 2. logP分布
    axes[0, 1].hist(props_df['logP'], bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(props_df['logP'].mean(), color='red', linestyle='--', label=f"Mean: {props_df['logP'].mean():.2f}")
    axes[0, 1].set_xlabel('logP')
    axes[0, 1].set_ylabel('频数')
    axes[0, 1].set_title('脂溶性分布')
    axes[0, 1].legend()

    # 3. TPSA分布
    axes[0, 2].hist(props_df['TPSA'], bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(props_df['TPSA'].mean(), color='red', linestyle='--', label=f"Mean: {props_df['TPSA'].mean():.0f}")
    axes[0, 2].set_xlabel('TPSA (Ų)')
    axes[0, 2].set_ylabel('频数')
    axes[0, 2].set_title('极性表面积分布')
    axes[0, 2].legend()

    # 4. 旋转键分布
    axes[1, 0].hist(props_df['num_rotatable_bonds'], bins=range(0, int(props_df['num_rotatable_bonds'].max())+1),
                     color='plum', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('旋转键数')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('旋转键分布')

    # 5. HBD/HBA分布
    axes[1, 1].scatter(props_df['num_hbd'], props_df['num_hba'], alpha=0.5, s=10)
    axes[1, 1].set_xlabel('氢键供体数 (HBD)')
    axes[1, 1].set_ylabel('氢键受体数 (HBA)')
    axes[1, 1].set_title('HBD vs HBA')

    # 6. MW vs logP
    scatter = axes[1, 2].scatter(props_df['MW'], props_df['logP'],
                                  c=props_df['TPSA'], cmap='viridis', alpha=0.5, s=10)
    axes[1, 2].set_xlabel('分子量 (Da)')
    axes[1, 2].set_ylabel('logP')
    axes[1, 2].set_title('MW vs logP (color=TPSA)')
    plt.colorbar(scatter, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig(output_path / "data_distribution.png", dpi=150, bbox_inches='tight')
    print(f"图表保存到: {output_path / 'data_distribution.png'}")

    # 第二张图: 药物相似性
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Lipinski符合情况
    lipinski_scores = []
    for _, row in props_df.iterrows():
        score = sum([
            150 <= row['MW'] <= 500,
            -2 <= row['logP'] <= 5,
            row['num_hbd'] <= 5,
            row['num_hba'] <= 10
        ])
        lipinski_scores.append(score)

    axes[0].bar(range(5), [lipinski_scores.count(i) for i in range(5)],
                color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
    axes[0].set_xlabel('Lipinski规则符合数')
    axes[0].set_ylabel('分子数')
    axes[0].set_title('Lipinski Rule of 5 符合情况')
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(['0', '1', '2', '3', '4'])

    # 环数分布
    ring_counts = props_df['num_rings'].value_counts().sort_index()
    axes[1].bar(ring_counts.index, ring_counts.values, color='steelblue', alpha=0.7)
    axes[1].set_xlabel('环数')
    axes[1].set_ylabel('频数')
    axes[1].set_title('环数分布')

    # 芳香比例
    axes[2].hist(props_df['aromatic_proportion'], bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('芳香原子比例')
    axes[2].set_ylabel('频数')
    axes[2].set_title('芳香性分布')

    plt.tight_layout()
    plt.savefig(output_path / "drug_likeness.png", dpi=150, bbox_inches='tight')
    print(f"图表保存到: {output_path / 'drug_likeness.png'}")

    plt.close('all')


def show_example_molecules(smiles_list: List[str], num_examples: int = 12):
    """展示示例分子"""
    print("\n" + "=" * 80)
    print("5. 示例分子展示")
    print("=" * 80)

    # 随机采样
    np.random.seed(42)
    indices = np.random.choice(len(smiles_list), min(num_examples, len(smiles_list)), replace=False)

    print(f"\n随机选择 {len(indices)} 个分子示例:\n")

    for i, idx in enumerate(indices, 1):
        smi = smiles_list[idx]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        props = compute_zinc_properties(smi)
        if props is None:
            continue

        print(f"{i}. {smi}")
        print(f"   MW={props.mw:.1f}, logP={props.logp:.2f}, TPSA={props.tpsa:.1f}, "
              f"HBD={props.num_hbd}, HBA={props.num_hba}, Rings={props.num_rings}")
        print()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles-file", type=str, default=None,
                    help="SMILES文件路径 (默认使用最新下载的)")
    ap.add_argument("--output-dir", type=str, default="outputs/pretraining_analysis",
                    help="输出目录")
    ap.add_argument("--visualize", action="store_true",
                    help="生成可视化图表")
    ap.add_argument("--examples", type=int, default=12,
                    help="显示示例分子数量")

    args = ap.parse_args()

    # 自动查找数据文件
    if args.smiles_file is None:
        data_dir = PROJECT_ROOT / "data" / "zinc20"
        smiles_files = list(data_dir.glob("zinc20_*.csv"))
        if not smiles_files:
            print("ERROR: 未找到SMILES文件!")
            print("请先运行: python pretrain_zinc20.py --step download")
            return

        # 使用最新的文件
        smiles_file = sorted(smiles_files, key=lambda p: p.stat().st_mtime)[-1]
    else:
        smiles_file = Path(args.smiles_file)

    # 分析数据
    result = analyze_smiles_file(smiles_file)
    if result is None:
        return

    props_df, smiles_list = result

    # 打印统计
    print_basic_statistics(props_df)
    print_drug_likeness(props_df)
    analyze_diversity(props_df, smiles_list)
    print_pretraining_method()
    show_example_molecules(smiles_list, args.examples)

    # 可视化
    if args.visualize:
        visualize_data(props_df, args.output_dir)

    # 保存统计信息
    stats_file = Path(args.output_dir) / "data_statistics.csv"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    props_df.describe().to_csv(stats_file)
    print(f"\n统计信息已保存到: {stats_file}")

    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
