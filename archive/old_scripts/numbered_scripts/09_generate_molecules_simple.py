"""
简化版分子生成脚本 - 使用BBB预测模型筛选BBB+分子

由于VAE需要完整的图解码器，这里使用已知BBB+分子库进行筛选演示
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.multi_model_predictor import MultiModelPredictor, EnsembleStrategy
from rdkit import Chem
from rdkit.Chem import QED

def compute_sa(smiles):
    """简化的SA分数计算"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        from rdkit.Chem import Descriptors
        num_rings = Descriptors.RingCount(mol)
        num_hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [1, 6])
        num_rot = Descriptors.NumRotatableBonds(mol)
        sa = 1.0 + (num_rings * 0.5) + (num_hetero * 0.3) + (num_rot * 0.1)
        return min(sa, 10.0)
    except:
        return 10.0

def main():
    print("=" * 60)
    print("BBB分子生成/筛选演示")
    print("=" * 60)

    # 加载BBB预测模型
    print("\n加载BBB预测模型...")
    predictor = MultiModelPredictor(
        seed=0,
        strategy=EnsembleStrategy.SOFT_VOTING,
    )
    print("模型加载完成!")

    # 示例BBB+分子库 (来自B3DB和其他来源)
    print("\n准备分子库...")
    bbb_molecules = [
        # 常见BBB+药物
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CC(C)NCC(COC1=CC=CC2=C1CCCC2)O",  # Propranolol
        "CC(C)NCC(O)CO",  # Alprenolol
        "CNCCO",  # 2-(Methylamino)ethanol
        "CCN(CC)CC",  # Triethylamine
        "Cc1ccccc1N",  # Toluidine
        "CC(C)NCCc1ccccc1",  # Amphetamine
        "CNCCc1ccccc1",  # Phenylethylamine
        # 苯衍生物
        "Cc1ccccc1",  # Toluene
        "Cc1ccc(cc1)C",  # p-Xylene
        "Cc1ccc(C)cc1",  # m-Xylene
        # 醇类
        "CCO",  # Ethanol
        "CC(C)O",  # Isopropanol
        "CCCO",  # Propanol
        "CC(C)(C)O",  # tert-Butanol
        # 醚类
        "CCOCC",  # Diethyl ether
        "COCCOC",  # Dimethoxyethane
        # 酯类
        "CC(=O)OC",  # Methyl acetate
        "CCOC(=O)C",  # Ethyl acetate
        # 酮类
        "CC(=O)C",  # Acetone
        "CCC(=O)C",  # Butanone
        # 胺类
        "CN",  # Methylamine
        "CCN",  # Ethylamine
        "CCCN",  # Propylamine
        "CC(C)N",  # Isopropylamine
        "NCCO",  # Ethanolamine
        "NCC",  # Ethylenediamine
        # 其他常见BBB+分子
        "c1ccccc1",  # Benzene
        "C1CCCCC1",  # Cyclohexane
        "C1CCCC1",  # Cyclopentane
        "C=CC=C",  # Butadiene
        "C#C",  # Acetylene
    ]

    print(f"分子库大小: {len(bbb_molecules)}")

    # 预测BBB概率
    print("\n预测BBB渗透性...")
    results = predictor.predict(bbb_molecules)

    # 计算性质
    print("计算分子性质...")
    data = []
    for i, smi in enumerate(bbb_molecules):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        qed = QED.qed(mol)
        sa = compute_sa(smi)
        bbb_prob = results.ensemble_probability[i]

        data.append({
            'SMILES': smi,
            'BBB_prob': bbb_prob,
            'QED': qed,
            'SA': sa,
            'passes': bbb_prob >= 0.7 and qed >= 0.5 and sa <= 4.0
        })

    df = pd.DataFrame(data)

    # 筛选通过过滤的分子
    filtered = df[df['passes']]

    print("\n" + "=" * 60)
    print("结果")
    print("=" * 60)
    print(f"总分子数: {len(df)}")
    print(f"通过过滤: {len(filtered)} ({len(filtered)/len(df)*100:.1f}%)")
    print(f"平均BBB概率: {df['BBB_prob'].mean():.3f}")
    print(f"平均QED: {df['QED'].mean():.3f}")
    print(f"平均SA: {df['SA'].mean():.1f}")

    print("\n通过过滤的分子:")
    print(filtered[['SMILES', 'BBB_prob', 'QED', 'SA']].to_string(index=False))

    # 保存结果
    output_dir = PROJECT_ROOT / "outputs" / "generated_molecules"
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered.to_csv(output_dir / "bbb_positive_molecules.csv", index=False)
    df.to_csv(output_dir / "all_molecules_predictions.csv", index=False)

    print(f"\n结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()
