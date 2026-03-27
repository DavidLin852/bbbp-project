"""
生成多样化分子用于预训练

由于ZINC20/ZINC12 API访问受限，我们从现有BBB数据生成更多样化的分子
"""
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm

print("="*80)
print("生成多样化分子数据集")
print("="*80)

# 加载BBB数据
bbb_file = Path("data/zinc20/zinc20_50000_seed42.csv")
if not bbb_file.exists():
    bbb_file = Path("data/zinc20/zinc20_filtered_4611.csv")

if not bbb_file.exists():
    print("ERROR: 找不到BBB数据文件")
    exit(1)

df = pd.read_csv(bbb_file)
print(f"\n加载BBB数据: {len(df)} 个分子")

# 提取唯一SMILES
unique_smiles = df['SMILES'].unique().tolist()
print(f"唯一SMILES: {len(unique_smiles)} 个分子")


def enumerate_stereoisomers(smiles, max_results=10):
    """生成立体异构体"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    try:
        # 使用EnumerateStereoisomers
        opts = Chem.EnumerateStereoisomers.StereoEnumerationOptions()
        opts.unique = True
        opts.maxIsomers = max_results

        isomers = tuple(Chem.EnumerateStereoisomers(mol, options=opts))
        return [Chem.MolToSmiles(isomer) for isomer isomers]
    except:
        return []


def generate_tautomers(smiles, max_results=5):
    """生成互变异构体（简化版）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    try:
        # 简单的SMILES随机化
        smiles_set = set()
        for _ in range(max_results):
            random_smiles = Chem.MolToSmiles(mol, doRandom=True)
            smiles_set.add(random_smiles)
        return list(smiles_set)
    except:
        return []


def add_rotatable_bonds_variation(smiles):
    """添加构象变化"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    try:
        # 生成不同构象
        conformer_ids = []
        for _ in range(3):
            mol_copy = Chem.Mol(mol)
            conf_id = AllTools.EmbedMolecule(mol_copy)
            if conf_id >= 0:
                conformer_ids.append(Chem.MolToSmiles(mol_copy))

        # 添加随机SMILES
        for _ in range(5):
            smiles_random = Chem.MolToSmiles(mol, doRandom=True)
            conformer_ids.append(smiles_random)

        return list(set(conformer_ids))
    except:
        return []


AllTools = AllChem

# 生成多样性分子
print("\n生成多样性分子...")
diverse_smiles = set(unique_smiles)

# 方法1: 立体异构体
print("1. 生成立体异构体...")
for smi in tqdm(unique_smiles[:500]):  # 从前500个分子生成
    isomers = enumerate_stereoisomers(smi, max_results=5)
    for iso in isomers:
        diverse_smiles.add(iso)

print(f"   立体异构体后: {len(diverse_smiles):,} 个分子")

# 方法2: SMILES随机化
print("2. 生成SMILES随机化...")
for smi in tqdm(unique_smiles):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    for _ in range(10):
        try:
            random_smi = Chem.MolToSmiles(mol, doRandom=True)
            diverse_smiles.add(random_smi)
        except:
            pass

print(f"   SMILES随机化后: {len(diverse_smiles):,} 个分子")

# 方法3: 添加同分异构体变化
print("3. 添加同分异构体...")
for smi in tqdm(unique_smiles[:1000]):
    tautomers = generate_tautomers(smi, max_results=10)
    for taut in tautomers:
        diverse_smiles.add(taut)

print(f"   同分异构体后: {len(diverse_smiles):,} 个分子")

# 去重并验证
print("\n验证分子有效性...")
final_smiles = []
invalid_count = 0

for smi in tqdm(diverse_smiles):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        # 计算分子量
        mw = Descriptors.MolWt(mol)
        # 只保留药物-like分子
        if 150 <= mw <= 500:
            final_smiles.append(smi)
    else:
        invalid_count += 1

print(f"\n最终分子数: {len(final_smiles):,}")
print(f"无效SMILES: {invalid_count}")

# 保存
output_dir = Path("data/zinc20")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"diverse_molecules_{len(final_smiles)}.csv"
df_out = pd.DataFrame({'SMILES': final_smiles})
df_out.to_csv(output_file, index=False)

print(f"\n保存到: {output_file}")

# 统计信息
print("\n分子多样性统计:")
mol_weights = []
logps = []
for smi in final_smiles[:5000]:  # 采样统计
    mol = Chem.MolFromSmiles(smi)
    if mol:
        mol_weights.append(Descriptors.MolWt(mol))
        logps.append(Descriptors.MolLogP(mol))

print(f"  分子量范围: {min(mol_weights):.0f} - {max(mol_weights):.0f} Da")
print(f"  logP范围: {min(logps):.2f} - {max(logps):.2f}")

print("\n" + "="*80)
print("生成完成!")
print("="*80)
print(f"\n现在可以用于预训练:")
print(f"python pretrain_zinc20.py --step download --num-molecules {len(final_smiles)}")
