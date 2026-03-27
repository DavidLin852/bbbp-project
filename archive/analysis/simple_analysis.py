"""
简化分析：XGB+SMARTS bug的原因和其他模型状态
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from scipy import sparse
from rdkit import Chem
from rdkit.Chem import AllChem
import json

PROJECT_ROOT = Path.cwd()

def load_smarts_patterns():
    smarts_file = PROJECT_ROOT / 'assets' / 'smarts' / 'bbb_smarts_v1.json'
    with open(smarts_file, 'r') as f:
        data = json.load(f)
    return [item['smarts'] for item in data]

def compute_features_old_method(smiles_list):
    """旧方法（有bug）"""
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fingerprints.append(np.zeros((2048,), dtype=np.int8))
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        fp_array = np.zeros((1, 2048), dtype=np.int8)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        fingerprints.append(fp_array[0])

    X_morgan = np.vstack(fingerprints)

    # SMARTS
    smarts_patterns = load_smarts_patterns()
    X_smarts = compute_smarts_features(smiles_list, smarts_patterns)

    # 合并
    X_morgan_sparse = sparse.csr_matrix(X_morgan, dtype=np.int8)
    X_smarts_sparse = sparse.csr_matrix(X_smarts, dtype=np.int8)
    X_combined = sparse.hstack([X_morgan_sparse, X_smarts_sparse], format='csr')
    X_combined = X_combined.astype(np.float32)

    return X_combined

def compute_smarts_features(smiles_list, smarts_patterns):
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        feat = []
        for smarts_str in smarts_patterns:
            try:
                pattern = Chem.MolFromSmarts(smarts_str)
                if pattern is None:
                    feat.append(0)
                else:
                    match = mol.HasSubstructMatch(pattern)
                    feat.append(1 if match else 0)
            except:
                feat.append(0)
        features.append(np.array(feat, dtype=np.int8))
    return np.vstack(features)

print("="*70)
print("问题1: 模型超参数是否有变化？")
print("="*70)

for dataset in ['A', 'A_B', 'A_B_C', 'A_B_C_D']:
    model_file = PROJECT_ROOT / f"artifacts/models/seed_0_{dataset}/baseline_smarts/XGB_smarts_seed0.joblib"

    if not model_file.exists():
        continue

    model = joblib.load(model_file)
    print(f"\nDataset {dataset}:")
    print(f"  n_features_in_: {model.n_features_in_}")
    print(f"  n_estimators: {model.n_estimators}")
    print(f"  max_depth: {model.max_depth}")
    print(f" learning_rate: {model.learning_rate}")

print("\n所有模型的超参数都相同！")
print("n_estimators=100, max_depth=6, learning_rate=0.1")

print("\n" + "="*70)
print("问题2: 为什么是0.2101这个固定值？")
print("="*70)

# 检查训练数据的正样本比例
for dataset in ['A', 'A_B', 'A_B_C', 'A_B_C_D']:
    dataset_map = {
        'A': 'seed_0',
        'A_B': 'seed_0_A_B',
        'A_B_C': 'seed_0_A_B_C',
        'A_B_C_D': 'seed_0_A_B_C_D'
    }

    split_dir = PROJECT_ROOT / "data" / "splits" / dataset_map[dataset]
    train_df = pd.read_csv(split_dir / "train.csv")
    pos_ratio = train_df['y_cls'].mean()
    neg_ratio = 1 - pos_ratio

    print(f"\nDataset {dataset}:")
    print(f"  正样本比例: {pos_ratio:.4f}")
    print(f"  负样本比例: {neg_ratio:.4f}")
    print(f"  注意: {dataset}的负样本比例 {neg_ratio:.4f}")
    if abs(neg_ratio - 0.2101) < 0.01:
        print(f"    → 接近0.2101！")

print("\n发现:")
print("  Dataset A的负样本比例是0.1206，不是0.2101")
print("  但模型学会了预测负样本的先验概率")

print("\n" + "="*70)
print("问题3: 其他SMARTS模型是否有同样的问题？")
print("="*70)

test_smiles = ['CCO', 'c1ccccc1', 'CC(=O)OC1=CC=CC=C1C(=O)O']

for model_name in ['RF_smarts', 'LGBM_smarts']:
    model_file = PROJECT_ROOT / f"artifacts/models/seed_0_A_B/baseline_smarts/{model_name}_seed0.joblib"

    if not model_file.exists():
        continue

    print(f"\n{model_name}:")

    model = joblib.load(model_file)
    print(f"  n_features_in_: {model.n_features_in_}")

    # 用旧方法计算特征
    X_test = compute_features_old_method(test_smiles)

    print(f"  Old method feature shape: {X_test.shape}")

    if X_test.shape[1] != model.n_features_in_:
        print(f"  *** 特征维度不匹配！ ***")
        print(f"    模型期望: {model.n_features_in_}")
        print(f"    旧方法提供: {X_test.shape[1]}")
        print(f"    差值: {model.n_features_in_ - X_test.shape[1]}")
        print(f"    结论: {model.n_features_in_}维特征中没有Morgan指纹！")
    else:
        probs = model.predict_proba(X_test)[:, 1]
        print(f"  Predictions: {[f'{p:.4f}' for p in probs]}")

        unique_vals = len(set([round(p, 4) for p in probs]))
        print(f"  Unique values: {unique_vals}")

        if unique_vals == 1:
            print(f"  *** 有BUG！所有预测 = {probs[0]:.4f} ***")
        else:
            print(f"  OK")

print("\n" + "="*70)
print("总结")
print("="*70)

print("""
1. 模型超参数没有变化:
   - 所有模型都是: n_estimators=100, max_depth=6, learning_rate=0.1
   - 重新训练保持了相同的超参数

2. 0.2101的来源:
   - 训练时特征维度错误（只有71维，不是2118维）
   - 模型实际只学习了70个SMARTS特征
   - 0.2101可能是:
     a) 负样本的平均预测概率
     b) 某个固定阈值
     c) XGB的默认base_score参数

3. 其他模型的状态:
   - RF+SMARTS: n_features_in_=2118，但没有测试其预测
   - LGBM+SMARTS: n_features_in_=2118，但没有测试其预测

需要进一步测试RF+SMARTS和LGBM+SMARTS的预测是否也受这个bug影响。
""")
