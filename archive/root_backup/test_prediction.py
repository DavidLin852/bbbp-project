"""
Test prediction functionality
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, Descriptors
from scipy import sparse


def smiles_to_features(smiles):
    """将SMILES转换为分子特征"""
    # SMILES auto-fix mapping
    smiles_fixes = {
        'CC(C)(C)C1CCCc2ccccc2Cl': 'CC(C)C1CCCC1c2ccccc2Cl',  # MPC molecule fix
        'C=C(C)C=O)OCC[N+](C)(C)CCCS(=O)(=O)[O-]': 'C=C(C)C(=O)OCC[N+](C)(C)CCCS(=O)(=O)[O-]',  # SBMA fix
        'C=C(C)C=O)OCC[N+](C)(C)[O-]': 'C=C(C)C(=O)OCC[N+](C)(C)[O-]',  # ONMA fix
    }
    smiles_to_use = smiles_fixes.get(smiles, smiles)

    try:
        mol = Chem.MolFromSmiles(smiles_to_use)
        if mol is None:
            return None

        # Morgan指纹
        fp_morgan = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        morgan = np.array(fp_morgan, dtype=np.float32).reshape(1, -1)

        # MACCS
        fp_maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs = np.array(fp_maccs, dtype=np.float32).reshape(1, -1)

        # Atom Pairs
        fp_ap = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024)
        atompairs = np.array(fp_ap, dtype=np.float32).reshape(1, -1)

        # FP2
        fp_fp2 = Chem.RDKFingerprint(mol, fpSize=2048)
        fp2 = np.array(fp_fp2, dtype=np.float32).reshape(1, -1)

        # 合并特征 (仅4种，与训练时一致)
        combined = sparse.hstack([
            sparse.csr_matrix(morgan.reshape(1, -1)),
            sparse.csr_matrix(maccs.reshape(1, -1)),
            sparse.csr_matrix(atompairs.reshape(1, -1)),
            sparse.csr_matrix(fp2.reshape(1, -1))
        ])

        return {
            'morgan': morgan,
            'maccs': maccs,
            'atompairs': atompairs,
            'fp2': fp2,
            'combined': combined
        }

    except Exception as e:
        print(f"特征提取失败: {str(e)}")
        return None


def load_model(model_name, feature_name):
    """加载训练好的模型"""
    model_dir = PROJECT_ROOT / "artifacts" / "ablation"

    # 支持单个特征的模型
    single_feature_models = ['RF', 'SVM_RBF', 'KNN5', 'NB_Bernoulli']
    if model_name in single_feature_models:
        model_path = model_dir / f"{model_name}_{feature_name}"
        model_file = model_path / f"{model_name}_seed0.joblib"
        if model_file.exists():
            return joblib.load(model_file)

    # 仅支持 combined 特征的模型
    combined_only_models = ['XGB', 'LGBM', 'GB', 'ETC', 'ADA', 'LR']
    if model_name in combined_only_models:
        if feature_name == 'combined':
            model_file = model_dir / f"{model_name}_seed0.joblib"
            if model_file.exists():
                return joblib.load(model_file)

    # Ensemble 模型
    ensemble_dir = PROJECT_ROOT / "artifacts" / "models" / "ensemble"
    ensemble_models = {
        'Stacking_rf': 'stacking_rf.joblib',
        'Stacking_xgb': 'stacking_xgb.joblib',
        'SoftVoting': 'soft_voting.joblib',
    }
    if model_name in ensemble_models:
        if feature_name == 'combined':
            model_file = ensemble_dir / ensemble_models[model_name]
            if model_file.exists():
                return joblib.load(model_file)

    return None


def predict_with_model(model, features, feature_name):
    """使用模型进行预测"""
    X = features[feature_name]
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    return y_pred[0], y_prob[0]


# 测试分子
test_smiles = [
    'C=C(C)C(=O)OCCOP(=O)([O-])OCC[N+](C)(C)C',          # MPC
    'C=C(C)C(O)OCC[N+](C)(C)CC(=O)[O-]',                 # CBMA
    'C=C(C)C(=O)OCC[N+](C)(C)CCCS(=O)(=O)[O-]',         # SBMA (fixed)
    'C=C(C)C(=O)OCC[N+](C)(C)[O-]',                     # ONMA (fixed)
]

# 测试不同的模型和特征组合
test_configs = [
    ('Stacking_xgb', 'combined'),
    ('RF', 'morgan'),
    ('XGB', 'combined'),
    ('LGBM', 'combined'),
]

print("=" * 80)
print("测试预测功能")
print("=" * 80)

for model_name, feature_name in test_configs:
    print(f"\n测试: {model_name} + {feature_name}")
    print("-" * 60)

    # 加载模型
    model = load_model(model_name, feature_name)
    if model is None:
        print(f"[X] Model loading failed: {model_name} + {feature_name}")
        continue

    print(f"[OK] Model loaded: {type(model).__name__}")

    # 预测
    for smiles in test_smiles:
        features = smiles_to_features(smiles)

        if features is None:
            print(f"  [X] Feature extraction failed: {smiles[:30]}...")
            continue

        try:
            pred, prob = predict_with_model(model, features, feature_name)
            result = "BBB+" if pred == 1 else "BBB-"
            print(f"  [OK] {smiles[:30]:30} -> {result:4} ({prob:.2%})")
        except Exception as e:
            print(f"  [X] Prediction failed: {str(e)}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
