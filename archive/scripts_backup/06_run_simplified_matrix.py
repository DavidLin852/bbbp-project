"""
Simplified Model Experiment - Each Feature Independent
简化模型实验 - 每个特征独立测试
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline.train_baselines import MODEL_CONFIGS, train_single_model

# =============================================================================
# 简化的模型配置
# =============================================================================

# 移除MLP，只保留SVM_RBF和KNN5
SIMPLIFIED_MODELS = {
    # 树模型
    "RF": MODEL_CONFIGS["RF"],
    "XGB": MODEL_CONFIGS["XGB"],
    "LGBM": MODEL_CONFIGS["LGBM"],
    "ETC": MODEL_CONFIGS["ETC"],
    "GB": MODEL_CONFIGS["GB"],
    "ADA": MODEL_CONFIGS["ADA"],

    # SVM - 只保留RBF
    "SVM_RBF": MODEL_CONFIGS["SVM_RBF"],

    # KNN - 只保留KNN5
    "KNN5": MODEL_CONFIGS["KNN5"],

    # 概率模型
    "LR": MODEL_CONFIGS["LR"],
    "NB_Bernoulli": MODEL_CONFIGS["NB_Bernoulli"],  # 适合二值特征
}

# 特征配置 - 每个特征独立测试
# 移除combined
FEATURE_CONFIG = {
    "morgan": {
        "file": "morgan.npz",
        "dims": 2048,
        "sparse": True,
        "display": "Morgan (2048D)"
    },
    "maccs": {
        "file": "maccs.npz",
        "dims": 167,
        "sparse": True,
        "display": "MACCS (167D)"
    },
    "atompairs": {
        "file": "atompairs.npz",
        "dims": 1024,
        "sparse": True,
        "display": "AtomPairs (1024D)"
    },
    "fp2": {
        "file": "fp2.npz",
        "dims": 2048,
        "sparse": True,
        "display": "FP2 (2048D)"
    },
    "rdkit_desc": {
        "file": "descriptors.npz",
        "dims": 98,
        "sparse": False,
        "display": "RDKitDescriptors (98D)"
    }
}

# =============================================================================
# 数据加载
# =============================================================================

def load_data(seed=0):
    """加载数据"""
    split_dir = PROJECT_ROOT / "data" / "splits" / f"seed_{seed}"
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    return train_df, val_df, test_df


def load_features(feature_name, seed=0):
    """加载特征"""
    feat_dir = PROJECT_ROOT / "artifacts" / "features" / f"seed_{seed}_enhanced"
    feat_config = FEATURE_CONFIG[feature_name]
    feat_file = feat_dir / feat_config["file"]

    from scipy import sparse
    data = np.load(feat_file)

    # 检查是否是稀疏矩阵格式
    if 'format' in data.files and 'data' in data.files:
        X = sparse.csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
    else:
        X = data[list(data.keys())[0]] if 'X' not in data.files else data['X']

    # 加载meta映射
    meta = pd.read_csv(feat_dir / "meta.csv")
    row_id_to_idx = {row_id: idx for idx, row_id in enumerate(meta["row_id"].values)}

    return X, row_id_to_idx


def get_feature_indices(train_df, val_df, test_df, row_id_to_idx):
    """获取特征索引"""
    train_idx = train_df["row_id"].map(row_id_to_idx).values
    val_idx = val_df["row_id"].map(row_id_to_idx).values
    test_idx = test_df["row_id"].map(row_id_to_idx).values

    return train_idx, val_idx, test_idx


# =============================================================================
# 实验运行
# =============================================================================

def run_simplified_experiment():
    """运行简化的实验矩阵"""

    results_dir = PROJECT_ROOT / "artifacts" / "ablation"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "SIMPLIFIED_RESULTS.csv"

    train_df, val_df, test_df = load_data(seed=0)

    all_results = []

    print("=" * 80)
    print("简化模型-特征实验矩阵")
    print("=" * 80)
    print(f"模型数量: {len(SIMPLIFIED_MODELS)}")
    print(f"特征数量: {len(FEATURE_CONFIG)}")
    print(f"总实验数: {len(SIMPLIFIED_MODELS) * len(FEATURE_CONFIG)}")
    print("=" * 80)

    for feature_name in FEATURE_CONFIG.keys():
        print(f"\n{'='*80}")
        print(f"特征: {feature_name.upper()} ({FEATURE_CONFIG[feature_name]['display']})")
        print(f"{'='*80}")

        try:
            X_all, row_id_to_idx = load_features(feature_name, seed=0)
            train_idx, val_idx, test_idx = get_feature_indices(train_df, val_df, test_df, row_id_to_idx)

            X_train = X_all[train_idx]
            X_val = X_all[val_idx]
            X_test = X_all[test_idx]

            y_train = train_df["y_cls"].values
            y_val = val_df["y_cls"].values
            y_test = test_df["y_cls"].values

            print(f"X_train shape: {X_train.shape}")
            print(f"类别分布: {np.bincount(y_train)}")

            for model_name in SIMPLIFIED_MODELS.keys():
                print(f"\n  运行: {model_name}...", end=" ")

                try:
                    result = train_single_model(
                        model_name=model_name,
                        model_config=SIMPLIFIED_MODELS[model_name],
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        X_test=X_test,
                        y_test=y_test,
                        out_model_dir=results_dir / f"{model_name}_{feature_name}",
                        seed=0
                    )

                    if result is not None:
                        result['feature'] = feature_name
                        result['feature_dims'] = FEATURE_CONFIG[feature_name]['dims']
                        all_results.append(result)
                        print(f"✅ AUC={result['auc']:.4f}")
                    else:
                        print(f"❌ 失败")

                except Exception as e:
                    print(f"❌ 错误: {e}")

        except Exception as e:
            print(f"\n❌ 特征 {feature_name} 加载失败: {e}")
            continue

    # 保存结果
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values('auc', ascending=False).reset_index(drop=True)
        df.to_csv(output_file, index=False)

        print(f"\n{'='*80}")
        print("实验完成")
        print(f"{'='*80}")
        print(f"总实验数: {len(df)}")
        print(f"结果已保存: {output_file}")

        # Top 10
        print("\nTop 10 模型:")
        for idx, row in df.head(10).iterrows():
            print(f"  {row['model']:10} + {row['feature']:15} → AUC={row['auc']:.4f}, F1={row['f1']:.4f}, MCC={row['mcc']:.4f}")

        return df
    else:
        print("\n❌ 没有成功的实验")
        return None


if __name__ == "__main__":
    df = run_simplified_experiment()
