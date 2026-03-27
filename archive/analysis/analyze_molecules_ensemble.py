"""
多模型综合分析小分子BBB渗透性
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

class BBBEnsemblePredictor:
    """多模型集成预测器"""

    def __init__(self):
        self.models = {}
        self.model_weights = {
            'RF': 0.25,      # AUC=0.958
            'XGB': 0.25,     # AUC=0.949
            'LGBM': 0.25,    # AUC=0.955
            'GAT': 0.25      # AUC=0.952
        }

    def load_baseline_models(self, dataset='A_B'):
        """加载传统ML模型"""
        model_dir = PROJECT_ROOT / f"artifacts/models/seed_0_{dataset}/baseline"

        self.models['RF'] = joblib.load(model_dir / 'RF_seed0.joblib')
        self.models['XGB'] = joblib.load(model_dir / 'XGB_seed0.joblib')
        self.models['LGBM'] = joblib.load(model_dir / 'LGBM_seed0.joblib')

        model_names = list(self.models.keys())
        print(f"已加载传统ML模型: {model_names}")

    def load_smarts_models(self, dataset='A_B'):
        """加载SMARTS模型"""
        model_dir = PROJECT_ROOT / f"artifacts/models/seed_0_{dataset}/baseline_smarts"

        self.models['RF_smarts'] = joblib.load(model_dir / 'RF_smarts_seed0.joblib')
        self.models['XGB_smarts'] = joblib.load(model_dir / 'XGB_smarts_seed0.joblib')
        self.models['LGBM_smarts'] = joblib.load(model_dir / 'LGBM_smarts_seed0.joblib')

        smarts_models = [k for k in self.models.keys() if 'smarts' in k]
        print(f"已加载SMARTS模型: {smarts_models}")

    def compute_morgan_fingerprints(self, smiles_list):
        """计算Morgan指纹"""
        fingerprints = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fingerprints.append(np.zeros(2048, dtype=np.int8))
                continue

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            on_bits = list(fp.GetOnBits())
            fp_np = np.zeros(2048, dtype=np.int8)
            fp_np[on_bits] = 1
            fingerprints.append(fp_np)

        return sparse.csr_matrix(np.vstack(fingerprints), dtype=np.int8).astype(np.float32)

    def compute_smarts_features(self, smiles_list):
        """计算SMARTS特征"""
        smarts_file = PROJECT_ROOT / 'assets' / 'smarts' / 'bbb_smarts_v1.json'
        with open(smarts_file, 'r') as f:
            smarts_patterns = [item['smarts'] for item in json.load(f)]

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

        return sparse.csr_matrix(np.vstack(features), dtype=np.int8).astype(np.float32)

    def predict_baseline(self, smiles_list):
        """使用baseline模型预测"""
        X = self.compute_morgan_fingerprints(smiles_list)

        results = {}
        for name, model in self.models.items():
            if 'smarts' not in name:
                results[name] = model.predict_proba(X)[:, 1]

        return results

    def predict_smarts(self, smiles_list):
        """使用SMARTS模型预测"""
        X_morgan = self.compute_morgan_fingerprints(smiles_list)
        X_smarts = self.compute_smarts_features(smiles_list)
        X = sparse.hstack([X_morgan, X_smarts], format='csr').astype(np.float32)

        results = {}
        for name, model in self.models.items():
            if 'smarts' in name:
                base_name = name.replace('_smarts', '')
                results[base_name] = model.predict_proba(X)[:, 1]

        return results

    def predict_ensemble(self, smiles_list, strategy='consensus'):
        """集成预测

        Args:
            smiles_list: SMILES列表
            strategy: 'consensus'(平均), 'conservative'(最小), 'aggressive'(最大), 'weighted'(加权)

        Returns:
            DataFrame包含所有模型预测和集成结果
        """
        baseline_preds = self.predict_baseline(smiles_list)
        smarts_preds = self.predict_smarts(smiles_list)

        # 合并预测结果
        all_preds = {}
        for model_name in baseline_preds.keys():
            all_preds[model_name] = baseline_preds[model_name]

        for model_name in smarts_preds.keys():
            all_preds[f"{model_name}+SMARTS"] = smarts_preds[model_name]

        # 创建结果DataFrame
        results = pd.DataFrame({
            'SMILES': smiles_list,
            **{f'RF': all_preds.get('RF', np.nan),
               'RF+SMARTS': all_preds.get('RF+SMARTS', np.nan),
               'XGB': all_preds.get('XGB', np.nan),
               'XGB+SMARTS': all_preds.get('XGB+SMARTS', np.nan),
               'LGBM': all_preds.get('LGBM', np.nan),
               'LGBM+SMARTS': all_preds.get('LGBM+SMARTS', np.nan)}
        })

        # 计算集成预测
        pred_cols = ['RF', 'RF+SMARTS', 'XGB', 'XGB+SMARTS', 'LGBM', 'LGBM+SMARTS']
        valid_preds = results[pred_cols].values

        # 计算统计量
        results['mean'] = np.nanmean(valid_preds, axis=1)
        results['std'] = np.nanstd(valid_preds, axis=1)
        results['min'] = np.nanmin(valid_preds, axis=1)
        results['max'] = np.nanmax(valid_preds, axis=1)
        results['median'] = np.nanmedian(valid_preds, axis=1)

        # 根据策略选择最终预测
        if strategy == 'consensus':
            results['ensemble_pred'] = results['mean']
        elif strategy == 'conservative':
            results['ensemble_pred'] = results['min']
        elif strategy == 'aggressive':
            results['ensemble_pred'] = results['max']
        elif strategy == 'weighted':
            # 简单加权：所有模型权重相等
            results['ensemble_pred'] = results['mean']
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 分类和置信度
        results['prediction'] = results['ensemble_pred'].apply(lambda x: 'BBB+' if x > 0.5 else 'BBB-')
        results['confidence'] = results['std'].apply(self._compute_confidence)

        return results

    def _compute_confidence(self, std):
        """根据标准差计算置信度"""
        if std < 0.05:
            return '极高'
        elif std < 0.10:
            return '高'
        elif std < 0.20:
            return '中等'
        else:
            return '低'

    def analyze_results(self, results_df):
        """分析预测结果"""
        print("\n" + "="*80)
        print("多模型综合分析结果")
        print("="*80)

        # 格式化输出
        display_cols = ['SMILES', 'RF', 'XGB', 'LGBM', 'RF+SMARTS', 'XGB+SMARTS',
                       'LGBM+SMARTS', 'mean', 'min', 'max', 'std', 'prediction', 'confidence']

        display_df = results_df[display_cols].copy()
        for col in ['RF', 'XGB', 'LGBM', 'RF+SMARTS', 'XGB+SMARTS', 'LGBM+SMARTS',
                    'mean', 'min', 'max', 'std']:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}' if not np.isnan(x) else 'N/A')

        print("\n详细预测结果:")
        print(display_df.to_string(index=False))

        # 统计分析
        print("\n" + "-"*80)
        print("统计分析:")
        print(f"  平均预测概率: {results_df['mean'].mean():.4f}")
        print(f"  预测为BBB+的分子数: {(results_df['prediction'] == 'BBB+').sum()}")
        print(f"  预测为BBB-的分子数: {(results_df['prediction'] == 'BBB-').sum()}")

        print("\n置信度分布:")
        for conf in ['极高', '高', '中等', '低']:
            count = (results_df['confidence'] == conf).sum()
            print(f"  {conf}: {count} 个分子")

        # 高置信度BBB+分子（最有可能通过血脑屏障）
        high_conf_bbb_plus = results_df[
            (results_df['prediction'] == 'BBB+') &
            (results_df['confidence'].isin(['高', '极高']))
        ].sort_values('mean', ascending=False)

        print("\n高置信度BBB+分子（推荐优先验证）:")
        if len(high_conf_bbb_plus) > 0:
            for idx, row in high_conf_bbb_plus.iterrows():
                print(f"  {row['SMILES'][:50]:50} | mean={row['mean']:.4f} | {row['confidence']}")
        else:
            print("  无")

        # 低置信度分子（模型分歧大，需要进一步分析）
        low_conf = results_df[
            results_df['confidence'].isin(['低', '中等'])
        ].sort_values('std', ascending=False)

        print("\n低置信度分子（模型分歧大，建议谨慎）:")
        if len(low_conf) > 0:
            for idx, row in low_conf.iterrows():
                print(f"  {row['SMILES'][:50]:50} | mean={row['mean']:.4f} | std={row['std']:.4f}")
        else:
            print("  无")

        return results_df


def main():
    """示例使用"""
    # 初始化预测器
    predictor = BBBEnsemblePredictor()

    # 加载模型
    predictor.load_baseline_models(dataset='A_B')
    predictor.load_smarts_models(dataset='A_B')

    # 示例分子
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # 阿司匹林
        'CC(=O)NC1=CC=C(C=C1)O',     # 对乙酰氨基酚
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # 咖啡因
        'c1ccccc1',                  # 苯
        'CCO',                       # 乙醇
    ]

    print("预测分子:")
    for i, smi in enumerate(test_smiles, 1):
        print(f"  {i}. {smi}")

    # 预测
    results = predictor.predict_ensemble(test_smiles, strategy='consensus')

    # 分析
    predictor.analyze_results(results)

    # 保存结果
    output_file = 'ensemble_predictions.csv'
    results.to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
