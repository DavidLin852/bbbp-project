"""
Multi-Model BBB Predictor with Ensemble Learning

支持多模型集成预测的模块，提供多种集成策略：
- Hard Voting (硬投票): 多数投票
- Soft Voting (软投票): 概率平均
- Weighted Voting (加权投票): 基于模型性能加权
- Stacking (堆叠): 使用元学习器组合基学习器

使用示例:
    from src.multi_model_predictor import MultiModelPredictor, EnsembleStrategy

    predictor = MultiModelPredictor(
        seed=0,
        strategy=EnsembleStrategy.HARD_VOTING,
        threshold=0.5
    )

    # 单个预测
    results = predictor.predict(['CCO', 'CC(=O)OC1=CC=C(C=C)C=C1'])

    # 查看集成结果
    print(results.ensemble_prediction)  # 最终预测
    print(results.agreement)  # 模型一致性
"""
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Union
from enum import Enum

import pandas as pd
import numpy as np
import joblib
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy import sparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier


class EnsembleStrategy(str, Enum):
    """集成策略枚举"""
    HARD_VOTING = "hard_voting"  # 硬投票：多数投票
    SOFT_VOTING = "soft_voting"  # 软投票：概率平均
    WEIGHTED = "weighted"  # 加权平均：基于模型性能
    MAX_PROB = "max_prob"  # 最大概率：取最高概率模型
    MIN_PROB = "min_prob"  # 最小概率：最保守预测
    STACKING = "stacking"  # 堆叠：使用元学习器


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    path: Path
    model_type: Literal['rf', 'gnn']  # 'rf' for traditional ML, 'gnn' for graph models
    weight: float = 1.0
    auc: float = 0.0
    precision: float = 0.0

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)


@dataclass
class PredictionResults:
    """预测结果数据类"""
    smiles: List[str]
    individual_predictions: Dict[str, np.ndarray]  # {model_name: predictions}
    individual_probabilities: Dict[str, np.ndarray]  # {model_name: probabilities}
    ensemble_prediction: np.ndarray  # 集成后的预测
    ensemble_probability: np.ndarray  # 集成后的概率
    agreement: np.ndarray  # 模型一致性 (0-1之间，1表示完全一致)
    strategy: EnsembleStrategy

    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        df = pd.DataFrame({
            'SMILES': self.smiles,
            'ensemble_prob': self.ensemble_probability,
            'ensemble_pred': ['BBB+' if p == 1 else 'BBB-' for p in self.ensemble_prediction],
            'agreement': self.agreement,
            'consensus': ['高' if a >= 0.75 else '中' if a >= 0.5 else '低' for a in self.agreement]
        })

        # 添加各模型结果
        for model_name in self.individual_predictions.keys():
            df[f'{model_name}_prob'] = self.individual_probabilities[model_name]
            df[f'{model_name}_pred'] = ['BBB+' if p == 1 else 'BBB-'
                                        for p in self.individual_predictions[model_name]]

        return df

    def get_summary(self) -> Dict:
        """获取预测摘要"""
        n_total = len(self.smiles)
        n_pos = int(self.ensemble_prediction.sum())
        n_unanimous = int(np.sum(self.agreement == 1.0))
        n_majority = int(np.sum((self.agreement >= 0.5) & (self.agreement < 1.0)))
        n_split = int(np.sum(self.agreement < 0.5))

        return {
            'total_samples': n_total,
            'predicted_bbb_plus': n_pos,
            'predicted_bbb_minus': n_total - n_pos,
            'positive_rate': n_pos / n_total if n_total > 0 else 0,
            'unanimous_agreement': n_unanimous,
            'majority_agreement': n_majority,
            'split_decision': n_split,
            'strategy': self.strategy.value
        }


class MultiModelPredictor:
    """多模型集成预测器

    支持同时使用多个模型进行预测，并提供多种集成策略
    """

    # 默认模型配置
    DEFAULT_MODELS = {
        'Random Forest': ModelConfig(
            name='Random Forest',
            path='artifacts/models/seed_0_full/baseline/RF_seed0.joblib',
            model_type='rf',
            auc=0.958,
            precision=0.876
        ),
        'XGBoost': ModelConfig(
            name='XGBoost',
            path='artifacts/models/seed_0_full/baseline/XGB_seed0.joblib',
            model_type='rf',
            auc=0.949,
            precision=0.866
        ),
        'LightGBM': ModelConfig(
            name='LightGBM',
            path='artifacts/models/seed_0_full/baseline/LGBM_seed0.joblib',
            model_type='rf',
            auc=0.955,
            precision=0.896
        ),
        'GAT+SMARTS': ModelConfig(
            name='GAT+SMARTS',
            path='artifacts/models/gat_finetune_bbb/seed_0/pretrained_partial/best.pt',
            model_type='gnn',
            auc=0.952,
            precision=0.942
        )
    }

    def __init__(
        self,
        seed: int = 0,
        strategy: EnsembleStrategy = EnsembleStrategy.HARD_VOTING,
        threshold: float = 0.5,
        models: Optional[Dict[str, ModelConfig]] = None,
        project_root: Optional[Path] = None
    ):
        """初始化多模型预测器

        Args:
            seed: 随机种子
            strategy: 集成策略
            threshold: 分类阈值
            models: 自定义模型配置（默认使用DEFAULT_MODELS）
            project_root: 项目根目录
        """
        self.seed = seed
        self.strategy = strategy
        self.threshold = threshold
        self.project_root = project_root or PROJECT_ROOT

        # 使用自定义模型或默认模型
        self.models_config = models or self.DEFAULT_MODELS

        # 过滤出可用的模型
        self.available_models = self._check_available_models()

        # 懒加载模型（首次使用时加载）
        self._loaded_models = {}

        if len(self.available_models) == 0:
            raise ValueError("没有可用的模型！请检查模型文件是否存在。")

    def _check_available_models(self) -> Dict[str, ModelConfig]:
        """检查哪些模型文件可用"""
        available = {}
        for name, config in self.models_config.items():
            model_path = self.project_root / config.path
            if model_path.exists():
                available[name] = config
        return available

    def _load_model(self, model_name: str):
        """加载单个模型"""
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        config = self.available_models[model_name]
        model_path = self.project_root / config.path

        if config.model_type == 'gnn':
            # GNN模型：返回配置，实际预测时再加载
            self._loaded_models[model_name] = {'type': 'gnn', 'path': model_path}
        else:
            # 传统ML模型：直接加载
            model = joblib.load(model_path)
            self._loaded_models[model_name] = {'type': 'rf', 'model': model}

        return self._loaded_models[model_name]

    def _predict_with_rf_model(self, model, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """使用传统ML模型预测"""
        # 计算Morgan指纹
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                arr = np.zeros((2048,), dtype=np.int8)
                fps.append(arr)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros((2048,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)

        X = sparse.csr_matrix(np.vstack(fps))

        # 预测
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= self.threshold).astype(int)

        return prob, pred

    def _predict_with_gnn_model(self, model_path, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """使用GNN模型预测"""
        from src.featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig
        from torch_geometric.loader import DataLoader
        from src.finetune.train_gat_bbb_from_pretrain import GATBBB, FinetuneCfg

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建临时DataFrame
        df_temp = pd.DataFrame({
            'SMILES': smiles_list,
            'y_cls': [0] * len(smiles_list),
            'row_id': range(len(smiles_list))
        })

        # 构建图数据集
        gcfg = GraphBuildConfig(smiles_col="SMILES", label_col="y_cls", id_col="row_id")

        # 加载模型checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # 创建模型
        cfg = FinetuneCfg(
            seed=self.seed,
            hidden=128,
            gat_heads=4,
            num_layers=3,
            dropout=0.2,
            epochs=60,
            batch_size=64,
            lr=2e-3,
            grad_clip=5.0,
            init='pretrained',
            strategy='freeze'
        )

        # 获取输入维度
        graph_data_dir = self.project_root / "artifacts" / "features" / f"seed_{self.seed}_full" / "pyg_graphs_baseline"
        if graph_data_dir.exists():
            ref_ds = BBBGraphDataset(
                root=str(graph_data_dir / "train"),
                df=pd.DataFrame({'SMILES': ['C'], 'y_cls': [0], 'row_id': [0]}),
                cfg=gcfg
            )
            in_dim = ref_ds[0].x.size(-1)
        else:
            in_dim = 23

        model = GATBBB(in_dim, cfg).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        # 创建临时图数据集
        import time
        import uuid
        unique_id = f"predict_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        temp_cache = self.project_root / "artifacts" / "temp_predict" / unique_id

        temp_ds = BBBGraphDataset(root=str(temp_cache), df=df_temp, cfg=gcfg)

        # 检查有效样本
        if len(temp_ds) == 0:
            prob = np.full(len(smiles_list), 0.5)
            pred = np.zeros(len(smiles_list), dtype=int)
            return prob, pred

        temp_loader = DataLoader(temp_ds, batch_size=64, shuffle=False)

        # 预测
        probs = []
        row_ids = []
        with torch.no_grad():
            for batch in temp_loader:
                batch = batch.to(device)
                logit = model(batch)
                prob_batch = torch.sigmoid(logit).cpu().numpy()
                probs.append(prob_batch)
                if hasattr(batch, 'row_id'):
                    row_ids.extend([int(rid) for rid in batch.row_id])
                else:
                    row_ids.extend(range(len(prob_batch)))

        if not probs:
            prob = np.full(len(smiles_list), 0.5)
            pred = np.zeros(len(smiles_list), dtype=int)
            return prob, pred

        probs_array = np.concatenate(probs)

        # 映射回原始顺序
        full_probs = np.full(len(smiles_list), 0.5)
        full_preds = np.full(len(smiles_list), 0)

        for i, rid in enumerate(row_ids):
            if rid < len(full_probs):
                full_probs[rid] = float(probs_array[i])
                full_preds[rid] = int(float(probs_array[i]) >= self.threshold)

        return full_probs, full_preds

    def predict(self, smiles_list: List[str]) -> PredictionResults:
        """使用所有可用模型进行预测

        Args:
            smiles_list: SMILES字符串列表

        Returns:
            PredictionResults: 包含所有模型预测结果和集成结果
        """
        if not smiles_list:
            raise ValueError("SMILES列表不能为空")

        # 存储各模型预测结果
        all_probabilities = {}
        all_predictions = {}

        # 对每个模型进行预测
        for model_name in self.available_models.keys():
            try:
                model_data = self._load_model(model_name)

                if model_data['type'] == 'gnn':
                    prob, pred = self._predict_with_gnn_model(model_data['path'], smiles_list)
                else:
                    prob, pred = self._predict_with_rf_model(model_data['model'], smiles_list)

                all_probabilities[model_name] = prob
                all_predictions[model_name] = pred

            except Exception as e:
                print(f"Warning: {model_name} 预测失败: {e}")
                # 使用默认值填充
                all_probabilities[model_name] = np.full(len(smiles_list), 0.5)
                all_predictions[model_name] = np.zeros(len(smiles_list), dtype=int)

        # 计算集成预测
        ensemble_prob, ensemble_pred = self._ensemble_predictions(all_probabilities, all_predictions)

        # 计算模型一致性
        agreement = self._calculate_agreement(all_predictions)

        return PredictionResults(
            smiles=smiles_list,
            individual_predictions=all_predictions,
            individual_probabilities=all_probabilities,
            ensemble_prediction=ensemble_pred,
            ensemble_probability=ensemble_prob,
            agreement=agreement,
            strategy=self.strategy
        )

    def _ensemble_predictions(
        self,
        probabilities: Dict[str, np.ndarray],
        predictions: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """根据策略计算集成预测结果"""
        n_samples = len(next(iter(probabilities.values())))

        if self.strategy == EnsembleStrategy.HARD_VOTING:
            # 硬投票：多数投票
            return self._hard_voting(predictions)

        elif self.strategy == EnsembleStrategy.SOFT_VOTING:
            # 软投票：概率平均
            return self._soft_voting(probabilities)

        elif self.strategy == EnsembleStrategy.WEIGHTED:
            # 加权平均：基于性能
            return self._weighted_voting(probabilities)

        elif self.strategy == EnsembleStrategy.MAX_PROB:
            # 最大概率：取最高概率
            return self._max_prob_voting(probabilities)

        elif self.strategy == EnsembleStrategy.MIN_PROB:
            # 最小概率：最保守
            return self._min_prob_voting(probabilities)

        elif self.strategy == EnsembleStrategy.STACKING:
            # 堆叠：使用元学习器
            return self._stacking_ensemble(probabilities)

        else:
            # 默认使用硬投票
            return self._hard_voting(predictions)

    def _hard_voting(self, predictions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """硬投票：多数投票"""
        n_samples = len(next(iter(predictions.values())))
        n_models = len(predictions)

        # 计算每个样本的正例票数
        votes = np.zeros(n_samples)
        for pred in predictions.values():
            votes += pred

        # 多数投票
        ensemble_pred = (votes >= n_models / 2).astype(int)
        ensemble_prob = votes / n_models  # 正例比例作为概率

        return ensemble_prob, ensemble_pred

    def _soft_voting(self, probabilities: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """软投票：概率平均"""
        # 计算平均概率
        avg_prob = np.zeros_like(next(iter(probabilities.values())))
        for prob in probabilities.values():
            avg_prob += prob

        avg_prob /= len(probabilities)

        # 基于阈值预测
        ensemble_pred = (avg_prob >= self.threshold).astype(int)

        return avg_prob, ensemble_pred

    def _weighted_voting(self, probabilities: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """加权投票：基于模型性能"""
        # 计算权重（归一化）
        weights = np.array([
            self.available_models[name].auc
            for name in probabilities.keys()
        ])
        weights = weights / weights.sum()

        # 加权平均概率
        weighted_prob = np.zeros_like(next(iter(probabilities.values())))
        for i, (name, prob) in enumerate(probabilities.items()):
            weighted_prob += weights[i] * prob

        # 基于阈值预测
        ensemble_pred = (weighted_prob >= self.threshold).astype(int)

        return weighted_prob, ensemble_pred

    def _max_prob_voting(self, probabilities: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """最大概率投票：取每个样本的最大概率"""
        # 堆叠所有概率
        stacked = np.stack(list(probabilities.values()), axis=0)

        # 取最大值
        max_prob = np.max(stacked, axis=0)
        ensemble_pred = (max_prob >= self.threshold).astype(int)

        return max_prob, ensemble_pred

    def _min_prob_voting(self, probabilities: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """最小概率投票：最保守预测"""
        # 堆叠所有概率
        stacked = np.stack(list(probabilities.values()), axis=0)

        # 取最小值
        min_prob = np.min(stacked, axis=0)
        ensemble_pred = (min_prob >= self.threshold).astype(int)

        return min_prob, ensemble_pred

    def _stacking_ensemble(self, probabilities: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Stacking集成：使用元学习器组合基学习器

        Note: This is a simplified stacking implementation using the
        trained stacking model stored in _stacking_model.
        """
        # Stack features: each model's probability as a feature
        X_stack = np.stack(list(probabilities.values()), axis=1)

        if hasattr(self, '_stacking_model') and self._stacking_model is not None:
            # Use trained stacking model
            prob = self._stacking_model.predict_proba(X_stack)[:, 1]
        else:
            # Fallback to soft voting if stacking model not available
            prob = np.mean(X_stack, axis=1)

        ensemble_pred = (prob >= self.threshold).astype(int)
        return prob, ensemble_pred

    def fit_stacking_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        base_estimators: List[Tuple[str, str]] = None,
        meta_estimator: str = "lr",
        cv: int = 5
    ) -> "MultiModelPredictor":
        """训练Stacking集成模型

        Args:
            X_train: 训练特征 (基模型预测概率)
            y_train: 训练标签
            base_estimators: 基学习器列表 [(name, model_path), ...]
            meta_estimator: 元学习器类型 ('lr' for LogisticRegression, 'rf' for RandomForest)
            cv: 交叉验证折数

        Returns:
            self
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        # Default base estimators
        if base_estimators is None:
            base_estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
                ('lgbm', LGBMClassifier(n_estimators=100, random_state=42))
            ]

        # Meta estimator
        if meta_estimator == "lr":
            meta = LogisticRegression(max_iter=1000, random_state=42)
        elif meta_estimator == "rf":
            meta = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            meta = LogisticRegression(max_iter=1000, random_state=42)

        # Create stacking classifier
        self._stacking_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta,
            cv=cv,
            passthrough=True,  # Include original features
            n_jobs=-1
        )

        self._stacking_model.fit(X_train, y_train)
        return self

    def _calculate_agreement(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """计算模型一致性

        Returns:
            agreement: 一致性分数 (0-1)
                     1.0 = 所有模型一致
                     0.5 = 完全分裂
        """
        n_samples = len(next(iter(predictions.values())))
        n_models = len(predictions)

        agreement = np.zeros(n_samples)

        for i in range(n_samples):
            # 计算正例数量
            n_pos = sum(predictions[model][i] for model in predictions.keys())

            # 一致性 = max(正例数, 负例数) / 总模型数
            n_neg = n_models - n_pos
            agreement[i] = max(n_pos, n_neg) / n_models

        return agreement

    def get_model_info(self) -> pd.DataFrame:
        """获取可用模型信息"""
        info = []
        for name, config in self.available_models.items():
            info.append({
                'Model': name,
                'Type': config.model_type.upper(),
                'AUC': config.auc,
                'Precision': config.precision,
                'Available': 'Yes'
            })

        return pd.DataFrame(info)

    def predict_single(self, smiles: str) -> Dict:
        """预测单个分子

        Args:
            smiles: SMILES字符串

        Returns:
            dict: 包含预测结果的字典
        """
        results = self.predict([smiles])

        return {
            'SMILES': smiles,
            'ensemble_prediction': 'BBB+' if results.ensemble_prediction[0] == 1 else 'BBB-',
            'ensemble_probability': float(results.ensemble_probability[0]),
            'agreement': float(results.agreement[0]),
            'individual_predictions': {
                name: {
                    'prediction': 'BBB+' if results.individual_predictions[name][0] == 1 else 'BBB-',
                    'probability': float(results.individual_probabilities[name][0])
                }
                for name in results.individual_predictions.keys()
            },
            'strategy': self.strategy.value
        }


def create_ensemble_predictor(
    strategy: str = 'hard_voting',
    threshold: float = 0.5,
    seed: int = 0,
    models: Optional[List[str]] = None
) -> MultiModelPredictor:
    """便捷函数：创建集成预测器

    Args:
        strategy: 集成策略 ('hard_voting', 'soft_voting', 'weighted', 'max_prob', 'min_prob')
        threshold: 分类阈值
        seed: 随机种子
        models: 要使用的模型列表（默认使用所有可用模型）

    Returns:
        MultiModelPredictor: 集成预测器实例
    """
    strategy_enum = EnsembleStrategy(strategy)

    # 如果指定了模型，只使用这些模型
    if models:
        selected_models = {
            name: MultiModelPredictor.DEFAULT_MODELS[name]
            for name in models
            if name in MultiModelPredictor.DEFAULT_MODELS
        }
    else:
        selected_models = None

    return MultiModelPredictor(
        seed=seed,
        strategy=strategy_enum,
        threshold=threshold,
        models=selected_models
    )


if __name__ == "__main__":
    # 测试代码
    print("Multi-Model BBB Predictor")
    print("=" * 50)

    # 创建预测器
    predictor = create_ensemble_predictor(
        strategy='hard_voting',
        threshold=0.5
    )

    print("\n可用模型:")
    print(predictor.get_model_info())

    # 测试预测
    test_smiles = [
        "CCO",  # 乙醇
        "CC(=O)OC1=CC=C(C=C)C=C1",  # 阿司匹林
        "CCN",  # 乙胺
        "c1ccccc1"  # 苯
    ]

    print("\n测试分子:")
    for smi in test_smiles:
        print(f"  - {smi}")

    print("\n预测中...")
    results = predictor.predict(test_smiles)

    print("\n集成预测结果:")
    df = results.to_dataframe()
    print(df[['SMILES', 'ensemble_prob', 'ensemble_pred', 'agreement', 'consensus']])

    print("\n摘要:")
    summary = results.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
