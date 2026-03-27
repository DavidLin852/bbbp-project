"""
SHAP Explainability for BBB Permeability Prediction

Provides SHAP-based explainability for tree-based models (RF, XGB, LGBM)
and deep learning models (Transformer, GNN).

Features:
- SHAP values calculation using TreeExplainer
- SHAP summary plots (beeswarm, bar, violin)
- SHAP dependence plots for top features
- Feature importance ranking
- Toxicophore identification
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ModelType(str, Enum):
    """支持的模型类型"""
    RF = "rf"
    XGB = "xgb"
    LGBM = "lgbm"
    TRANSFORMER = "transformer"
    GNN = "gnn"
    ENSEMBLE = "ensemble"


@dataclass
class SHAPConfig:
    """SHAP分析配置"""
    n_background_samples: int = 100
    n_test_samples: int = 100
    feature_names: List[str] = None
    output_dir: Path = PROJECT_ROOT / "outputs" / "shap"
    plot_type: str = "summary"  # "summary", "dependence", "force", "bar"
    class_label: str = "BBB+"  # 解释正类


class SHAPExplainer:
    """SHAP解释器类

    支持多种模型类型的SHAP值计算和可视化。
    """

    def __init__(self, model: BaseEstimator, model_type: ModelType,
                 feature_names: List[str] = None):
        """初始化SHAP解释器

        Args:
            model: 训练好的模型
            model_type: 模型类型
            feature_names: 特征名称列表
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

        # 初始化解释器
        self._init_explainer()

    def _init_explainer(self):
        """根据模型类型初始化SHAP解释器"""
        if self.model_type == ModelType.RF:
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == ModelType.XGB:
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == ModelType.LGBM:
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == ModelType.TRANSFORMER:
            # 对于Transformer，使用KernelExplainer作为近似
            # 使用训练数据的子集作为背景
            self.explainer = shap.KernelExplainer(
                self._model_predict,
                self._get_background_data()
            )
        elif self.model_type == ModelType.GNN:
            # 对于GNN，使用类似方法
            self.explainer = shap.KernelExplainer(
                self._model_predict,
                self._get_background_data()
            )
        else:
            # 默认使用KernelExplainer
            self.explainer = shap.KernelExplainer(
                self._model_predict,
                self._get_background_data()
            )

    def _model_predict(self, X: np.ndarray) -> np.ndarray:
        """模型预测函数（用于KernelExplainer）"""
        return self.model.predict_proba(X)[:, 1]

    def _get_background_data(self, n_samples: int = 100) -> np.ndarray:
        """获取背景数据用于KernelExplainer"""
        # 如果模型有train方法，可以使用训练数据的子集
        if hasattr(self.model, 'train_data'):
            X_train = self.model.train_data[:n_samples]
        else:
            # 默认返回零矩阵
            X_train = np.zeros((n_samples, len(self.feature_names) if self.feature_names else 100))
        return X_train

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """计算SHAP值

        Args:
            X: 输入特征矩阵 (n_samples, n_features)

        Returns:
            SHAP值矩阵
        """
        if self.explainer is None:
            self._init_explainer()

        if self.model_type in [ModelType.RF, ModelType.XGB, ModelType.LGBM]:
            # TreeExplainer直接返回SHAP值
            self.shap_values = self.explainer.shap_values(X)
        else:
            # KernelExplainer需要计算
            self.shap_values = self.explainer.shap_values(X)

        return self.shap_values

    def get_feature_importance(self, X: np.ndarray = None,
                               sort_by: str = "mean_abs") -> pd.DataFrame:
        """获取特征重要性排名

        Args:
            X: 可选的输入数据，如果未提供则使用之前计算的SHAP值
            sort_by: 排序方式 ("mean_abs", "positive", "negative")

        Returns:
            包含特征重要性信息的DataFrame
        """
        if self.shap_values is None and X is None:
            raise ValueError("需要提供X或先计算SHAP值")

        if self.shap_values is None:
            self.compute_shap_values(X)

        # 计算各特征的SHAP统计量
        importance_df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else [f'f{i}' for i in range(self.shap_values.shape[1])],
            'mean_abs_shap': np.abs(self.shap_values).mean(axis=0),
            'mean_shap': self.shap_values.mean(axis=0),
            'std_shap': self.shap_values.std(axis=0),
            'max_shap': np.abs(self.shap_values).max(axis=0),
            'min_shap': self.shap_values.min(axis=0),
            'positive_contribution': (self.shap_values > 0).mean(axis=0),
            'negative_contribution': (self.shap_values < 0).mean(axis=0)
        })

        # 排序
        if sort_by == "mean_abs":
            importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
        elif sort_by == "positive":
            importance_df = importance_df.sort_values('positive_contribution', ascending=False)
        elif sort_by == "negative":
            importance_df = importance_df.sort_values('negative_contribution', ascending=False)

        return importance_df.reset_index(drop=True)

    def get_toxicophores(self, X: np.ndarray, threshold: float = 0.01) -> List[Dict]:
        """识别与BBB通透性相关的毒性基团/药效团

        分析哪些特征（指纹位点/描述符）对预测有显著影响，
        并解释其物理化学意义。

        Args:
            X: 输入特征矩阵
            threshold: 显著性阈值

        Returns:
            毒性基团信息列表
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        # 获取特征重要性
        importance = self.get_feature_importance(X)

        # 过滤显著特征
        significant = importance[importance['mean_abs_shap'] > threshold]

        # 识别正/负贡献的基团
        toxicophores = []
        for _, row in significant.iterrows():
            feature_info = {
                'feature': row['feature'],
                'mean_shap': row['mean_shap'],
                'mean_abs_shap': row['mean_abs_shap'],
                'positive_freq': row['positive_contribution'],
                'negative_freq': row['negative_contribution'],
                'impact': 'BBB+' if row['mean_shap'] > 0 else 'BBB-',
                'significance': 'high' if row['mean_abs_shap'] > 0.05 else 'medium' if row['mean_abs_shap'] > 0.02 else 'low'
            }
            toxicophores.append(feature_info)

        return toxicophores

    def plot_summary(self, X: np.ndarray, output_path: Path = None,
                      plot_type: str = "beeswarm", max_features: int = 20):
        """绘制SHAP summary图

        Args:
            X: 输入特征矩阵
            output_path: 输出路径
            plot_type: 图表类型 ("beeswarm", "bar", "violin", "dot")
            max_features: 显示的最大特征数
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_features,
            show=False,
            plot_type=plot_type
        )
        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"SHAP summary plot saved to: {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_dependence(self, X: np.ndarray, feature_idx: int,
                        interaction_idx: int = None,
                        output_path: Path = None,
                        alpha: float = 0.5):
        """绘制SHAP dependence图

        Args:
            X: 输入特征矩阵
            feature_idx: 特征索引
            interaction_idx: 交互特征索引
            output_path: 输出路径
            alpha: 散点透明度
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        feature_name = self.feature_names[feature_idx] if self.feature_names else f'f{feature_idx}'

        plt.figure(figsize=(10, 8))
        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_idx,
            show=False,
            alpha=alpha
        )
        plt.title(f'SHAP Dependence: {feature_name}')
        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"SHAP dependence plot saved to: {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_feature_importance(self, X: np.ndarray, output_path: Path = None,
                                 top_n: int = 20):
        """绘制特征重要性条形图

        Args:
            X: 输入特征矩阵
            output_path: 输出路径
            top_n: 显示Top N特征
        """
        importance = self.get_feature_importance(X)
        top_features = importance.head(top_n)

        plt.figure(figsize=(12, 8))
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_features['mean_shap']]

        plt.barh(range(len(top_features)), top_features['mean_abs_shap'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mean |SHAP Value|')
        plt.ylabel('Feature')
        plt.title('Top Features by SHAP Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {output_path}")
        else:
            plt.show()

        plt.close()

    def explain_instance(self, X_instance: np.ndarray) -> Dict:
        """解释单个实例的预测

        Args:
            X_instance: 单个实例的特征向量

        Returns:
            包含SHAP解释的字典
        """
        if self.shap_values is None:
            self.compute_shap_values(X_instance.reshape(1, -1))

        shap_vals = self.shap_values[0] if self.shap_values.ndim > 1 else self.shap_values

        # 获取按绝对值排序的特征
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]

        contributions = []
        for i in sorted_idx[:10]:  # Top 10贡献特征
            contributions.append({
                'feature': self.feature_names[i] if self.feature_names else f'f{i}',
                'value': float(X_instance[i]),
                'shap_value': float(shap_vals[i]),
                'direction': 'BBB+' if shap_vals[i] > 0 else 'BBB-'
            })

        return {
            'base_value': float(self.explainer.expected_value[1]) if hasattr(self.explainer.expected_value, '__len__') else float(self.explainer.expected_value),
            'prediction': float(self.model.predict_proba(X_instance.reshape(1, -1))[0, 1]),
            'top_contributions': contributions
        }

    def save_shap_values(self, X: np.ndarray, output_path: Path):
        """保存SHAP值到文件

        Args:
            X: 输入特征矩阵
            output_path: 输出文件路径
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(output_path, self.shap_values)
        print(f"SHAP values saved to: {output_path}")


def explain_model(model: BaseEstimator, model_type: ModelType,
                  X: np.ndarray, feature_names: List[str] = None,
                  config: SHAPConfig = None) -> Tuple[SHAPExplainer, pd.DataFrame]:
    """便捷函数：解释模型预测

    Args:
        model: 训练好的模型
        model_type: 模型类型
        X: 特征矩阵
        feature_names: 特征名称
        config: SHAP配置

    Returns:
        Tuple of (SHAPExplainer, importance DataFrame)
    """
    explainer = SHAPExplainer(model, model_type, feature_names)
    shap_values = explainer.compute_shap_values(X)
    importance = explainer.get_feature_importance(X)

    return explainer, importance


def identify_toxicophores_from_smarts(
    shap_values: np.ndarray,
    smarts_patterns: Dict[str, str],
    feature_names: List[str] = None
) -> pd.DataFrame:
    """从SMARTS模式中识别毒性基团

    Args:
        shap_values: SHAP值矩阵
        smarts_patterns: SMARTS模式字典 {name: pattern}
        feature_names: 特征名称

    Returns:
        毒性基团DataFrame
    """
    # 计算每个SMARTS模式的平均SHAP值
    smarts_importance = []

    for name, pattern in smarts_patterns.items():
        idx = None
        if feature_names:
            try:
                idx = feature_names.index(name)
            except ValueError:
                continue

        if idx is not None and idx < len(shap_values[0]):
            mean_shap = shap_values[:, idx].mean()
            std_shap = shap_values[:, idx].std()
            pos_freq = (shap_values[:, idx] > 0).mean()
            neg_freq = (shap_values[:, idx] < 0).mean()

            smarts_importance.append({
                'smarts_pattern': name,
                'pattern': pattern,
                'mean_shap': mean_shap,
                'std_shap': std_shap,
                'positive_frequency': pos_freq,
                'negative_frequency': neg_freq,
                'overall_impact': 'BBB+' if mean_shap > 0 else 'BBB-',
                'significance': abs(mean_shap)
            })

    df = pd.DataFrame(smarts_importance)
    df = df.sort_values('significance', ascending=False)
    return df


# =============================================================================
# Toxicophore Mapping
# =============================================================================

# 常见与BBB通透性相关的结构特征
COMMON_TOXICOPHORES = {
    'aromatic_ring': {
        'pattern': 'c1ccccc1',
        'description': '苯环/芳香环',
        'typical_impact': 'variable',
        'notes': '芳香性通常增加BBB通透性'
    },
    'heteroaromatic': {
        'pattern': 'c1ccncc1',
        'description': '含氮杂芳香环',
        'typical_impact': 'BBB+',
        'notes': '吡啶、吡咯等杂环可增强通透性'
    },
    'alkyl_halide': {
        'pattern': 'CCl',
        'description': '卤代烷基',
        'typical_impact': 'BBB+',
        'notes': '卤素取代通常增加脂溶性'
    },
    'amine_primary': {
        'pattern': 'CN',
        'description': '伯胺',
        'typical_impact': 'BBB-',
        'notes': '质子化后难以透过BBB'
    },
    'amine_tertiary': {
        'pattern': 'N(C)(C)',
        'description': '叔胺',
        'typical_impact': 'BBB+',
        'notes': '中性形式更易透过BBB'
    },
    'ether': {
        'pattern': 'COC',
        'description': '醚键',
        'typical_impact': 'BBB+',
        'notes': '增加分子柔性'
    },
    'hydroxyl': {
        'pattern': 'CO',
        'description': '羟基',
        'typical_impact': 'BBB-',
        'notes': '形成氢键，降低通透性'
    },
    'carbonyl': {
        'pattern': 'C=O',
        'description': '羰基',
        'typical_impact': 'variable',
        'notes': '形成氢键受体'
    },
    'carboxylic_acid': {
        'pattern': 'C(=O)O',
        'description': '羧基',
        'typical_impact': 'BBB-',
        'notes': '质子化形式，难以透过'
    },
    'ester': {
        'pattern': 'C(=O)OC',
        'description': '酯基',
        'typical_impact': 'BBB+',
        'notes': '中等脂溶性'
    },
    'amide': {
        'pattern': 'C(=O)N',
        'description': '酰胺',
        'typical_impact': 'BBB-',
        'notes': '可形成氢键'
    },
    'nitrile': {
        'pattern': 'C#N',
        'description': '腈基',
        'typical_impact': 'BBB+',
        'notes': '增加极性'
    },
    'nitro': {
        'pattern': '[N+](=O)[O-]',
        'description': '硝基',
        'typical_impact': 'BBB-',
        'notes': '强吸电子基团'
    },
    'sulfonamide': {
        'pattern': 'S(=O)(=O)N',
        'description': '磺酰胺',
        'typical_impact': 'BBB-',
        'notes': '极性大，难以透过'
    },
    'thioether': {
        'pattern': 'CSC',
        'description': '硫醚',
        'typical_impact': 'BBB+',
        'notes': '增加脂溶性'
    }
}


def map_shap_to_toxicophores(shap_values: np.ndarray,
                               feature_names: List[str],
                               smarts_vocab: Dict[str, str] = None) -> pd.DataFrame:
    """将SHAP值映射到毒性基团

    Args:
        shap_values: SHAP值矩阵
        feature_names: 特征名称
        smarts_vocab: SMARTS词汇表

    Returns:
        毒性基团SHAP分析结果
    """
    if smarts_vocab is None:
        smarts_vocab = COMMON_TOXICOPHORES

    # 创建特征到索引的映射
    feature_to_idx = {f: i for i, f in enumerate(feature_names)}

    results = []
    for name, info in smarts_vocab.items():
        idx = feature_to_idx.get(name)
        if idx is not None and idx < shap_values.shape[1]:
            mean_shap = shap_values[:, idx].mean()
            std_shap = shap_values[:, idx].std()

            results.append({
                'toxicophore': name,
                'description': info['description'],
                'typical_impact': info['typical_impact'],
                'observed_impact': 'BBB+' if mean_shap > 0 else 'BBB-',
                'mean_shap': mean_shap,
                'std_shap': std_shap,
                'abs_impact': abs(mean_shap),
                'notes': info['notes']
            })

    df = pd.DataFrame(results)
    df = df.sort_values('abs_impact', ascending=False)
    return df


if __name__ == "__main__":
    # 示例用法
    from pathlib import Path
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # 创建示例数据
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=20,
                                 n_redundant=10, random_state=42)

    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 创建特征名称
    feature_names = [f'feature_{i}' for i in range(100)]

    # 创建SHAP解释器
    explainer = SHAPExplainer(model, ModelType.RF, feature_names)
    shap_values = explainer.compute_shap_values(X)

    # 获取特征重要性
    importance = explainer.get_feature_importance(X)
    print("Top 10 important features:")
    print(importance.head(10))

    # 绘制summary图
    explainer.plot_summary(X, Path("outputs/shap_summary.png"))

    # 绘制特征重要性
    explainer.plot_feature_importance(X, Path("outputs/shap_importance.png"))

    print("\nSHAP analysis complete!")
