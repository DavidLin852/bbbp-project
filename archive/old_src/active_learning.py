"""
Active Learning Module for BBB Prediction

功能：
1. 检查SMILES是否存在于训练数据中
2. 如果存在，返回其标签
3. 如果不存在，进行预测
4. 支持添加新标注数据
5. 创建新数据集并重新训练模型
"""
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import json
from datetime import datetime
import shutil

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Paths, DatasetConfig
from src.multi_model_predictor import create_ensemble_predictor


@dataclass
class DataCheckResult:
    """数据检查结果"""
    exists: bool
    smiles: str
    label: Optional[int]  # 1 for BBB+, 0 for BBB-
    label_str: Optional[str]  # 'BBB+' or 'BBB-'
    split: Optional[str]  # 'train', 'val', 'test'
    source_file: Optional[str]  # 来源文件


@dataclass
class NewAnnotation:
    """新标注数据"""
    smiles: str
    label: int
    label_str: str
    timestamp: str
    predicted_label: Optional[int] = None
    predicted_probability: Optional[float] = None
    annotation_source: str = "manual"  # 'manual' or 'predicted'


class ActiveLearningManager:
    """主动学习管理器"""

    def __init__(
        self,
        seed: int = 0,
        project_root: Optional[Path] = None
    ):
        """初始化主动学习管理器

        Args:
            seed: 随机种子
            project_root: 项目根目录
        """
        self.seed = seed
        self.project_root = project_root or PROJECT_ROOT
        self.paths = Paths()

        # 加载现有数据
        self.train_data = self._load_split_data('train')
        self.val_data = self._load_split_data('val')
        self.test_data = self._load_split_data('test')

        # 合并所有数据用于快速查找
        self.all_data = pd.concat([
            self.train_data.assign(split='train'),
            self.val_data.assign(split='val'),
            self.test_data.assign(split='test')
        ], ignore_index=True)

        # 创建SMILES到标签的映射
        self.smiles_to_label = dict(zip(
            self.all_data[DatasetConfig.smiles_col],
            zip(
                self.all_data['y_cls'],
                self.all_data['split']
            )
        ))

        # 新标注数据存储
        self.new_annotations: List[NewAnnotation] = []

        # 创建预测器
        self.predictor = create_ensemble_predictor(
            strategy='hard_voting',
            threshold=0.5
        )

    def _load_split_data(self, split: str) -> pd.DataFrame:
        """加载指定split的数据"""
        split_file = self.paths.data_splits / f"seed_{self.seed}" / f"{split}.csv"

        if not split_file.exists():
            return pd.DataFrame()

        return pd.read_csv(split_file)

    def check_smiles(self, smiles: str) -> DataCheckResult:
        """检查SMILES是否存在于训练数据中

        Args:
            smiles: SMILES字符串

        Returns:
            DataCheckResult: 检查结果
        """
        smiles = smiles.strip()

        if smiles in self.smiles_to_label:
            label, split = self.smiles_to_label[smiles]
            return DataCheckResult(
                exists=True,
                smiles=smiles,
                label=int(label),
                label_str='BBB+' if int(label) == 1 else 'BBB-',
                split=split,
                source_file=f"{split}.csv"
            )
        else:
            return DataCheckResult(
                exists=False,
                smiles=smiles,
                label=None,
                label_str=None,
                split=None,
                source_file=None
            )

    def predict_smiles(self, smiles: str) -> Tuple[int, float, Dict]:
        """预测单个SMILES

        Args:
            smiles: SMILES字符串

        Returns:
            (prediction, probability, individual_results)
        """
        result = self.predictor.predict_single(smiles)

        pred_int = 1 if result['ensemble_prediction'] == 'BBB+' else 0
        prob = result['ensemble_probability']

        return pred_int, prob, result['individual_predictions']

    def add_annotation(
        self,
        smiles: str,
        label: int,
        predicted_label: Optional[int] = None,
        predicted_probability: Optional[float] = None
    ):
        """添加新标注

        Args:
            smiles: SMILES字符串
            label: 标签 (1 for BBB+, 0 for BBB-)
            predicted_label: 预测标签（可选）
            predicted_probability: 预测概率（可选）
        """
        annotation = NewAnnotation(
            smiles=smiles.strip(),
            label=int(label),
            label_str='BBB+' if int(label) == 1 else 'BBB-',
            timestamp=datetime.now().isoformat(),
            predicted_label=predicted_label,
            predicted_probability=predicted_probability,
            annotation_source="manual"
        )

        # 检查是否重复
        for existing in self.new_annotations:
            if existing.smiles == smiles:
                # 更新现有标注
                existing.label = label
                existing.label_str = annotation.label_str
                existing.timestamp = annotation.timestamp
                return

        # 添加新标注
        self.new_annotations.append(annotation)

    def get_new_annotations_dataframe(self) -> pd.DataFrame:
        """获取新标注的DataFrame"""
        if not self.new_annotations:
            return pd.DataFrame()

        data = []
        for ann in self.new_annotations:
            row = {
                'SMILES': ann.smiles,
                'BBB+/BBB-': ann.label_str,
                'y_cls': ann.label,
                'timestamp': ann.timestamp,
                'predicted_label': 'BBB+' if ann.predicted_label == 1 else 'BBB-' if ann.predicted_label == 0 else 'N/A',
                'predicted_probability': ann.predicted_probability,
                'annotation_source': ann.annotation_source
            }
            data.append(row)

        return pd.DataFrame(data)

    def save_new_annotations(self, dataset_name: str) -> Path:
        """保存新标注到新数据集

        Args:
            dataset_name: 新数据集名称

        Returns:
            保存的文件路径
        """
        # 创建新数据集目录
        new_dataset_dir = self.paths.data_splits.parent / "custom_datasets" / dataset_name
        new_dataset_dir.mkdir(parents=True, exist_ok=True)

        # 合并原始数据和新标注
        new_data = self.all_data.copy()

        # 添加新标注数据（添加到训练集）
        new_annotations_list = []
        for ann in self.new_annotations:
            new_annotations_list.append({
                DatasetConfig.smiles_col: ann.smiles,
                'y_cls': ann.label,
                'group': 'custom',  # 标记为自定义数据
                'source': 'active_learning'
            })

        if new_annotations_list:
            new_annotations_df = pd.DataFrame(new_annotations_list)
            new_data = pd.concat([new_data, new_annotations_df], ignore_index=True)

        # 保存完整数据集
        output_file = new_dataset_dir / "full_dataset.csv"
        new_data.to_csv(output_file, index=False)

        # 保存新标注记录
        annotations_file = new_dataset_dir / "new_annotations.csv"
        self.get_new_annotations_dataframe().to_csv(annotations_file, index=False)

        # 保存元数据
        metadata = {
            'dataset_name': dataset_name,
            'created_at': datetime.now().isoformat(),
            'seed': self.seed,
            'original_data_size': len(self.all_data),
            'new_annotations_count': len(self.new_annotations),
            'total_size': len(new_data),
            'original_train_size': len(self.train_data),
            'new_train_size': len(self.train_data) + len(self.new_annotations)
        }

        metadata_file = new_dataset_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return output_file

    def prepare_retrain_splits(
        self,
        dataset_name: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[Path, Path, Path]:
        """准备重新训练的数据划分

        Args:
            dataset_name: 数据集名称
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例

        Returns:
            (train_file, val_file, test_file)
        """
        from sklearn.model_selection import train_test_split

        # 加载完整数据集
        dataset_dir = self.paths.data_dir / "custom_datasets" / dataset_name
        full_data = pd.read_csv(dataset_dir / "full_dataset.csv")

        # 分离原始测试集（保持不变）
        original_test = full_data[full_data['split'] == 'test'].copy()
        new_data = full_data[full_data['split'] != 'test'].copy()

        # 对新数据进行划分（包含原始训练集、验证集和新标注数据）
        X = new_data[DatasetConfig.smiles_col].values
        y = new_data['y_cls'].values

        # 第一次划分：训练集 vs 临时集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            train_size=train_ratio / (train_ratio + val_ratio),
            random_state=self.seed,
            stratify=y
        )

        # 第二次划分：验证集 vs 测试集
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=self.seed,
            stratify=y_temp
        )

        # 创建DataFrame
        train_df = pd.DataFrame({
            DatasetConfig.smiles_col: X_train,
            'y_cls': y_train
        })

        val_df = pd.DataFrame({
            DatasetConfig.smiles_col: X_val,
            'y_cls': y_val
        })

        test_df = pd.DataFrame({
            DatasetConfig.smiles_col: X_test,
            'y_cls': y_test
        })

        # 合并原始测试集（如果有的话）
        if len(original_test) > 0:
            original_test_renamed = pd.DataFrame({
                DatasetConfig.smiles_col: original_test[DatasetConfig.smiles_col],
                'y_cls': original_test['y_cls']
            })
            test_df = pd.concat([test_df, original_test_renamed], ignore_index=True)

        # 保存划分文件
        splits_dir = dataset_dir / "splits"
        splits_dir.mkdir(exist_ok=True)

        train_file = splits_dir / "train.csv"
        val_file = splits_dir / "val.csv"
        test_file = splits_dir / "test.csv"

        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)

        # 保存划分信息
        split_info = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'total_size': len(full_data),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'seed': self.seed
        }

        info_file = splits_dir / "split_info.json"
        with open(info_file, 'w') as f:
            json.dump(split_info, f, indent=2)

        return train_file, val_file, test_file

    def get_statistics(self) -> Dict:
        """获取数据统计信息"""
        return {
            'original_train_size': len(self.train_data),
            'original_val_size': len(self.val_data),
            'original_test_size': len(self.test_data),
            'original_total_size': len(self.all_data),
            'new_annotations_count': len(self.new_annotations),
            'potential_new_train_size': len(self.train_data) + len(self.new_annotations),
            'unique_smiles_in_training': self.all_data[DatasetConfig.smiles_col].nunique()
        }


def create_active_learning_manager(seed: int = 0) -> ActiveLearningManager:
    """便捷函数：创建主动学习管理器

    Args:
        seed: 随机种子

    Returns:
        ActiveLearningManager实例
    """
    return ActiveLearningManager(seed=seed)


if __name__ == "__main__":
    # 测试代码
    print("Active Learning Manager Test")
    print("=" * 50)

    manager = create_active_learning_manager(seed=0)

    # 测试1：检查已存在的SMILES
    print("\n测试1：检查已存在的SMILES")
    test_smiles_existing = "CCO"  # 假设这个存在于数据中
    result = manager.check_smiles(test_smiles_existing)
    print(f"SMILES: {test_smiles_existing}")
    print(f"存在: {result.exists}")
    if result.exists:
        print(f"标签: {result.label_str}")
        print(f"划分: {result.split}")

    # 测试2：检查不存在的SMILES
    print("\n测试2：检查不存在的SMILES")
    test_smiles_new = "C1=CC=CC=C1C=O"  # 苯甲醛
    result = manager.check_smiles(test_smiles_new)
    print(f"SMILES: {test_smiles_new}")
    print(f"存在: {result.exists}")

    if not result.exists:
        print("进行预测...")
        pred, prob, individual = manager.predict_smiles(test_smiles_new)
        print(f"预测: {'BBB+' if pred == 1 else 'BBB-'}")
        print(f"概率: {prob:.3f}")

        # 添加标注
        print("\n添加标注...")
        manager.add_annotation(test_smiles_new, label=1, predicted_label=pred, predicted_probability=prob)
        print("标注已添加")

    # 测试3：查看统计信息
    print("\n测试3：统计信息")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 测试4：查看新标注
    print("\n测试4：新标注")
    new_df = manager.get_new_annotations_dataframe()
    print(new_df)
