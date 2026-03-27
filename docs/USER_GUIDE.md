# BBB渗透性预测平台 - 完整使用指南

> **最后更新**: 2026-02-24
> **版本**: 2.0
> **项目状态**: 生产就绪

---

## 目录

1. [项目简介](#1-项目简介)
2. [环境配置](#2-环境配置)
3. [数据准备](#3-数据准备)
4. [模型训练流程](#4-模型训练流程)
5. [模型预测](#5-模型预测)
6. [Web应用使用](#6-web应用使用)
7. [高级功能](#7-高级功能)
8. [常见问题](#8-常见问题)
9. [API参考](#9-api参考)

---

## 1. 项目简介

### 1.1 项目概述

BBB渗透性预测平台是一个基于机器学习的血脑屏障（Blood-Brain Barrier, BBB）渗透性预测系统。该平台使用B3DB数据集训练了多种机器学习模型，包括传统机器学习模型（RF、XGB、LGBM）、图神经网络（GAT）以及集成模型。

### 1.2 核心特性

- **32个训练好的模型** × 4个数据集 = 完整覆盖
- **多种集成策略**: Hard Voting、Soft Voting、Stacking等
- **SMARTS子结构分析**: 化学结构-活性关系解释
- **多种分子指纹**: Morgan、MACCS、AtomPairs、FP2、RDKit描述符
- **可视化分析**: 降维可视化、ROC曲线、模型对比热力图
- **批量预测**: 支持单个/批量SMILES预测

### 1.3 项目结构

```
bbb_project/
├── app_bbb_predict.py          # 主Web应用入口
├── src/                        # 源代码模块
│   ├── config.py               # 集中式配置
│   ├── baseline/               # 基础模型训练
│   ├── featurize/              # 特征提取
│   ├── pretrain/               # GNN预训练
│   ├── finetune/               # GNN微调
│   ├── transformer/            # Transformer模型
│   └── utils/                  # 工具函数
├── scripts/                    # 训练脚本（01-18）
├── pages/                      # Streamlit页面（0-6）
├── data/                       # 数据目录
├── artifacts/                  # 训练产物（模型、特征、指标）
├── outputs/                    # 输出结果（图像、文档）
└── assets/                     # 资源文件（SMARTS模式等）
```

---

## 2. 环境配置

### 2.1 系统要求

- **Python**: 3.8+
- **操作系统**: Windows/Linux/macOS
- **内存**: 8GB+ (推荐16GB)
- **存储**: 5GB+ 可用空间

### 2.2 依赖安装

#### 使用Conda（推荐）

```bash
# 创建conda环境
conda create -n bbb python=3.9
conda activate bbb

# 安装PyTorch（根据你的CUDA版本调整）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装RDKit
conda install -c conda-forge rdkit

# 安装其他依赖
pip install streamlit pandas numpy scipy scikit-learn xgboost lightgbm
pip install torch-geometric matplotlib seaborn plotly
pip install joblib
```

#### 使用pip

```bash
pip install rdkit-pypi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install streamlit pandas numpy scipy scikit-learn xgboost lightgbm
pip install matplotlib seaborn plotly joblib
```

### 2.3 验证安装

```bash
# 测试RDKit
python -c "from rdkit import Chem; print('RDKit OK')"

# 测试PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# 测试PyG
python -c "import torch_geometric; print('PyG OK')"
```

---

## 3. 数据准备

### 3.1 B3DB数据集

平台使用B3DB（Blood-Brain Barrier Database）数据集，这是一个经过精心整理的BBB渗透性数据集。

#### 数据集结构

```csv
NO.    CID    compound_name    SMILES    group    BBB+/BBB-    logBB
1      1      Example1         CCO       A        BBB+         0.5
2      2      Example2         CCN       B        BBB-         -1.2
...
```

#### 数据分组

| 分组 | 描述 | 样本数 | 正例率 | 推荐用途 |
|------|------|--------|--------|----------|
| **A** | 高质量数据 | 846 | 87.7% | 最高精度，最低假阳性 |
| **A,B** | 推荐数据集 | 3,743 | 76.5% | 最佳综合性能 ⭐ |
| **A,B,C** | 扩展数据集 | 6,203 | 66.7% | 大规模覆盖 |
| **A,B,C,D** | 完整数据集 | 6,244 | 63.5% | 最大数据覆盖 |

### 3.2 数据划分

使用分层划分（stratified split）确保训练/验证/测试集的标签分布一致：

- **训练集**: 80%
- **验证集**: 10%
- **测试集**: 10%

### 3.3 数据预处理

运行数据划分脚本：

```bash
# 使用默认种子和分组（seed=0, groups=A,B）
python scripts/01_prepare_splits.py --seed 0 --keep_groups "A,B"

# 使用不同种子进行多次实验
python scripts/01_prepare_splits.py --seed 1 --keep_groups "A,B,C"
python scripts/01_prepare_splits.py --seed 2 --keep_groups "A,B,C,D"
```

**输出**:
```
data/splits/seed_0/
├── train.csv
├── val.csv
├── test.csv
└── split_report.json
```

---

## 4. 模型训练流程

### 4.1 完整训练流程

```
步骤1: 数据划分 → 步骤2: 特征提取 → 步骤3: 基础模型训练
   ↓
步骤4: GAT训练 → 步骤5: SMARTS预训练 → 步骤6: BBB微调
   ↓
步骤7-13: 评估与可视化
```

### 4.2 特征提取

#### 提取所有特征

```bash
# 提取Morgan指纹、RDKit描述符、图特征
python scripts/02_featurize_all.py --seed 0
```

**输出特征**:

| 特征类型 | 维度 | 描述 | 文件 |
|----------|------|------|------|
| Morgan指纹 | 2048 | ECFP4-like，radius=2 | morgan_2048.npz |
| MACCS键 | 167 | 预定义化学子结构 | maccs.npz |
| Atom Pairs | 1024 | 拓扑距离指纹 | atom_pairs_1024.npz |
| FP2 | 2048 | Daylight类指纹 | fp2_2048.npz |
| RDKit描述符 | 39-200 | 分子描述符 | descriptors.csv |
| Combined | 5287 | 4种指纹组合 | combined.npz |
| Graph图特征 | - | PyG分子图 | pyg_graphs_baseline/ |

### 4.3 基础模型训练

训练RF、XGB、LGBM三个基础模型：

```bash
# 使用Morgan指纹
python scripts/03_run_baselines.py --seed 0 --feature morgan

# 使用RDKit描述符
python scripts/03_run_baselines.py --seed 0 --feature desc
```

**输出**:
```
artifacts/models/seed_0_full/baseline/
├── RF_seed0.joblib
├── XGB_seed0.joblib
├── LGBM_seed0.joblib
└── *_metrics.json
```

### 4.4 GAT模型训练

#### 无预训练GAT（基线）

```bash
python scripts/04_run_gat_aux.py --seed 0 --dataset A_B
```

#### SMARTS预训练 + BBB微调

```bash
# 步骤1: SMARTS预训练
python scripts/05_pretrain_smarts.py --seed 0 --dataset A_B

# 步骤2: BBB微调
python scripts/06_finetune_bbb_from_smarts.py --seed 0 --dataset A_B --init pretrained --strategy freeze
```

### 4.5 模型消融研究

训练SMARTS增强版本的基础模型：

```bash
python scripts/15_ablate_smarts_on_model.py \
    --seed 0 \
    --dataset A_B \
    --model RF \
    --feature morgan
```

### 4.6 多种子实验

为了获得更可靠的性能估计，建议使用多个随机种子：

```bash
# 使用5个种子进行完整训练
for seed in 0 1 2 3 4; do
    python scripts/01_prepare_splits.py --seed $seed
    python scripts/02_featurize_all.py --seed $seed
    python scripts/03_run_baselines.py --seed $seed --feature morgan
    python scripts/04_run_gat_aux.py --seed $seed --dataset A_B
    python scripts/05_pretrain_smarts.py --seed $seed --dataset A_B
    python scripts/06_finetune_bbb_from_smarts.py --seed $seed --dataset A_B
done
```

---

## 5. 模型预测

### 5.1 命令行预测

使用单个模型进行预测：

```bash
python scripts/14_predict_smiles.py \
    --smiles "CCO" \
    --model RF \
    --dataset A_B \
    --seed 0
```

批量预测：

```bash
python scripts/14_predict_smiles.py \
    --input molecules.csv \
    --model RF+SMARTS \
    --dataset A_B \
    --output predictions.csv
```

### 5.2 Python API预测

#### 单个模型预测

```python
from src.multi_model_predictor import MultiModelPredictor, EnsembleStrategy

# 创建预测器
predictor = MultiModelPredictor(
    seed=0,
    strategy=EnsembleStrategy.SOFT_VOTING,
    threshold=0.5
)

# 单个预测
result = predictor.predict_single("CCO")
print(f"预测: {result['ensemble_prediction']}")
print(f"概率: {result['ensemble_probability']:.3f}")
```

#### 批量预测

```python
smiles_list = ["CCO", "CC(=O)OC1=CC=C(C=C)C=C1", "CCN"]

results = predictor.predict(smiles_list)

# 转换为DataFrame
df = results.to_dataframe()
print(df[['SMILES', 'ensemble_prob', 'ensemble_pred', 'agreement']])

# 获取摘要
summary = results.get_summary()
print(f"预测为BBB+: {summary['predicted_bbb_plus']}/{summary['total_samples']}")
```

#### 使用指定模型

```python
from pathlib import Path
from src.multi_model_predictor import ModelConfig

# 自定义模型配置
models = {
    'RF': ModelConfig(
        name='RF',
        path=Path('artifacts/models/seed_0_A_B/baseline/RF_seed0.joblib'),
        model_type='rf',
        auc=0.958,
        precision=0.876
    )
}

predictor = MultiModelPredictor(
    seed=0,
    models=models,
    strategy=EnsembleStrategy.SOFT_VOTING
)
```

### 5.3 特征提取API

```python
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import numpy as np

def smiles_to_features(smiles: str):
    """将SMILES转换为模型所需的特征"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = {}

    # Morgan指纹 (2048位)
    fp_morgan = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, 2, nBits=2048
    )
    features['morgan'] = np.array(fp_morgan, dtype=np.float32).reshape(1, -1)

    # MACCS键 (167位)
    fp_maccs = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    features['maccs'] = np.array(fp_maccs, dtype=np.float32).reshape(1, -1)

    # Atom Pairs (1024位)
    fp_ap = rdMolDescriptors.GetAtomPairFingerprint(mol)
    features['atom_pairs'] = np.array(fp_ap, dtype=np.float32).reshape(1, -1)

    # FP2 (2048位)
    fp_fp2 = Chem.RDKFingerprint(mol, fpSize=2048)
    features['fp2'] = np.array(fp_fp2, dtype=np.float32).reshape(1, -1)

    # RDKit描述符
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit.Chem import Descriptors

    descriptor_names = [x[0] for x in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
        descriptor_names
    )
    descriptors = calculator.CalcDescriptors(mol)
    features['rdkit_desc'] = np.array(descriptors, dtype=np.float32).reshape(1, -1)

    return features

# 使用示例
features = smiles_to_features("CCO")
print(f"Morgan shape: {features['morgan'].shape}")
print(f"Combined shape: {np.hstack([features['morgan'], features['maccs'],
                                     features['atom_pairs'], features['fp2']]).shape}")
```

---

## 6. Web应用使用

### 6.1 启动Web应用

```bash
# 进入项目目录
cd bbb_project

# 启动Streamlit应用
streamlit run app_bbb_predict.py

# 或指定端口
streamlit run app_bbb_predict.py --server.port 8501
```

应用将在 `http://localhost:8501` 启动。

### 6.2 主页（Home）

主页提供：
- **平台统计**: 模型数量、数据集数量、最佳性能指标
- **功能介绍**: 各数据集和模型的特点
- **快速开始指南**: 预测、模型选择、最佳实践

### 6.3 预测页面（Prediction）

#### 单个分子预测

1. **选择数据集**: A、A,B、A,B,C、A,B,C,D
2. **选择模型**: RF、XGB、LGBM、GAT等（含SMARTS增强版本）
3. **设置阈值**: 默认0.5，保守建议0.65
4. **输入SMILES**: 每行一个SMILES
5. **点击预测**: 查看结果和置信度

#### 批量预测

1. **上传CSV**: 必须包含`smiles`或`SMILES`列
2. **选择数据集和模型**
3. **点击批量预测**
4. **下载结果**: CSV格式，包含预测和概率

#### 全部模型预测

使用所有32个模型同时预测，查看：
- 各模型预测对比
- 一致性分析
- 概率热力图

### 6.4 SMARTS分析页面

分析化学子结构对BBB渗透性的影响：
- **全局SMARTS重要性**: 哪些子结构最重要
- **局部SMARTS贡献**: 特定分子的子结构贡献
- **可视化**: 结构式和重要性评分

### 6.5 模型对比页面（Model Comparison）

交互式模型对比：
- **条形图**: 按数据集或模型类型对比
- **散点图**: AUC vs F1、Precision vs Recall
- **热力图**: 完整性能矩阵
- **导出结果**: 下载对比结果CSV

### 6.6 降维可视化页面（Dimension Reduction）

使用PCA、t-SNE、LDA可视化分子分布：
- **6种特征类型**: Morgan、MACCS、AtomPairs、FP2、RDKitDesc、Combined
- **18个预设分子**: 带颜色标注
- **交互式探索**: 输入新SMILES查看位置

### 6.7 集成预测页面（Ensemble Prediction）

使用13个集成模型进行预测：
- **3个集成模型**: Stacking_XGB、Stacking_RF、SoftVoting
- **10个基础模型**: RF、XGB、LGBM等
- **批量预测**: CSV上传和下载
- **性能指标**: 实时显示选定模型性能

---

## 7. 高级功能

### 7.1 SMARTS子结构分析

#### 全局重要性分析

```bash
python scripts/10_global_smarts_importance.py \
    --seed 0 \
    --dataset A_B \
    --model RF+SMARTS
```

#### 交互分析

```bash
python scripts/11_global_smarts_interactions.py \
    --seed 0 \
    --dataset A_B
```

#### 可视化

```bash
python scripts/13_plot_smarts_interactions.py \
    --seed 0 \
    --dataset A_B
```

### 7.2 降维分析

#### t-SNE分析

```bash
python scripts/16_tsne_analysis.py \
    --seed 0 \
    --feature morgan \
    --perplexity 30
```

#### 高级降维

```bash
python scripts/17_advanced_dim_reduction.py \
    --seed 0 \
    --methods pca tsne lda umap \
    --features morgan maccs combined
```

### 7.3 模型解释

#### SHAP分析

```bash
python scripts/08_explain_atoms.py \
    --seed 0 \
    --model RF \
    --feature morgan \
    --n_samples 100
```

#### SMARTS解释

```bash
python scripts/09_explain_smarts.py \
    --seed 0 \
    --dataset A_B
```

### 7.4 主动学习

```python
from src.active_learning import ActiveLearningLoop

# 创建主动学习循环
al_loop = ActiveLearningLoop(
    seed=0,
    dataset='A_B',
    model_name='RF',
    n_iterations=10,
    n_query_per_iter=10
)

# 运行主动学习
al_loop.run()

# 查看学习曲线
al_loop.plot_learning_curve()
```

### 7.5 模型集成

#### 训练Stacking集成

```python
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

# 基学习器
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=0)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=0)),
    ('lgbm', LGBMClassifier(n_estimators=100, random_state=0))
]

# 元学习器
meta_estimator = LogisticRegression(max_iter=1000, random_state=0)

# Stacking分类器
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_estimator,
    cv=5,
    passthrough=True,
    n_jobs=-1
)

# 训练
stacking_clf.fit(X_train, y_train)
```

#### 使用多模型预测器

```python
from src.multi_model_predictor import MultiModelPredictor, EnsembleStrategy

# 创建预测器（使用多种集成策略）
predictors = {
    'hard_voting': MultiModelPredictor(seed=0, strategy=EnsembleStrategy.HARD_VOTING),
    'soft_voting': MultiModelPredictor(seed=0, strategy=EnsembleStrategy.SOFT_VOTING),
    'weighted': MultiModelPredictor(seed=0, strategy=EnsembleStrategy.WEIGHTED),
    'stacking': MultiModelPredictor(seed=0, strategy=EnsembleStrategy.STACKING)
}

# 比较不同策略
smiles = "CCO"
for name, predictor in predictors.items():
    result = predictor.predict_single(smiles)
    print(f"{name}: {result['ensemble_prediction']} (prob={result['ensemble_probability']:.3f})")
```

---

## 8. 常见问题

### 8.1 安装问题

#### Q1: RDKit安装失败

**A**: 使用conda安装RDKit更可靠：
```bash
conda install -c conda-forge rdkit
```

#### Q2: PyTorch Geometric安装问题

**A**: 确保PyTorch和PyG版本匹配：
```bash
# 先安装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 再安装PyG（指定版本）
pip install torch-geometric==2.3.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 8.2 训练问题

#### Q3: 训练时内存不足

**A**: 减小batch_size或使用更小的特征集：
```python
# 在config.py中调整
batch_size=32  # 默认64

# 或使用更少的特征
use_morgan=True
use_maccs=True
use_atom_pairs=False  # 禁用Atom Pairs
use_fp2=False  # 禁用FP2
```

#### Q4: GAT训练非常慢

**A**: 使用in-memory模式训练大数据集：
```bash
python scripts/06_finetune_bbb_from_smarts.py \
    --seed 0 \
    --dataset A_B_C \
    --init pretrained \
    --strategy freeze \
    --in_memory
```

### 8.3 预测问题

#### Q5: SMILES解析错误

**A**: 使用SMILES自动修复或检查格式：
```python
# 使用已知的修复映射
smiles_fixes = {
    'CC(C)(C)C1CCCc2ccccc2Cl': 'CC(C)C1CCCC1c2ccccc2Cl',
    'C=C(C)C=O)OCC[N+](C)(C)CCCS(=O)(=O)[O-]': 'C=C(C)C(=O)OCC[N+](C)(C)CCCS(=O)(=O)[O-]'
}

fixed_smiles = smiles_fixes.get(smiles, smiles)
mol = Chem.MolFromSmiles(fixed_smiles)
```

#### Q6: 特征维度不匹配

**A**: 确保使用正确的特征组合。Combined特征是5287维（Morgan 2048 + MACCS 167 + AtomPairs 1024 + FP2 2048）：
```python
# 错误：包含RDKitDesc（会导致维度错误）
combined_wrong = np.hstack([morgan, maccs, atom_pairs, fp2, rdkit_desc])

# 正确：只使用4种指纹
combined_correct = np.hstack([morgan, maccs, atom_pairs, fp2])
assert combined_correct.shape[1] == 5287
```

#### Q7: LGBM类型错误

**A**: 确保特征是float32类型：
```python
X = X.astype(np.float32)  # LGBM需要float类型
```

### 8.4 Web应用问题

#### Q8: Streamlit页面显示异常

**A**: 清除缓存并重启：
```bash
# 清除Streamlit缓存
streamlit cache clear

# 重新启动
streamlit run app_bbb_predict.py
```

#### Q9: GAT模型加载失败

**A**: 检查模型路径是否存在：
```python
# 对于大数据集，使用in_memory版本
if dataset in ['A_B_C', 'A_B_C_D']:
    gat_path = MODEL_DIR / "gat_finetune_bbb" / "pretrained_partial_inmemory" / "best.pt"
else:
    gat_path = MODEL_DIR / "gat_finetune_bbb" / "pretrained_partial" / "best.pt"
```

### 8.5 性能问题

#### Q10: 预测速度慢

**A**: 使用RF模型（最快）或禁用SMARTS分析：
```python
# 使用RF模型（最快）
model = 'Random Forest'  # 不使用SMARTS

# 或禁用SMARTS分析
show_smarts = False
show_atom_attributions = False
```

---

## 9. API参考

### 9.1 配置类

#### Paths

```python
from src.config import Paths

P = Paths()
print(P.root)           # 项目根目录
print(P.data_raw)       # 原始数据目录
print(P.models)         # 模型目录
```

#### DatasetConfig

```python
from src.config import DatasetConfig

D = DatasetConfig()
print(D.filename)       # "B3DB_classification.tsv"
print(D.smiles_col)     # "SMILES"
print(D.bbb_col)        # "BBB+/BBB-"
```

#### FeaturizeConfig

```python
from src.config import FeaturizeConfig

F = FeaturizeConfig()
print(F.morgan_bits)    # 2048
print(F.morgan_radius)  # 2
print(F.combine_features)  # True
```

### 9.2 核心函数

#### 数据划分

```python
from src.utils.split import stratified_train_val_test

split = stratified_train_val_test(
    df=df,
    label_col="y_cls",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=0
)

train_df = split.train
val_df = split.val
test_df = split.test
```

#### 特征提取

```python
from src.featurize.fingerprints import compute_all_fingerprints
from src.featurize.descriptors import compute_rdkit_descriptors

# 计算所有指纹
fingerprints = compute_all_fingerprints(smiles_list, seed=0)
# 返回: {'morgan': ..., 'maccs': ..., 'atom_pairs': ..., 'fp2': ...}

# 计算描述符
descriptors = compute_rdkits_descriptors(smiles_list, descriptor_set='all')
```

#### 模型训练

```python
from src.baseline.train_rf_xgb_lgb import train_eval_models

rows, roc_preds = train_eval_models(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    out_model_dir=output_dir,
    run_info={'seed': 0, 'feature': 'morgan'}
)
```

### 9.3 多模型预测器

#### EnsembleStrategy枚举

```python
from src.multi_model_predictor import EnsembleStrategy

# 可用策略
EnsembleStrategy.HARD_VOTING   # 硬投票：多数投票
EnsembleStrategy.SOFT_VOTING   # 软投票：概率平均
EnsembleStrategy.WEIGHTED      # 加权平均：基于性能
EnsembleStrategy.MAX_PROB      # 最大概率：最激进
EnsembleStrategy.MIN_PROB      # 最小概率：最保守
EnsembleStrategy.STACKING      # 堆叠：元学习器
```

#### MultiModelPredictor

```python
from src.multi_model_predictor import MultiModelPredictor, EnsembleStrategy

# 创建预测器
predictor = MultiModelPredictor(
    seed=0,
    strategy=EnsembleStrategy.SOFT_VOTING,
    threshold=0.5
)

# 预测
results = predictor.predict(smiles_list)

# 获取DataFrame
df = results.to_dataframe()

# 获取摘要
summary = results.get_summary()

# 获取模型信息
model_info = predictor.get_model_info()
```

### 9.4 图神经网络

#### GATBBB模型

```python
from src.finetune.train_gat_bbb_from_pretrain import GATBBB, FinetuneCfg

# 创建配置
cfg = FinetuneCfg(
    seed=0,
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

# 创建模型
model = GATBBB(in_dim=23, cfg=cfg)
```

#### 图数据集

```python
from src.featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig

# 创建配置
gcfg = GraphBuildConfig(
    smiles_col="SMILES",
    label_col="y_cls",
    id_col="row_id"
)

# 创建数据集
dataset = BBBGraphDataset(
    root=str(cache_dir),
    df=dataframe,
    cfg=gcfg
)
```

---

## 附录

### A. 模型性能速查表

| 模型 | 数据集 | AUC | Precision | Recall | F1 | MCC |
|------|--------|-----|-----------|--------|----|----|
| RF+SMARTS | A,B | 0.989 | 0.946 | 0.977 | 0.961 | 0.873 |
| XGB+SMARTS | A,B | 0.986 | 0.968 | 0.954 | 0.961 | 0.874 |
| LGBM+SMARTS | A,B | 0.984 | 0.968 | 0.946 | 0.957 | 0.864 |
| GAT+SMARTS | A,B | 0.948 | 0.899 | 0.964 | 0.931 | 0.773 |

### B. 特征类型速查表

| 特征 | 维度 | 描述 | 最佳模型 |
|------|------|------|----------|
| Morgan | 2048 | ECFP4-like，radius=2 | RF (AUC=0.972) |
| MACCS | 167 | 预定义子结构 | LGBM (MCC=0.781) |
| AtomPairs | 1024 | 拓扑距离 | LGBM (AUC=0.967) |
| FP2 | 2048 | Daylight类 | RF (AUC=0.961) |
| RDKitDesc | 39-200 | 分子描述符 | XGB (AUC=0.919) |
| Combined | 5287 | 4种指纹组合 | Stacking_XGB (AUC=0.973) |
| Graph | - | PyG分子图 | GAT+SMARTS (Recall=0.964) |

### C. 数据集速查表

| 数据集 | 样本数 | 正例率 | 推荐模型 | 最佳用途 |
|--------|--------|--------|----------|----------|
| A | 846 | 87.7% | GAT+SMARTS (AUC=0.948) | 高精度，低FP |
| A,B | 3,743 | 76.5% | RF+SMARTS (AUC=0.989) | 综合性能 ⭐ |
| A,B,C | 6,203 | 66.7% | XGB+SMARTS (AUC=0.972) | 大规模 |
| A,B,C,D | 6,244 | 63.5% | XGB+SMARTS (AUC=0.969) | 最大覆盖 |

### D. 推荐配置

#### 药物发现
- 数据集: A,B
- 模型: LGBM+SMARTS
- 阈值: 0.65
- 预期Precision: 0.968

#### 生产部署
- 数据集: A,B
- 模型: RF+SMARTS
- 阈值: 0.5
- 预期AUC: 0.989

#### 高通量筛选
- 数据集: A,B
- 模型: RF
- 阈值: 0.5
- 特点: 最快速度

#### 新化学型探索
- 数据集: A,B,C
- 模型: GAT+SMARTS
- 阈值: 0.5
- 特点: 化学结构感知

---

**文档结束**

如有问题，请参考项目README或提交issue。
