# 多模型集成预测模块

## 概述

本模块提供了血脑屏障（BBB）渗透性预测的多模型集成学习功能，支持多种集成策略，可以显著提高预测准确性和鲁棒性。

## 主要特点

1. **多种集成策略**
   - Hard Voting (硬投票): 多数投票
   - Soft Voting (��投票): 概率平均
   - Weighted Voting (加权投票): 基于模型性能加权
   - Max Prob (最大概率): 取最高概率模型
   - Min Prob (最小概率): 最保守预测

2. **灵活的模型组合**
   - 支持选择任意模型子集
   - 自动检测可用模型
   - 统一的预测接口

3. **丰富的分析功能**
   - 模型一致性分析
   - 预测摘要统计
   - 结果可视化和导出

## 快速开始

### 1. 基本使用

```python
from src.multi_model_predictor import create_ensemble_predictor

# 创建预测器（使用默认的硬投票策略）
predictor = create_ensemble_predictor(
    strategy='hard_voting',
    threshold=0.5
)

# 批量预测
smiles_list = ['CCO', 'CC(=O)OC1=CC=C(C=C)C=C1', 'c1ccccc1']
results = predictor.predict(smiles_list)

# 查看集成结果
print(results.ensemble_prediction)  # 最终预测 (0/1数组)
print(results.ensemble_probability)  # 集成概率
print(results.agreement)  # 模型一致性

# 转换为DataFrame
df = results.to_dataframe()

# 获取摘要统计
summary = results.get_summary()
```

### 2. 单分子预测

```python
# 预测单个分子
result = predictor.predict_single('CCO')

print(result['ensemble_prediction'])  # 'BBB+' 或 'BBB-'
print(result['ensemble_probability'])  # 概率值
print(result['individual_predictions'])  # 各模型详细结果
```

### 3. 使用不同集成策略

```python
# 硬投票（多数投票）
predictor_hard = create_ensemble_predictor(strategy='hard_voting')

# 软投票（概率平均）
predictor_soft = create_ensemble_predictor(strategy='soft_voting')

# 加权投票（基于模型性能）
predictor_weighted = create_ensemble_predictor(strategy='weighted')

# 最大概率
predictor_max = create_ensemble_predictor(strategy='max_prob')

# 最小概率（最保守）
predictor_min = create_ensemble_predictor(strategy='min_prob')
```

### 4. 使用自定义模型子集

```python
# 只使用RF和GAT+SMARTS
predictor = create_ensemble_predictor(
    strategy='hard_voting',
    models=['Random Forest', 'GAT+SMARTS']
)
```

## 集成策略说明

### Hard Voting (硬投票)
- **原理**: 每个模型进行投票，最终结果由多数决定
- **公式**: `Prediction = mode(Model1, Model2, ..., ModelN)`
- **适用场景**: 模型性能相近，追求简单直接
- **示例**: 如果4个模型中有3个预测为BBB+，则最终结果为BBB+

### Soft Voting (软投票)
- **原理**: 对所有模型的预测概率进行平均
- **公式**: `Probability = mean(P1, P2, ..., PN)`
- **适用场景**: 想要考虑模型的不确定性
- **示例**: 如果3个模型的概率分别是0.6, 0.7, 0.8，则最终概率为0.7

### Weighted Voting (加权投票)
- **原理**: 根据模型性能（AUC）进行加权平均
- **公式**: `Probability = sum(Wi * Pi) / sum(Wi)`
- **适用场景**: 模型性能差异较大
- **示例**: AUC高的模型（如RF: 0.958）对最终结果影响更大

### Max Prob (最大概率)
- **原理**: 选择预测概率最高的模型结果
- **公式**: `Probability = max(P1, P2, ..., PN)`
- **适用场景**: 有模型特别自信时
- **示例**: 选择所有模型中概率最高的那个

### Min Prob (最小概率)
- **原理**: 选择预测概率最低的模型结果
- **公式**: `Probability = min(P1, P2, ..., PN)`
- **适用场景**: 需要最保守预测的高风险场景
- **示例**: 选择所有模型中概率最低的那个，最谨慎

## Streamlit Web界面

启动应用：
```bash
streamlit run app_bbb_predict.py
```

然后访问 "Ensemble Prediction" 页面（页面3）。

### Web界面功能
1. 选择集成策略
2. 设置分类阈值
3. 选择要使用的模型
4. 单个或批量预测
5. 查看详细结果和可视化
6. 导出预测结果

## 示例脚本

查看完整示例：
```bash
python examples/multi_model_prediction_demo.py
```

示例包括：
1. 单个分子预测
2. 批量预测
3. 不同集成策略对比
4. 结果可视化
5. 自定义模型子集
6. 阈值敏感性分析

## API 参考

### MultiModelPredictor 类

主要方法：
- `predict(smiles_list: List[str]) -> PredictionResults`: 批量预测
- `predict_single(smiles: str) -> Dict`: 单分子预测
- `get_model_info() -> pd.DataFrame`: 获取可用模型信息

### PredictionResults 类

属性：
- `smiles`: 输入的SMILES列表
- `individual_predictions`: 各模型的预测结果
- `individual_probabilities`: 各模型的预测概率
- `ensemble_prediction`: 集成后的预测
- `ensemble_probability`: 集成后的概率
- `agreement`: 模型一致性 (0-1之间)
- `strategy`: 使用的集成策略

方法：
- `to_dataframe() -> pd.DataFrame`: 转换为DataFrame
- `get_summary() -> Dict`: 获取预测摘要

## 模型性能对比

| 模型 | AUC | Precision | FP | 特点 |
|------|-----|-----------|-----|------|
| Random Forest | 0.958 | 0.876 | 67 | 最佳综合性能 |
| GAT+SMARTS | 0.952 | 0.942 | 20 | 最低假阳性 |
| LightGBM | 0.955 | 0.896 | 53 | 平衡性能 |
| XGBoost | 0.949 | 0.866 | 72 | 稳健基线 |

## 建议使用场景

1. **通用推荐**: Hard Voting + 所有模型
   - 平衡了准确性和鲁棒性
   - 适用于大多数场景

2. **保守部署**: Weighted + RF + GAT+SMARTS
   - 使用性能最好的两个模型
   - 加权平均进一步优化

3. **高风险应用**: Min Prob + 所有模型
   - 最保守的预测策略
   - 适用于药物安全性评估

4. **快速预测**: Hard Voting + RF + LGBM
   - 只使用传统ML模型
   - 推理速度最快

## 常见问题

### Q: 如何选择集成策略？
A:
- 默认使用Hard Voting，简单有效
- 如果模型性能差异大，使用Weighted
- 如果需要保守预测，使用Min Prob
- 如果想考虑不确定性，使用Soft Voting

### Q: 模型一致性低怎么办？
A:
- 检查输入SMILES是否有效
- 查看各模型的预测概率分布
- 考虑使用更保守的集成策略
- 可能需要更多模型进行投票

### Q: 如何提高预测速度？
A:
- 只使用传统ML模型（RF, XGB, LGBM），不用GNN
- 减少模型数量
- 使用批量预测而非单次预测

## 文件结构

```
bbb_project/
├── src/
│   └── multi_model_predictor.py  # 核心模块
├── pages/
│   └── 3_ensemble_prediction.py  # Streamlit页面
├── examples/
│   └── multi_model_prediction_demo.py  # 示例脚本
├── test_ensemble.py  # 测试脚本
└── docs/
    └── multi_model_prediction.md  # 本文档
```

## 更新日志

### v1.0.0 (2024-01-26)
- 初始版本发布
- 支持5种集成策略
- 完整的Web界面和示例脚本
- 模型一致性分析功能
- 结果可视化和导出

## 许可证

本项目遵循主项目的许可证。

## 联系方式

如有问题或建议，请提交Issue或Pull Request。
