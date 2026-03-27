# 集成机制预测器使用指南

## 概述

集成机制预测器 (`IntegratedMechanismPredictor`) 结合了多种方法来预测分子的血脑屏障(BBB)穿透机制：

### 预测的机制

1. **BBB渗透性** - 分子是否能穿透血脑屏障
2. **被动扩散 (Passive Diffusion)** - 通过脂质双分子层的被动扩散
3. **主动内排 (Active Influx)** - 通过转运蛋白的主动摄取
4. **主动外排 (Active Efflux)** - 通过外排泵(如P-gp)的主动外排

## 方法

### 1. 机器学习模型 (来自Cornelissen 2022)
- **BBB模型**: AUC = 0.958
- **PAMPA模型**: AUC = 0.946 (作为被动扩散的代理)
- **Influx模型**: AUC = 0.927
- **Efflux模型**: AUC = 0.828

### 2. 基于理化性质的启发式规则
- **被动扩散**: 低TPSA (<90), 适中LogP (1-3), MW <500
- **主动内排**: 高TPSA (>100), 高HBA (>5)
- **主动外排**: 高MW (>500), 高TPSA (>80)

### 3. 集成策略
- 被动扩散: 60% ML + 40% 启发式
- 主动内排: 70% ML + 30% 启发式
- 主动外排: 70% ML + 30% 启发式

## 使用方法

### 基本使用

```python
from src.path_prediction.integrated_mechanism_predictor import IntegratedMechanismPredictor

# 初始化预测器
predictor = IntegratedMechanismPredictor()

# 预测单个分子
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
result = predictor.predict_mechanisms(smiles)

# 查看BBB预测
print(f"BBB+: {result['BBB']['prediction']}")
print(f"概率: {result['BBB']['probability']:.2%}")
print(f"置信度: {result['BBB']['confidence']}")

# 查看机制预测
print("\n机制预测:")
for mech in ['Passive_Diffusion', 'Active_Influx', 'Active_Efflux']:
    pred = result[mech]
    status = "是" if pred['prediction'] else "否"
    print(f"{mech}: {status} (概率: {pred['probability']:.2%})")

# 查看理化性质
print("\n理化性质:")
props = result['properties']
print(f"分子量: {props['MW']:.1f} Da")
print(f"TPSA: {props['TPSA']:.1f} A^2")
print(f"LogP: {props['LogP']:.2f}")
```

### 打印详细报告

```python
# 打印格式化的预测报告
predictor.print_prediction(smiles)
```

输出示例：
```
================================================================================
Mechanism Prediction for: CC(=O)OC1=CC=CC=C1C(=O)O
================================================================================

Physicochemical Properties:
  MW:   180.2 Da
  TPSA: 63.6 A^2
  LogP: 1.31
  HBA:  3
  HBD:  1

BBB Permeability:
  Prediction: BBB+ (Prob: 98.37%, Conf: High)
  Evidence:
    - Low TPSA (63.6 A^2) favors BBB penetration
    - Moderate MW (180.2 Da) suitable for BBB
    - Optimal LogP (1.31) for BBB penetration

Transport Mechanisms:
  Passive_Diffusion   : + (Prob: 91.05%, Conf: High)
    - Low TPSA (63.6 A^2) supports passive diffusion
    - Optimal LogP (1.31) for passive diffusion
  Active_Influx       : - (Prob: 0.14%, Conf: High)
  Active_Efflux       : - (Prob: 1.98%, Conf: High)

Overall Assessment:
  Primary Mechanism: Passive_Diffusion
  Certainty: High
================================================================================
```

### 批量预测

```python
# 预测多个分子
smiles_list = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # 阿司匹林
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # 咖啡因
    "NCCc1cc(O)c(O)cc1",  # 多巴胺
]

results = predictor.predict_batch(smiles_list)

for result in results:
    if 'error' not in result:
        print(f"{result['SMILES']}:")
        print(f"  BBB+: {result['BBB']['prediction']} ({result['BBB']['probability']:.2%})")
        print(f"  主要机制: {result['mechanism_summary']['primary_mechanism']}")
```

## 结果解读

### 概率范围

| 概率 | 置信度 | 解释 |
|------|--------|------|
| ≥80% | High | 非常确定 |
| 60-80% | Medium | 较为确定 |
| 40-60% | Low | 不确定 |
| ≤20% | High | 非常确定不会发生 |

### 机制判断

- **被动扩散为主**: 概率 >60%，且远高于其他机制
- **主动内排为主**: 概率 >60%，且远高于其他机制
- **主动外排为主**: 概率 >60%，且远高于其他机制
- **混合/不确定**: 所有机制概率都 <50%

### 典型案例

#### 阿司匹林 - 被动扩散
```
BBB+: 98.37% (High)
被动扩散: 91.05% (High)
主动内排: 0.14% (High)
主动外排: 1.98% (High)
主要机制: Passive_Diffusion (高确定性)
```

#### 咖啡因 - 混合机制
```
BBB+: 99.94% (High)
被动扩散: 23.26% (Medium)
主动内排: 7.86% (High)
主动外排: 2.18% (High)
主要机制: Mixed/Uncertain (低确定性)
```

#### 葡萄糖 - 主动内排
```
BBB-: 14.71% (High)
被动扩散: 13.04% (High)
主动内排: 22.16% (Medium) - 通过葡萄糖转运蛋白
主动外排: 12.42% (High)
主要机制: Mixed/Uncertain (低确定性)
```

## 支持证据

每个预测都附带支持性证据：

```python
# 查看BBB预测的证据
result['BBB']['supporting_evidence']
# ['Low TPSA (63.6 A^2) favors BBB penetration',
#  'Moderate MW (180.2 Da) suitable for BBB',
#  'Optimal LogP (1.31) for BBB penetration']

# 查看机制预测的证据
result['Passive_Diffusion']['supporting_evidence']
# ['Low TPSA (63.6 A^2) supports passive diffusion',
#  'Optimal LogP (1.31) for passive diffusion']
```

## 与单一模型对比

### 优势

1. **更高的可靠性**: 结合ML和启发式规则
2. **更好的解释性**: 提供支持证据
3. **更全面的评估**: 同时评估多种机制
4. **置信度评分**: 明确预测的不确定性

### 示例对比

| 分子 | 单一ML | 集成预测 | 优势 |
|------|--------|----------|------|
| 阿司匹林 | 被动扩散 92% | 被动扩散 91% | 一致，有证据支持 |
| 咖啡因 | 被动扩散 5% | 被动扩散 23% | 结合启发式，更准确 |
| 葡萄糖 | 内排 27% | 内排 22% | 考虑了高TPSA |

## 文件位置

- **预测器代码**: `src/path_prediction/integrated_mechanism_predictor.py`
- **基础ML预测器**: `src/path_prediction/mechanism_predictor_cornelissen.py`
- **模型文件**: `artifacts/models/cornelissen_2022/`

## 示例脚本

```bash
# 运行示例
python src/path_prediction/integrated_mechanism_predictor.py
```

## 注意事项

1. **预测基于训练数据**: 模型在Cornelissen 2022数据上训练，对新结构类型的分子可能不够准确
2. **置信度很重要**: 低置信度的预测需要谨慎对待
3. **仅作参考**: 这些预测不能替代实验验证
4. **机制复杂性**: 实际生物系统中，分子可能同时使用多种机制

## 故障排除

### 错误: 模型未找到
```bash
# 确保已训练模型
python scripts/mechanism_training/train_cornelissen_models.py
```

### 错误: 无效的SMILES
```python
# 检查SMILES有效性
from rdkit import Chem
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    print("Invalid SMILES")
```

### 预测速度慢
```python
# 批量预测使用predict_batch而非循环调用predict_mechanisms
results = predictor.predict_batch(smiles_list)  # 更快
```

## 更新日志

- **v1.0** (2026-03-13): 初始版本
  - 集成ML模型和启发式规则
  - 支持5种预测(BBB + 3种机制 + 总体评估)
  - 提供详细的支持证据
