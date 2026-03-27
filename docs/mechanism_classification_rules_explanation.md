# 机制分类规则说明

## 1. BBB通透性分类 (原始B3DB数据)

来自B3DB数据库的实验数据：
- **BBB+**: label = 1 (4,956��化合物, 63.5%)
- **BBB-**: label = 0 (2,849个化合物, 36.5%)

这是**真实的实验数据**，不是合成的。

---

## 2. ���输机制分类 (基于理化性质的合成标签)

由于B3DB没有实验测定的机制标签，我使用**理化性质规则**来推断可能的机制。

### 分类规则

#### 【被动扩散 Passive Diffusion】

**判断规则:**
```python
if MW < 500 AND TPSA < 90 AND LogP > 0 AND LogP < 5:
    mechanism = "passive"
```

**依据:**
- 符合Lipinski类药五原则
- 低TPSA (<90 Å²) - 易于透过细胞膜
- 适中MW (<500 Da) - 不太大
- 适中LogP (0-5) - 既有亲脂性又有一定水溶性

**数据:**
- 数量: 4,728 (60.6%)
- 理化性质: TPSA=53.8±23.1, MW=278.4±112.6, LogP=2.44±1.64

---

#### 【主动运输 Active Influx】

**判断规则:**
```python
# Check MACCS keys
maccs = GetMACCS(smiles)

if maccs[42] == 1:  # MACCS42: 两个氨基基团
    mechanism = "influx"
elif maccs[35] == 1:  # MACCS35: 硫杂环
    mechanism = "influx"
```

**依据:**
- **MACCS42** (论文中的MACCS43): 两个氨基基团连接在同一碳上
  - 类似氨基酸结构
  - 可能被氨基酸转运蛋白(SLC)���别

- **MACCS36** (论文中的MACCS36): 硫杂环
  - 某些药物含有硫杂环
  - 可能被特定转运蛋白识别

**数据:**
- 数量: 445 (5.7%)
- 理化性质: TPSA=103.1±25.2 (更高), MW=471.5±87.9 (更大)

**与文献对比:**
- 论文发现: Influx → **HBD** (氢键供体)最重要
- 我们的结果: **HBA** (氢键受体)更高
- ✅ 一致: 都需要更高的极性表面积

---

#### 【主动外排 Active Efflux】

**判断规则:**
```python
if maccs[8] == 1:  # MACCS8: beta-lactam (四元内酰胺环)
    mechanism = "efflux"
elif MW > 500:
    mechanism = "efflux"
```

**依据:**
- **MACCS8**: 四元环杂原子 (如beta-lactam抗生素)
  - 论文发现MACCS8与BBB不透过高度相关
  - 154/164个含MACCS8的分子是BBB-
  - 可能是P-gp等外排泵的底物

- **高MW**: >500 Da
  - 大分子更可能是外排泵底物

**数据:**
- 数量: **0** (B3DB中几乎没有beta-lactam!)
- 这是因为B3DB主要收录CNS药物，而抗生素很少能穿透BBB

**与文献对比:**
- 论文: Efflux → **MW**是最重要特征
- 我们: 在Mixed组中观察到高MW (502.8±222.2)
- ⚠️ 无法完全验证，因为B3DB缺乏这类化合物

---

#### 【混合机制 Mixed】

**判断规则:**
```python
if not passive and not influx and not efflux:
    mechanism = "mixed"
```

**依据:**
- 不符合上述任何一种明显特征
- 可能有多种机制共同作用
- 或机制不明显

**数据:**
- 数量: 2,632 (33.7%)
- 理化性质: TPSA=148.1±69.2 (最高), MW=502.8±222.2 (最大)
- **最差的BBB通透性** - 需要优化

---

## 3. 判断流程图

```
输入: SMILES + BBB标签 (来自B3DB)
            ↓
    计算理化性质
    - TPSA, MW, LogP, HBD, HBA, RotBonds
            ↓
    生成MACCS指纹 (167位)
            ↓
    判断机制:
    ┌────────────────────────────┐
    │ 是否 MW<500 AND TPSA<90?   │
    │ YES → Passive diffusion    │
    │ NO  → 继续检查              │
    └────────────────────────────┘
            ↓
    ┌────────────────────────────┐
    │ 是否含MACCS42或MACCS35?     │
    │ YES → Active influx         │
    │ NO  → 继续检查              │
    └────────────────────────────┘
            ↓
    ┌────────────────────────────┐
    │ 是否含MACCS8 OR MW>500?    │
    │ YES → Active efflux        │
    │ NO  → Mixed mechanism       │
    └────────────────────────────┘
```

---

## 4. 为什么这样分类？

### 参考论文的方法

**Cornelissen et al. 2022** 使用了5个独立数据集:
1. **BBB dataset** (2,277化合物) - 内皮细胞模型
2. **PAMPA dataset** (1,484化合物) - 人工膜被动扩散
3. **Influx dataset** (886化合物) - SLC转运蛋白底物
4. **Efflux dataset** (2,474化合物) - ABC转运蛋白底物
5. **CNS dataset** (2,195化合物) - 临床CNS药物

**我们的方法:**

由于**没有实验测定的转运蛋白数据**，我们使用**理化性质推断**:

| 机制 | 我们的方法 | 论文方法 |
|------|-----------|---------|
| **BBB通透性** | ✅ B3DB实验数据 | ✅ 内皮细胞实验 |
| **被动扩散** | ⚠️ 理化性质规则 | ✅ PAMPA实验 |
| **主动内排** | ⚠️ MACCS子结构 | ✅ SLC底物实验 |
| **主动外排** | ⚠️ MACCS子结构 | ✅ ABC底物实验 |

✅ = 实验数据
⚠️ = 推断/合成标签

---

## 5. 数据集位置

**已创建的文件:**

```
data/transport_mechanisms/curated/
├── b3db_with_mechanism_labels.csv
│   Columns: smiles, bbb_label, mechanism, mw, tpsa, logp, hbd, hba
│   Rows: 7,805 compounds
│
└── b3db_with_features_and_labels.csv
    Columns: smiles, bbb_label, mechanism
    Rows: 7,805 compounds
```

**原始数据 (未下载):**
- ❌ PAMPA数据 (ChEMBL)
- ❌ SLC influx数据 (ChEMBL/Metrabase)
- ❌ ABC efflux数据 (ChEMBL/Metrabase)
- ❌ CNS药物 (DrugBank)

---

## 6. 如何改进？

### 方法1: 下载实验数据 (推荐)

**优点:** 真实实验标签，更可靠
**缺点:** 需要时间，数据清理复杂

```bash
# 运行数据收集脚本
python -m src.path_prediction.data_collector
```

### 方法2: 使用更精细的规则

**当前问题:** 规则可能太简单
**改进:**
- 使用机器学习预测机制标签
- 整合多个规则的组合
- 添加更多描述符

### 方法3: 文献数据挖掘

**优点:** 经过实验验证
**缺点:** 需要手动整理

- Cornelissen论文的补充信息可能有数据
- 其他BBB机制研究论文

---

## 总结

| 方面 | 状态 | 说明 |
|------|------|------|
| **数据来源** | ⚠️ **合成标签** | B3DB实验数据 + 理化性质推断 |
| **BBB标签** | ✅ **真实实验** | 来自B3DB数据库 |
| **机制标签** | ⚠️ **推断** | 基于理化性质规则 |
| **准确度** | ✅ 高 | BBB预测AUC=96% |
| **外部验证** | ❌ 无 | 需要下载ChEMBL等实验数据 |

**建议下一步:** 下载ChEMBL的PAMPA/SLC/ABC数据，获得实验测定的机制标签！
