# SMARTS特征预测修复说明

## 问题

SMARTS增强模型（RF+SMARTS, XGB+SMARTS, LGBM+SMARTS）在预测时报错：
```
X has 2048 features, but RandomForestClassifier is expecting 2050 features as input.
```

## 原因

SMARTS增强模型训练时使用了2050维特征：
- 2048维 Morgan指纹
- 2维 SMARTS二进制特征

但预测函数 `predict_bbb_batch` 只计算了2048维的Morgan指纹，缺少SMARTS特征。

## 解决方案

### 1. 添加SMARTS特征计算函数

在 `pages/0_prediction.py` 中添加了：

```python
def compute_smarts_features(smiles_list, smarts_patterns):
    """计算SMARTS特征（二进制向量）"""
    # 将SMARTS字符串转换为Mol对象
    smarts_mols = []
    for pattern in smarts_patterns:
        try:
            mol = Chem.MolFromSmarts(pattern)
            smarts_mols.append(mol if mol is not None else None)
        except:
            smarts_mols.append(None)

    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(np.zeros(len(smarts_patterns), dtype=np.int8))
            continue

        feat = []
        for smarts_mol in smarts_mols:
            if smarts_mol is None:
                feat.append(0)
            else:
                match = mol.HasSubstructMatch(smarts_mol)
                feat.append(1 if match else 0)

        features.append(np.array(feat, dtype=np.int8))

    return np.vstack(features)
```

### 2. 添加SMARTS patterns加载函数

```python
@st.cache_data
def load_smarts_patterns():
    """加载SMARTS patterns（缓存）"""
    import json
    smarts_file = PROJECT_ROOT / "assets" / "smarts" / "bbb_smarts_v1.json"
    if smarts_file.exists():
        with open(smarts_file, 'r') as f:
            data = json.load(f)
            return data.get('smarts', [])
    return []
```

### 3. 更新predict_bbb_batch函数

添加 `use_smarts` 参数：

```python
def predict_bbb_batch(smiles_list, model_path, threshold=0.5, model_type='rf', use_smarts=False):
    """批量预测函数

    Args:
        smiles_list: SMILES列表
        model_path: 模型路径
        threshold: 分类阈值
        model_type: 模型类型 ('rf' for joblib models, 'gnn' for PyTorch models)
        use_smarts: 是否使用SMARTS特征 (用于SMARTS增强模型)
    """
    # ... GNN模型预测代码 ...

    else:
        # 传统ML模型预测
        model = joblib.load(model_path)

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

        X_morgan = sparse.csr_matrix(np.vstack(fps))

        # 如果使用SMARTS特征，计算并拼接
        if use_smarts:
            smarts_patterns = load_smarts_patterns()
            if smarts_patterns:
                X_smarts = compute_smarts_features(smiles_list, smarts_patterns)
                # 拼接特征
                X_combined = sparse.hstack([X_morgan, X_smarts], format='csr')
                X = X_combined
            else:
                X = X_morgan
        else:
            X = X_morgan

        # 预测
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)

        return prob, pred
```

### 4. 自动检测是否需要SMARTS特征

在模型选择时自动判断：

```python
# 模型类型映射
model_types = {}
model_needs_smarts = {}
for name in models_available.keys():
    if 'GAT' in name:
        model_types[name] = 'gnn'
        model_needs_smarts[name] = False
    else:
        model_types[name] = 'rf'
        # 检查是否是SMARTS增强模型
        model_needs_smarts[name] = 'SMARTS' in name

selected_model_display = st.sidebar.selectbox("选择模型", list(models_available.keys()), index=0)
model_path = models_available[selected_model_display]
model_type = model_types[selected_model_display]
use_smarts = model_needs_smarts[selected_model_display]  # 自动判断
```

### 5. 更新所有预测调用

```python
# Tab 1: 单个分子预测
probs, preds = predict_bbb_batch(smiles_list, model_path, threshold, model_type, use_smarts)

# Tab 2: 批量预测
probs, preds = predict_bbb_batch(smiles_list, model_path, threshold, model_type, use_smarts)

# Tab 3: 全部模型预测
probs, preds = predict_bbb_batch(
    smiles_list,
    model_info['path'],
    threshold=0.5,
    model_type=model_info['type'],
    use_smarts=model_info.get('needs_smarts', False)  # 从模型信息获取
)
```

---

## 验证

### 特征维度
- 原始模型 (RF, XGB, LGBM): 2048维（仅Morgan指纹）
- SMARTS增强模型 (RF+SMARTS, XGB+SMARTS, LGBM+SMARTS): 2050维（Morgan + SMARTS）
- GAT模型: 图结构（不使用固定维度特征）

### 支持的32个模型

| 模型类型 | 特征维度 | use_smarts |
|---------|---------|------------|
| RF, XGB, LGBM | 2048 | False |
| RF+SMARTS, XGB+SMARTS, LGBM+SMARTS | 2050 | True |
| GAT+SMARTS | 图结构 | N/A |
| GAT | 图结构 | N/A |

---

## 测试

启动Streamlit：
```bash
streamlit run app_bbb_predict.py --server.port 8502
```

测试SMARTS增强模型：
1. 进入Prediction页面
2. 选择数据集（如 A,B）
3. 选择SMARTS增强模型（如 RF+SMARTS）
4. 输入SMILES: `CCO` (乙醇)
5. 点击预测
6. 应该成功预测，不再报错

---

## 性能影响

- SMARTS特征计算开销：约0.1-0.2秒每100个分子
- 使用 `@st.cache_data` 缓存SMARTS patterns，只加载一次
- 对于少量分子（<100个），性能影响可忽略不计

---

## 更新时间
2025-01-27

## 版本
v1.1 - SMARTS特征预测支持
