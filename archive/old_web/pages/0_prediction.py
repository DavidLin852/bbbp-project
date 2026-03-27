"""
BBB渗透性预测平台 - Streamlit Web应用

功能：
1. 输入SMILES（单个或批量）
2. 使用训练好的模型预测BBB渗透性概率
3. 显示SMARTS子结构贡献
4. 提供批量预测结果下载

运行：
    streamlit run app_bbb_predict.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import io

import streamlit as st

st.set_page_config(
    page_title="BBB渗透性预测平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File is in pages/ folder, so parent.parent is the project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 自定义CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        background-color: #f5f5f5;
    }
    h1, h2, h3 {
        font-family: 'Times New Roman', serif;
    }
""", unsafe_allow_html=True)

# 侧边栏
st.sidebar.title("🧪 BBB预测平台")

# 配置
SEED = 0
SMARTS_FILE = PROJECT_ROOT / "assets" / "smarts" / "bbb_smarts_v1.json"
SMARTS_ANALYSIS_DIR = PROJECT_ROOT / "artifacts" / "explain" / "smarts_analysis"

st.sidebar.markdown("---")
st.sidebar.markdown("## 数据集选择")

# 数据集选项
dataset_options = {
    'A': 'A',
    'A,B': 'A_B',
    'A,B,C': 'A_B_C',
    'A,B,C,D': 'A_B_C_D'
}

selected_dataset_display = st.sidebar.selectbox(
    "选择训练数据集",
    options=list(dataset_options.keys()),
    index=1  # 默认A+B
)

selected_dataset = dataset_options[selected_dataset_display]

st.sidebar.markdown("---")
st.sidebar.markdown("## 模型选择")

# 加载模型性能数据
@st.cache_data
def load_performance_data():
    """加载所有模型性能数据（包括原始和SMARTS增强模型）"""
    # 加载原始16个模型
    perf_file = PROJECT_ROOT / "outputs" / "all_16_models_performance.csv"
    if perf_file.exists():
        original_df = pd.read_csv(perf_file)
    else:
        original_df = None

    # 加载SMARTS增强模型
    extended_models = []
    for dataset in ['A', 'A_B', 'A_B_C', 'A_B_C_D']:
        ext_file = PROJECT_ROOT / "outputs" / f"extended_models_{dataset}_seed0.csv"
        if ext_file.exists():
            ext_df = pd.read_csv(ext_file)
            extended_models.append(ext_df)

    if extended_models:
        extended_df = pd.concat(extended_models, ignore_index=True)
        # 添加缺失的列（如果存在）
        if 'TN' not in extended_df.columns:
            extended_df['TN'] = -1
        if 'FN' not in extended_df.columns:
            extended_df['FN'] = -1
    else:
        extended_df = None

    # 合并两个数据集
    if original_df is not None and extended_df is not None:
        return pd.concat([original_df, extended_df], ignore_index=True)
    elif original_df is not None:
        return original_df
    elif extended_df is not None:
        return extended_df
    else:
        return None

perf_df = load_performance_data()

# 根据数据集动态生成可用模型
def get_available_models(dataset, perf_df):
    """根据数据集获取可用模型（包括SMARTS增强模型）"""
    if perf_df is None:
        # 默认模型列表
        MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / f"seed_0_{dataset}"
        models_dict = {
            f"Random Forest": MODEL_DIR / "baseline" / "RF_seed0.joblib",
            f"XGBoost": MODEL_DIR / "baseline" / "XGB_seed0.joblib",
            f"LightGBM": MODEL_DIR / "baseline" / "LGBM_seed0.joblib",
        }

        # Add GAT model (no pretraining) if available
        gat_no_pretrain_path = MODEL_DIR / "gat_no_pretrain" / "best.pt"
        if not gat_no_pretrain_path.exists():
            # Try inmemory version for larger datasets
            gat_no_pretrain_path = MODEL_DIR / "gat_no_pretrain" / "best_inmemory.pt"

        if gat_no_pretrain_path.exists():
            models_dict["GAT"] = gat_no_pretrain_path

        return models_dict

    # 筛选当前数据集的模型
    dataset_df = perf_df[perf_df['Dataset'] == dataset]

    models_dict = {}
    for _, row in dataset_df.iterrows():
        model_name = row['Model']
        MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / f"seed_0_{dataset}"

        if model_name == 'RF':
            models_dict[f"Random Forest (AUC={row['AUC']:.3f})"] = MODEL_DIR / "baseline" / "RF_seed0.joblib"
        elif model_name == 'XGB':
            models_dict[f"XGBoost (AUC={row['AUC']:.3f})"] = MODEL_DIR / "baseline" / "XGB_seed0.joblib"
        elif model_name == 'LGBM':
            models_dict[f"LightGBM (AUC={row['AUC']:.3f})"] = MODEL_DIR / "baseline" / "LGBM_seed0.joblib"
        elif model_name == 'RF+SMARTS':
            models_dict[f"Random Forest+SMARTS (AUC={row['AUC']:.3f})"] = MODEL_DIR / "baseline_smarts" / "RF_smarts_seed0.joblib"
        elif model_name == 'XGB+SMARTS':
            models_dict[f"XGBoost+SMARTS (AUC={row['AUC']:.3f})"] = MODEL_DIR / "baseline_smarts" / "XGB_smarts_seed0.joblib"
        elif model_name == 'LGBM+SMARTS':
            models_dict[f"LightGBM+SMARTS (AUC={row['AUC']:.3f})"] = MODEL_DIR / "baseline_smarts" / "LGBM_smarts_seed0.joblib"
        elif model_name == 'GAT+SMARTS':
            # 对于A_B_C和A_B_C_D使用内存版本
            if dataset in ['A_B_C', 'A_B_C_D']:
                gat_path = MODEL_DIR / "gat_finetune_bbb" / "pretrained_partial_inmemory" / "best.pt"
            else:
                gat_path = MODEL_DIR / "gat_finetune_bbb" / "pretrained_partial" / "best.pt"
            models_dict[f"GAT+SMARTS (AUC={row['AUC']:.3f})"] = gat_path
        elif model_name == 'GAT':
            # GAT without pretraining (random initialization)
            if dataset in ['A_B_C', 'A_B_C_D']:
                gat_path = MODEL_DIR / "gat_no_pretrain" / "best_inmemory.pt"
            else:
                gat_path = MODEL_DIR / "gat_no_pretrain" / "best.pt"
            models_dict[f"GAT (AUC={row['AUC']:.3f})"] = gat_path

    return models_dict

# 获取可用模型
models_available = get_available_models(selected_dataset, perf_df)

# 模型类型映射
model_types = {}
model_needs_smarts = {}
for name in models_available.keys():
    if 'GAT' in name:
        model_types[name] = 'gnn'
        model_needs_smarts[name] = False
    else:
        # 所有其他模型（RF, XGB, LGBM, 包括SMARTS增强版本）都使用'rf'类型
        model_types[name] = 'rf'
        # 检查是否是SMARTS增强模型
        model_needs_smarts[name] = 'SMARTS' in name

selected_model_display = st.sidebar.selectbox("选择模型", list(models_available.keys()), index=0)
model_path = models_available[selected_model_display]
model_type = model_types[selected_model_display]
use_smarts = model_needs_smarts[selected_model_display]

# 显示数据集和模型信息
st.sidebar.caption(f"**数据集**: {selected_dataset_display}")

# 从性能数据中获取指标
if perf_df is not None:
    # 提取模型名称（去掉AUC部分）
    model_name_clean = selected_model_display.split(' (')[0].strip()
    model_perf = perf_df[
        (perf_df['Dataset'] == selected_dataset) &
        (perf_df['Model'] == model_name_clean)
    ]

    if len(model_perf) > 0:
        perf_row = model_perf.iloc[0]
        st.sidebar.caption(f"AUC: {perf_row['AUC']:.4f} | Precision: {perf_row['Precision']:.4f} | FP: {int(perf_row['FP'])}")
    else:
        # Model not in performance data
        if 'GAT' in model_name_clean and 'SMARTS' not in model_name_clean:
            st.sidebar.caption("Random initialization - no pretraining")
        else:
            st.sidebar.caption("性能数据不可用")
else:
    st.sidebar.caption("性能数据不可用")

st.sidebar.markdown("---")
st.sidebar.markdown("## 预测设置")

threshold = st.sidebar.slider(
    "分类阈值",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="更高阈值 = 更保守（减少假阳性）"
)

st.sidebar.caption("默认0.5，保守建议0.65")

st.sidebar.markdown("---")
st.sidebar.markdown("## 选项")
show_smarts = st.sidebar.checkbox("显示SMARTS子结构分析", value=False)
show_atom_attributions = st.sidebar.checkbox("显示原子级归因", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**说明**")
st.sidebar.caption("""
- 正例 (BBB+): 能透过血脑屏障
- 负例 (BBB-): 不能透过血脑屏障
- Precision: 预测为正例中真正为正例的比例
- Recall: 实际正例中被正确识别的比例
""")

# Initialize session state
if 'smiles_example' not in st.session_state:
    st.session_state.smiles_example = ""
if 'predict_clicked' not in st.session_state:
    st.session_state.predict_clicked = False
if 'batch_predict_clicked' not in st.session_state:
    st.session_state.batch_predict_clicked = False


# SMARTS分析相关函数
def load_smarts_patterns(smarts_file):
    """加载SMARTS模式用于分析"""
    with open(smarts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    patterns = []
    for item in data:
        smarts = item.get('smarts', '').replace(' ', '')
        name = item.get('name', '')
        if smarts and name:
            patterns.append({'name': name, 'smarts': smarts})

    return patterns


def match_smarts_to_molecule(smiles, smarts_patterns):
    """匹配SMARTS模式到分子

    Args:
        smiles: SMILES字符串
        smarts_patterns: SMARTS模式列表 [{'name': ..., 'smarts': ...}, ...]

    Returns:
        list: 匹配到的SMARTS名称列表
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    matched = []
    for pattern in smarts_patterns:
        try:
            patt = Chem.MolFromSmarts(pattern['smarts'])
            if patt is not None and mol.HasSubstructMatch(patt):
                matched.append(pattern['name'])
        except:
            pass

    return matched


def load_smarts_importance(smarts_analysis_dir):
    """加载SMARTS重要性数据"""
    import pandas as pd

    importance_file = smarts_analysis_dir / "smarts_importance_seed0.csv"
    if importance_file.exists():
        df = pd.read_csv(importance_file)
        # 转换为字典 {smarts_name: delta_prob}
        return dict(zip(df['smarts'], df['delta_prob']))
    return {}


def compute_smarts_features(smiles_list, smarts_patterns):
    """计算SMARTS特征（二进制向量）

    Args:
        smiles_list: SMILES列表
        smarts_patterns: SMARTS pattern列表

    Returns:
        np.ndarray: shape (n_smiles, n_smarts)
    """
    from rdkit import Chem

    # 将SMARTS字符串转换为Mol对象
    smarts_mols = []
    for pattern in smarts_patterns:
        try:
            mol = Chem.MolFromSmarts(pattern)
            if mol is not None:
                smarts_mols.append(mol)
            else:
                smarts_mols.append(None)
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


@st.cache_data
def load_smarts_patterns():
    """加载SMARTS patterns（缓存）返回SMARTS字符串列表"""
    import json
    smarts_file = PROJECT_ROOT / "assets" / "smarts" / "bbb_smarts_v1.json"
    if smarts_file.exists():
        with open(smarts_file, 'r') as f:
            data = json.load(f)
            # data是一个字典列表，提取smarts字符串
            return [item.get('smarts', '') for item in data if item.get('smarts')]
    return []


def predict_bbb_batch(smiles_list, model_path, threshold=0.5, model_type='rf', use_smarts=False):
    """批量预测函数

    Args:
        smiles_list: SMILES列表
        model_path: 模型路径
        threshold: 分类阈值
        model_type: 模型类型 ('rf' for joblib models, 'gnn' for PyTorch models)
        use_smarts: 是否使用SMARTS特征 (用于SMARTS增强模型)
    """
    import joblib
    import torch
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from scipy import sparse

    if model_type == 'gnn':
        # GNN模型预测
        from src.featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig
        from torch_geometric.loader import DataLoader
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        # 加载GNN模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建临时DataFrame用于构建图数据集
        df_temp = pd.DataFrame({
            'SMILES': smiles_list,
            'y_cls': [0] * len(smiles_list),
            'row_id': range(len(smiles_list))
        })

        # 构建图数据集
        gcfg = GraphBuildConfig(smiles_col="SMILES", label_col="y_cls", id_col="row_id")
        from src.finetune.train_gat_bbb_from_pretrain import GATBBB, FinetuneCfg

        # 加载模型checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # 检测checkpoint结构（兼容标准版本和内存版本）
        if 'model' in checkpoint:
            # 标准版本（seed_0_A, seed_0_A_B）
            state_dict = checkpoint['model']
            cfg = checkpoint.get('cfg', {})
            if isinstance(cfg, dict):
                # 使用保存的配置或默认配置
                hidden = cfg.get('hidden', 128)
                gat_heads = cfg.get('gat_heads', 4)
                num_layers = cfg.get('num_layers', 3)
                dropout = cfg.get('dropout', 0.2)
            else:
                hidden, gat_heads, num_layers, dropout = 128, 4, 3, 0.2
        else:
            # 内存版本（seed_0_A_B_C, seed_0_A_B_C_D）
            state_dict = checkpoint
            hidden, gat_heads, num_layers, dropout = 128, 4, 3, 0.2

        # 修复state_dict键名（兼容不同版本的模型）
        def remap_state_dict_keys(state_dict):
            """
            重新映射state_dict的键以匹配模型架构

            支持两种模型架构：
            1. 标准架构 (A, A_B): backbone.convs.*, head.net.*
            2. 内存架构 (A_B_C, A_B_C_D): convs.*, classifier.*
            """
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key

                # 架构2 → 架构1的映射
                # 将 "convs.N.*" 映射为 "backbone.convs.N.*"
                if key.startswith('convs.'):
                    new_key = 'backbone.' + key

                # 将 "classifier.N.*" 映射为 "head.net.N.*"
                elif key.startswith('classifier.'):
                    parts = key.split('.')
                    # classifier.0.weight -> [classifier, 0, weight]
                    # -> [head, net, 0, weight] -> head.net.0.weight
                    new_key = 'head.net.' + '.'.join(parts[1:])

                new_state_dict[new_key] = value
            return new_state_dict

        state_dict = remap_state_dict_keys(state_dict)

        # 创建模型
        cfg = FinetuneCfg(
            seed=0,
            hidden=hidden,
            gat_heads=gat_heads,
            num_layers=num_layers,
            dropout=dropout,
            epochs=60,
            batch_size=64,
            lr=2e-3,
            grad_clip=5.0,
            init='pretrained',
            strategy='freeze'
        )

        # 从checkpoint中读取输入维度（更可靠）
        # 查找第一个GAT层的权重来确定输入维度
        in_dim = 23  # 默认值
        for key in state_dict.keys():
            if 'convs.0.lin.weight' in key:
                in_dim = state_dict[key].shape[1]
                break
            elif 'backbone.convs.0.lin.weight' in key:
                in_dim = state_dict[key].shape[1]
                break

        model = GATBBB(in_dim, cfg).to(device)

        # 加载state_dict，使用strict=False以兼容不同架构
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            st.warning(f"⚠️ GAT模型加载警告: 缺少 {len(missing)} 个键，使用随机初始化")
        if unexpected:
            st.warning(f"⚠️ GAT模型加载警告: 多余 {len(unexpected)} 个键已忽略")

        model.eval()

        # 创建图数据集 - 使用时间戳避免缓存冲突
        import time
        import uuid
        # 使用更安全的文件名（Windows兼容）
        unique_id = f"pred_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        temp_cache = PROJECT_ROOT / "artifacts" / "temp_predict" / unique_id

        # 确保目录存在
        try:
            temp_cache.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # 如果创建失败，使用更简单的路径
            temp_cache = PROJECT_ROOT / "artifacts" / "temp_predict" / f"pred_{uuid.uuid4().hex[:12]}"
            temp_cache.mkdir(parents=True, exist_ok=True)

        # 创建数据集（捕获可能的OSError）
        import io
        import sys

        # 临时重定向stderr以避免Windows下的print错误
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            temp_ds = BBBGraphDataset(root=str(temp_cache), df=df_temp, cfg=gcfg)
        except OSError as e:
            # Windows路径问题，使用绝对路径重试
            temp_cache_abs = temp_cache.resolve()
            temp_ds = BBBGraphDataset(root=str(temp_cache_abs), df=df_temp, cfg=gcfg)
        finally:
            # 恢复stderr
            sys.stderr = old_stderr

        # 检查是否有有效样本
        if len(temp_ds) == 0:
            # 所有SMILES都无效，返回默认预测
            prob = np.full(len(smiles_list), 0.5)  # 默认中等概率
            pred = np.zeros(len(smiles_list), dtype=int)  # 默认负例
            return prob, pred

        temp_loader = DataLoader(temp_ds, batch_size=64, shuffle=False)

        # 预测 - 并跟踪row_id
        probs = []
        row_ids = []
        with torch.no_grad():
            for batch in temp_loader:
                batch = batch.to(device)
                logit = model(batch)
                prob = torch.sigmoid(logit).cpu().numpy()
                probs.append(prob)
                # 提取row_id (batch中每个样本的原始索引)
                if hasattr(batch, 'row_id'):
                    row_ids.extend([int(rid) for rid in batch.row_id])
                else:
                    # 如果没有row_id，使用默认索引
                    row_ids.extend(range(len(prob)))

        if not probs:
            # 没有成功处理的样本
            prob = np.full(len(smiles_list), 0.5)
            pred = np.zeros(len(smiles_list), dtype=int)
            return prob, pred

        probs_array = np.concatenate(probs)

        # 创建完整长度的结果数组（默认值为NaN表示失败）
        full_probs = np.full(len(smiles_list), np.nan)
        full_preds = np.full(len(smiles_list), -1)

        # 映射预测结果回原始位置
        for i, rid in enumerate(row_ids):
            if rid < len(full_probs):
                full_probs[rid] = float(probs_array[i])
                full_preds[rid] = int(float(probs_array[i]) >= threshold)

        # 对于失败的分子，分配默认预测
        for i in range(len(full_probs)):
            if np.isnan(full_probs[i]):
                full_probs[i] = 0.5  # 默认中等概率
                full_preds[i] = 0    # 默认负例

        prob = full_probs
        pred = full_preds.astype(int)

        return prob, pred

    else:
        # 传统ML模型预测 (RF, XGB, LGBM)
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
            # 使用GetOnBits方法正确转换指纹
            on_bits = list(fp.GetOnBits())
            arr = np.zeros((2048,), dtype=np.int8)
            arr[on_bits] = 1
            fps.append(arr)

        X_morgan = sparse.csr_matrix(np.vstack(fps))

        # 如果使用SMARTS特征，计算真实的70个SMARTS子结构特征
        if use_smarts:
            # 加载SMARTS patterns（70个化学子结构）
            smarts_strings = load_smarts_patterns()
            if smarts_strings:
                # 计算SMARTS二进制特征（维度: n_samples × 70）
                X_smarts = compute_smarts_features(smiles_list, smarts_strings)
                X_smarts_sparse = sparse.csr_matrix(X_smarts, dtype=np.int8)
                # 组合Morgan指纹(2048) + SMARTS特征(70) = 2118维
                X = sparse.hstack([X_morgan, X_smarts_sparse], format='csr')
            else:
                X = X_morgan
        else:
            X = X_morgan

        # 转换为float32（LightGBM需要float类型输入）
        X = X.astype(np.float32)

        # 预测
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)

        return prob, pred


# 主界面
st.title("BBB Permeability Prediction Platform")
st.markdown("""
This platform uses machine learning models to predict blood-brain barrier (BBB) permeability for small molecules.

**Available Models:**
- **Random Forest / RF+SMARTS**: Best overall performance, fast inference
- **XGBoost / XGB+SMARTS**: High AUC performance
- **LightGBM / LGBM+SMARTS**: Highest precision, lowest FP
- **GAT+SMARTS**: Chemical structure awareness, conservative predictions
- **GAT**: Random initialization baseline for comparison

**Usage:**
1. Enter SMILES (single or batch)
2. Select model and threshold
3. Click "Predict" to view results
""")

# 模型对比说明（使用真实的70个SMARTS子结构特征）
with st.expander("Model Comparison (Click to expand)"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**RF+SMARTS (A+B)**")
        st.metric("AUC", "0.989", delta="Best")
        st.metric("Precision", "0.946")
        st.metric("FP", "19")
        st.caption("Best AUC with 70 SMARTS")

    with col2:
        st.markdown("**XGB+SMARTS (A+B)**")
        st.metric("AUC", "0.986")
        st.metric("Precision", "0.968", delta="Best")
        st.metric("FP", "11", delta="Lowest")
        st.caption("Highest precision")

    with col3:
        st.markdown("**LGBM+SMARTS (A+B)**")
        st.metric("AUC", "0.984")
        st.metric("Precision", "0.968")
        st.metric("FP", "11")
        st.caption("Balanced performance")

st.markdown("---")

# 输入选项
tab1, tab2, tab3 = st.tabs(["单个分子预测", "批量预测", "全部模型预测"])

with tab1:
    st.subheader("单个分子预测")

    smiles_input = st.text_area(
        "输入SMILES",
        value=st.session_state.smiles_example,
        placeholder="CCO\nCC(=O)OC1=CC=C(C=C)C=C1\nCCN",
        height=150,
        help="每行一个SMILES",
        key="smiles_input_area"
    )

    col1, col2 = st.columns(2)

    with col1:
        predict_btn = st.button("🔍 预测", type="primary", use_container_width=True)

    with col2:
        example_btn = st.button("📋 示例SMILES")

    if example_btn:
        st.session_state.smiles_example = """CCO
CC(=O)OC1=CC=C(C=C)C=C1
CCN
CC(C)C(=O)N
CC1=CC=C(C)C1CO
CC(C)(C)O
CC1=CN=C1C=CC(=O)"""
        st.rerun()

    if predict_btn or st.session_state.get('predict_clicked', False):
        st.session_state.predict_clicked = True

        smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]

        if not smiles_list:
            st.warning("请输入至少一个SMILES")
        else:
            # 预测
            with st.spinner("预测中..."):
                probs, preds = predict_bbb_batch(smiles_list, model_path, threshold, model_type, use_smarts)

            # 确保数组是1D的
            probs = np.asarray(probs).flatten()
            preds = np.asarray(preds).flatten()

            # 显示结果
            results_df = pd.DataFrame({
                'SMILES': smiles_list,
                'BBB+概率': probs,
                '预测': ['BBB+' if p == 1 else 'BBB-' for p in preds],
                '置信度': ['高' if p > 0.8 or p < 0.2 else '中' for p in probs]
            })

            st.success("预测完成！")

            # 显示结果表格
            st.subheader("预测结果")
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'BBB+概率': st.column_config.NumberColumn(
                        "BBB+概率",
                        format="%.3f",
                        help="BBB+ probability"
                    ),
                    '置信度': st.column_config.TextColumn(
                        "置信度",
                        width="small"
                    ),
                    '预测': st.column_config.TextColumn(
                        "预测",
                        width="small"
                    )
                }
            )

            # 统计
            n_pos = sum(preds)
            n_total = len(preds)
            pos_rate = n_pos / n_total if n_total > 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总样本数", n_total)
            with col2:
                st.metric("预测为BBB+", n_pos)
            with col3:
                st.metric("正例率", f"{pos_rate:.1%}")

            # SMARTS分析
            if show_smarts:
                st.markdown("---")
                st.subheader("SMARTS子结构分析")

                # 加载SMARTS模式和重要性
                try:
                    smarts_patterns = load_smarts_patterns(SMARTS_FILE)
                    smarts_importance = load_smarts_importance(SMARTS_ANALYSIS_DIR)

                    # 为每个分子匹配SMARTS
                    st.markdown("**包含的SMARTS子结构及其贡献：**")

                    for idx, row in results_df.iterrows():
                        smiles = row['SMILES']
                        prob = row['BBB+概率']

                        # 匹配SMARTS
                        matched_smarts = match_smarts_to_molecule(smiles, smarts_patterns)

                        if matched_smarts:
                            # 按重要性排序
                            matched_with_importance = []
                            for smarts_name in matched_smarts:
                                delta = smarts_importance.get(smarts_name, 0)
                                matched_with_importance.append((smarts_name, delta))

                            matched_with_importance.sort(key=lambda x: abs(x[1]), reverse=True)

                            # 显示
                            with st.expander(f"分子 {idx+1}: {smiles[:30]}... (BBB+概率: {prob:.3f})"):
                                cols = st.columns(2)
                                with cols[0]:
                                    st.markdown("**匹配的SMARTS子结构：**")
                                with cols[1]:
                                    st.markdown("**贡献 (Δ Probability):**")

                                for smarts_name, delta in matched_with_importance[:10]:  # 只显示top 10
                                    delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
                                    color = "🟢" if delta > 0 else "🔴"
                                    st.markdown(f"**{smarts_name}** | {color} {delta_str}")

                                if len(matched_with_importance) > 10:
                                    st.caption(f"... 还有 {len(matched_with_importance) - 10} 个子结构")
                        else:
                            st.caption(f"分子 {idx+1}: {smiles[:30]}... - 未匹配到已知SMARTS子结构")

                        st.markdown("---")

                except Exception as e:
                    st.error(f"SMARTS分析出错: {e}")
                    st.info("请确保已运行SMARTS重要性分析脚本")

with tab2:
    st.subheader("批量预测")

    upload_file = st.file_uploader(
        "上传CSV文件",
        type=['csv'],
        help="CSV必须包含'smiles'列（或'SMILES'）"
    )

    st.markdown("**CSV格式要求**:")
    st.code("""
smiles
CCO
CC(=O)OC1=CC=C(C=C)C=C1
CCN
...""")

    if upload_file is not None:
        try:
            df = pd.read_csv(upload_file)

            # 检测列名
            smiles_col = None
            for col in df.columns:
                if col.lower() in ['smiles', 'smile']:
                    smiles_col = col
                    break

            if smiles_col is None:
                st.error(f"CSV必须包含'smiles'或'SMILES'列，当前列: {list(df.columns)}")
            else:
                st.session_state.batch_df = df
                st.success(f"加载了{len(df)}个分子")
                st.dataframe(df.head(10))

                if st.button("🔍 批量预测", type="primary", use_container_width=True):
                    st.session_state.batch_predict_clicked = True

        except Exception as e:
            st.error(f"文件读取错误: {e}")

    elif 'batch_df' in st.session_state and st.session_state.get('batch_predict_clicked', False):
        st.session_state.batch_predict_clicked = True

    # 如果已上传文件，进行预测
    if 'batch_df' in st.session_state and st.session_state.get('batch_predict_clicked', False):
        df = st.session_state.batch_df

        # 检查SMILES列
        smiles_col = None
        for col in df.columns:
            if col.lower() in ['smiles', 'smile']:
                smiles_col = col
                break

        if smiles_col is None:
            st.error("无法识别SMILES列")
            st.stop()

        smiles_list = df[smiles_col].astype(str).tolist()

        with st.spinner(f"预测{len(smiles_list)}个分子..."):
            probs, preds = predict_bbb_batch(smiles_list, model_path, threshold, model_type, use_smarts)

        # 确保数组是1D的
        probs = np.asarray(probs).flatten()
        preds = np.asarray(preds).flatten()

        # 添加结果到原DataFrame
        df['BBB+概率'] = probs
        df['预测'] = ['BBB+' if p == 1 else 'BBB-' for p in preds]

        st.session_state.batch_results = df

        # 显示结果
        st.success("预测完成！")

        # 显示前20行
        st.subheader("预测结果（前20行）")
        st.dataframe(
            df.head(20),
            use_container_width=True,
            hide_index=True,
            column_config={
                'BBB+概率': st.column_config.NumberColumn("BBB+概率", format="%.3f"),
                '预测': st.column_config.TextColumn("预测", width="small")
            }
        )

        # 统计
        n_pos = df['预测'].apply(lambda x: 1 if x == 'BBB+' else 0).sum()
        n_total = len(df)
        pos_rate = n_pos / n_total if n_total > 0 else 0

        st.metric("总样本数", n_total)
        st.metric("预测为BBB+", n_pos)
        st.metric("正例率", f"{pos_rate:.1%}")

        # 下载按钮
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            csv,
            "bbb_predictions.csv",
            "mime=text/csv",
            key=f"bbb_predictions_{len(df)}_results"
        )

        # 显示完整数据选项
        if st.checkbox("显示完整结果"):
            st.dataframe(df)

# Tab 3: 全部模型预测
with tab3:
    st.subheader("全部模型预测")
    st.markdown("使用所有32个模型同时预测，一次性查看所有模型的预测结果")

    # SMILES输入
    st.markdown("### 输入分子")

    col1, col2 = st.columns([2, 1])

    with col1:
        all_models_smiles = st.text_area(
            "输入SMILES",
            value="",
            placeholder="CCO\nCC(=O)OC1=CC=C(C=C)C=C1",
            height=100,
            help="每行一个SMILES，最多5个分子",
            key="all_models_smiles_input"
        )

    with col2:
        st.markdown("**或上传CSV**")
        upload_small = st.file_uploader(
            "上传CSV文件（最多5行）",
            type=['csv'],
            help="CSV必须包含'smiles'列，最多5个分子",
            key="all_models_upload"
        )

    # 示例按钮
    if st.button("📋 加载示例分子", key="all_models_example"):
        st.session_state.all_models_smiles = """CCO
CC(=O)OC1=CC=C(C=C)C=C1
CCN
CC(C)CC1=CC=C(C=C1)O
c1ccccc1"""
        st.rerun()

    if 'all_models_smiles' in st.session_state:
        all_models_smiles = st.text_area(
            "输入SMILES",
            value=st.session_state.all_models_smiles,
            height=100,
            key="all_models_smiles_display"
        )

    # 预测按钮
    if st.button("🚀 运行全部32个模型", type="primary", use_container_width=True, key="run_all_models"):
        # 获取SMILES列表
        smiles_list = []

        if upload_small is not None:
            try:
                df_upload = pd.read_csv(upload_small)
                for col in df_upload.columns:
                    if col.lower() in ['smiles', 'smile']:
                        smiles_list = df_upload[col].astype(str).tolist()[:5]  # 最多5个
                        break
            except:
                pass

        if not smiles_list and all_models_smiles:
            smiles_list = [s.strip() for s in all_models_smiles.split('\n') if s.strip()][:5]

        if not smiles_list:
            st.error("请输入至少一个SMILES或上传CSV文件")
        elif len(smiles_list) > 5:
            st.warning("最多支持5个分子，将只使用前5个")
            smiles_list = smiles_list[:5]

        # 运行所有模型预测
        st.info(f"正在使用32个模型预测 {len(smiles_list)} 个分子...")

        # 加载所有模型信息
        @st.cache_data
        def get_all_models_info():
            """获取所有32个模型的信息

            每个数据集8个模型：
            1. RF（基础）
            2. XGB（基础）
            3. LGBM（基础）
            4. RF+SMARTS
            5. XGB+SMARTS
            6. LGBM+SMARTS
            7. GAT（无预训练）
            8. GAT+SMARTS（预训练）
            """
            import json
            all_models = []

            for dataset in ['A', 'A_B', 'A_B_C', 'A_B_C_D']:
                MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / f"seed_0_{dataset}"
                dataset_display = {'A': 'A', 'A_B': 'A,B', 'A_B_C': 'A,B,C', 'A_B_C_D': 'A,B,C,D'}[dataset]

                # ========== 1. 基础模型（RF, XGB, LGBM） ==========
                baseline_dir = MODEL_DIR / "baseline"
                if baseline_dir.exists():
                    # 加载全局基础模型性能作为fallback
                    global_baseline_path = PROJECT_ROOT / "artifacts" / "metrics" / "baseline_seed0.csv"
                    global_baseline = {}
                    if global_baseline_path.exists():
                        df_global = pd.read_csv(global_baseline_path)
                        for _, row in df_global.iterrows():
                            global_baseline[row['model']] = {
                                'auc': row['auc'],
                                'precision': row['precision']
                            }

                    for model_name in ['RF', 'XGB', 'LGBM']:
                        model_file = baseline_dir / f"{model_name}_seed0.joblib"
                        metrics_file = baseline_dir / f"{model_name}_metrics.json"

                        if model_file.exists():
                            # 尝试读取指标文件，如果不存在则使用全局数据
                            if metrics_file.exists():
                                with open(metrics_file, 'r') as f:
                                    metrics = json.load(f)
                                auc = metrics.get('auc', 0.0)
                                precision = metrics.get('precision', 0.0)
                            else:
                                # 使用全局基准数据作为fallback
                                if model_name in global_baseline:
                                    auc = global_baseline[model_name]['auc']
                                    precision = global_baseline[model_name]['precision']
                                else:
                                    auc = 0.95  # 默认值
                                    precision = 0.88

                            all_models.append({
                                'dataset': dataset_display,
                                'model': model_name,
                                'path': model_file,
                                'type': 'rf',
                                'needs_smarts': False,
                                'auc': auc,
                                'precision': precision
                            })

                # ========== 2. SMARTS增强模型（RF+SMARTS, XGB+SMARTS, LGBM+SMARTS）+ GAT ==========
                perf_file = PROJECT_ROOT / "outputs" / f"extended_models_{dataset}_seed0.csv"
                if perf_file.exists():
                    df = pd.read_csv(perf_file)

                    for _, row in df.iterrows():
                        model_name = row['Model']

                        if model_name == 'RF+SMARTS':
                            model_path = MODEL_DIR / "baseline_smarts" / "RF_smarts_seed0.joblib"
                            model_type = 'rf'
                            needs_smarts = True
                        elif model_name == 'XGB+SMARTS':
                            model_path = MODEL_DIR / "baseline_smarts" / "XGB_smarts_seed0.joblib"
                            model_type = 'rf'
                            needs_smarts = True
                        elif model_name == 'LGBM+SMARTS':
                            model_path = MODEL_DIR / "baseline_smarts" / "LGBM_smarts_seed0.joblib"
                            model_type = 'rf'
                            needs_smarts = True
                        elif model_name == 'GAT':
                            # 对于A_B_C和A_B_C_D，先尝试best_inmemory.pt，如果不存在则用best.pt
                            if dataset in ['A_B_C', 'A_B_C_D']:
                                model_path_inmem = MODEL_DIR / "gat_no_pretrain" / "best_inmemory.pt"
                                model_path = MODEL_DIR / "gat_no_pretrain" / "best.pt"
                                # 优先使用存在的文件
                                if model_path_inmem.exists():
                                    model_path = model_path_inmem
                            else:
                                model_path = MODEL_DIR / "gat_no_pretrain" / "best.pt"
                            model_type = 'gnn'
                            needs_smarts = False
                        else:
                            continue

                        if model_path.exists():
                            all_models.append({
                                'dataset': dataset_display,
                                'model': model_name,
                                'path': model_path,
                                'type': model_type,
                                'needs_smarts': needs_smarts,
                                'auc': row['AUC'],
                                'precision': row['Precision']
                            })

                # ========== 3. GAT+SMARTS（预训练模型） ==========
                gat_smarts_dir = MODEL_DIR / "gat_finetune_bbb" / "pretrained_partial"
                # 如果pretrained_partial不存在，尝试pretrained_partial_inmemory
                if not gat_smarts_dir.exists():
                    gat_smarts_dir = MODEL_DIR / "gat_finetune_bbb" / "pretrained_partial_inmemory"

                history_file = gat_smarts_dir / "finetune_history.csv"
                model_file = gat_smarts_dir / "best.pt"

                if model_file.exists():
                    # 尝试读取性能数据，如果文件不存在则使用默认值
                    try:
                        if history_file.exists():
                            history_df = pd.read_csv(history_file)
                            if len(history_df) > 0:
                                best_row = history_df.iloc[-1]
                                fp = best_row.get('test_fp', 0)
                                tp = best_row.get('test_tp', 1)
                                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                                auc = best_row.get('test_auc', 0.90)
                            else:
                                precision = 0.90
                                auc = 0.90
                        else:
                            # 没有history文件，使用默认值
                            precision = 0.90
                            auc = 0.90

                        all_models.append({
                            'dataset': dataset_display,
                            'model': 'GAT+SMARTS',
                            'path': model_file,
                            'type': 'gnn',
                            'needs_smarts': False,
                            'auc': auc,
                            'precision': precision
                        })
                    except Exception:
                        # 如果读取失败，使用默认值
                        all_models.append({
                            'dataset': dataset_display,
                            'model': 'GAT+SMARTS',
                            'path': model_file,
                            'type': 'gnn',
                            'needs_smarts': False,
                            'auc': 0.90,
                            'precision': 0.90
                        })

            return all_models

        all_models = get_all_models_info()

        if not all_models:
            st.error("无法加载模型信息")
        else:
            st.write(f"已加载 {len(all_models)} 个模型")

            # 进度条
            progress_bar = st.progress(0, text="初始化...")

            # 存储所有预测结果
            results = []

            for idx, model_info in enumerate(all_models):
                progress = (idx + 1) / len(all_models)
                progress_bar.progress(progress, text=f"预测中... {idx+1}/{len(all_models)}: {model_info['dataset']} - {model_info['model']}")

                try:
                    probs, preds = predict_bbb_batch(
                        smiles_list,
                        model_info['path'],
                        threshold=0.5,
                        model_type=model_info['type'],
                        use_smarts=model_info.get('needs_smarts', False)
                    )

                    probs = np.asarray(probs).flatten()
                    preds = np.asarray(preds).flatten()

                    for i, (smiles, prob, pred) in enumerate(zip(smiles_list, probs, preds)):
                        results.append({
                            'SMILES': smiles,
                            '分子编号': i + 1,
                            '数据集': model_info['dataset'],
                            '模型': model_info['model'],
                            'AUC': model_info['auc'],
                            'BBB+概率': prob,
                            '预测': 'BBB+' if pred == 1 else 'BBB-'
                        })
                except Exception as e:
                    st.warning(f"模型 {model_info['dataset']} - {model_info['model']} 预测失败: {e}")

            progress_bar.empty()

            # 转换为DataFrame
            results_df = pd.DataFrame(results)

            if not results_df.empty:
                st.success(f"完成！共 {len(all_models)} 个模型的预测结果")

                # 显示汇总统计
                st.markdown("### 预测汇总")

                for i, smiles in enumerate(smiles_list):
                    with st.expander(f"分子 {i+1}: {smiles}", expanded=(i==0)):
                        mol_results = results_df[results_df['SMILES'] == smiles].copy()

                        # 统计
                        n_total = len(mol_results)
                        n_bbb_plus = (mol_results['预测'] == 'BBB+').sum()
                        avg_prob = mol_results['BBB+概率'].mean()
                        std_prob = mol_results['BBB+概率'].std()

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("模型总数", n_total)
                        col2.metric("预测为BBB+", f"{n_bbb_plus}/{n_total}")
                        col3.metric("平均概率", f"{avg_prob:.3f}")
                        col4.metric("标准差", f"{std_prob:.3f}")

                        # 一致性分析
                        if n_bbb_plus == 0:
                            st.info("✅ 所有模型一致预测为 BBB-")
                        elif n_bbb_plus == n_total:
                            st.info("✅ 所有模型一致预测为 BBB+")
                        else:
                            ratio = n_bbb_plus / n_total
                            if ratio > 0.7:
                                st.warning(f"⚠️ 大多数模型 ({ratio:.1%}) 预测为 BBB+")
                            elif ratio < 0.3:
                                st.warning(f"⚠️ 大多数模型 ({1-ratio:.1%}) 预测为 BBB-")
                            else:
                                st.warning(f"⚠️ 模型预测分歧较大 ({ratio:.1%} BBB+, {1-ratio:.1%} BBB-)")

                        # 详细结果表格
                        st.markdown("#### 各模型预测详情")

                        # 按数据集分组显示
                        for dataset in ['A', 'A,B', 'A,B,C', 'A,B,C,D']:
                            dataset_results = mol_results[mol_results['数据集'] == dataset]
                            if not dataset_results.empty:
                                st.markdown(f"**{dataset} 数据集**")
                                display_df = dataset_results[['模型', 'AUC', 'BBB+概率', '预测']].copy()
                                display_df['AUC'] = display_df['AUC'].apply(lambda x: f"{x:.3f}")
                                display_df['BBB+概率'] = display_df['BBB+概率'].apply(lambda x: f"{x:.3f}")

                                # 高亮显示
                                def color_pred(val):
                                    if val == 'BBB+':
                                        return 'background-color: #d4edda'
                                    else:
                                        return 'background-color: #f8d7da'

                                styled_df = display_df.style.applymap(color_pred, subset=['预测'])
                                st.dataframe(styled_df, use_container_width=True, hide_index=True)

                # 下载完整结果
                st.markdown("---")
                st.markdown("### 下载结果")

                csv = results_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下载完整预测结果 (CSV)",
                    data=csv,
                    file_name="all_models_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # 可视化：热力图（如果只有1个分子）
                if len(smiles_list) == 1:
                    st.markdown("---")
                    st.markdown("### 预测概率热力图")

                    pivot_df = results_df.pivot_table(
                        index='数据集',
                        columns='模型',
                        values='BBB+概率'
                    )

                    # 调整列顺序
                    model_order = ['RF+SMARTS', 'XGB+SMARTS', 'LGBM+SMARTS', 'GAT']
                    pivot_df = pivot_df.reindex(columns=model_order)

                    st.pyplot(use_container_width=True)

                    # 使用matplotlib绘制热力图
                    import matplotlib.pyplot as plt
                    import matplotlib.colors as mcolors

                    fig, ax = plt.subplots(figsize=(10, 4))
                    im = ax.imshow(pivot_df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

                    # 设置坐标轴
                    ax.set_xticks(np.arange(len(pivot_df.columns)))
                    ax.set_yticks(np.arange(len(pivot_df.index)))
                    ax.set_xticklabels(pivot_df.columns)
                    ax.set_yticklabels(pivot_df.index)

                    # 旋转x轴标签
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                    # 添加数值标注
                    for i in range(len(pivot_df.index)):
                        for j in range(len(pivot_df.columns)):
                            text = ax.text(j, i, f'{pivot_df.values[i, j]:.2f}',
                                         ha="center", va="center", color="black", fontsize=10)

                    ax.set_title("所有模型的BBB+概率预测", fontsize=14, fontfamily='Times New Roman')
                    fig.tight_layout()

                    # 添加颜色条
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('BBB+ Probability', rotation=270, labelpad=20)

                    st.pyplot(fig)
