"""
主动学习模块 - 输入新分子、验证、预测、标注、更新数据库、重新训练

功能流程:
1. 输入新SMILES
2. 验证SMILES格式
3. 检查是否已在训练数据中
4. 如果不存在，进行预测
5. 用户手动确认标签（BBB+ 或 BBB-）
6. 保存到数据库
7. 累积一定数量后可触发重新训练
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy import sparse
from datetime import datetime

st.set_page_config(
    page_title="Active Learning",
    page_icon="🔄",
    layout="wide"
)

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
</style>
""", unsafe_allow_html=True)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 缓存目录
CACHE_DIR = PROJECT_ROOT / "artifacts" / "active_learning_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 工具函数 ====================

def validate_smiles(smiles: str) -> dict:
    """验证SMILES格式和有效性"""
    result = {
        'smiles': smiles,
        'valid': False,
        'canonical_smiles': None,
        'error': None,
        'molecular_weight': None,
        'num_atoms': None,
        'num_heavy_atoms': None
    }

    if not smiles or not smiles.strip():
        result['error'] = "SMILES为空"
        return result

    smiles = smiles.strip()

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result['error'] = "无效的SMILES"
            return result

        # 计算分子属性
        result['valid'] = True
        result['canonical_smiles'] = Chem.MolToSmiles(mol)
        result['num_atoms'] = mol.GetNumAtoms()
        result['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
        result['molecular_weight'] = Chem.Descriptors.MolWt(mol)

        return result

    except Exception as e:
        result['error'] = f"解析错误: {str(e)}"
        return result


def check_in_training_data(smiles: str, dataset: str) -> dict:
    """检查SMILES是否已在训练数据中"""
    try:
        split_dir = PROJECT_ROOT / "data" / "splits" / f"seed_0_{dataset}"

        # 检查所有文件
        for split in ['train', 'val', 'test']:
            csv_file = split_dir / f"{split}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                # 检查SMILES列（可能是SMILES或smiles）
                smiles_col = None
                for col in df.columns:
                    if col.lower() in ['smiles', 'smile']:
                        smiles_col = col
                        break

                if smiles_col and smiles in df[smiles_col].values:
                    return {
                        'exists': True,
                        'dataset': dataset,
                        'split': split,
                        'index': int(df[df[smiles_col] == smiles].index[0])
                    }

        return {'exists': False, 'dataset': dataset}

    except Exception as e:
        return {'exists': False, 'error': str(e)}


def predict_with_model(smiles: str, model_path, model_type: str) -> dict:
    """使用指定模型进行预测"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'error': '无效的SMILES'}

        # 计算特征
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

        if model_type == 'gnn':
            # GNN模型
            from src.featurize.graph_pyg import BBBGraphDataset, GraphBuildConfig
            from torch_geometric.loader import DataLoader
            from src.finetune.train_gat_bbb_from_pretrain import GATBBB, FinetuneCfg

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 创建临时DataFrame
            df_temp = pd.DataFrame({
                'SMILES': [smiles],
                'y_cls': [0],
                'row_id': [0]
            })

            gcfg = GraphBuildConfig(smiles_col="SMILES", label_col="y_cls", id_col="row_id")

            # 使用唯一缓存
            import time
            import uuid
            unique_id = f"pred_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            temp_cache = PROJECT_ROOT / "artifacts" / "temp_predict" / unique_id

            temp_ds = BBBGraphDataset(root=str(temp_cache), df=df_temp, cfg=gcfg)

            if len(temp_ds) == 0:
                return {'error': '图数据集创建失败'}

            # 加载模型
            checkpoint = torch.load(model_path, map_location=device)

            # 检测checkpoint结构
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                cfg = checkpoint.get('cfg', {})
            else:
                state_dict = checkpoint
                cfg = {}

            # 创建模型
            if isinstance(cfg, dict):
                hidden = cfg.get('hidden', 128)
            else:
                hidden = 128

            finetune_cfg = FinetuneCfg(
                seed=0,
                hidden=hidden,
                gat_heads=4,
                num_layers=3,
                dropout=0.2
            )

            model = GATBBB(temp_ds[0].x.size(-1), finetune_cfg).to(device)
            model.load_state_dict(state_dict)
            model.eval()

            # 预测
            loader = DataLoader(temp_ds, batch_size=1, shuffle=False)
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    logit = model(batch)
                    prob = torch.sigmoid(logit).cpu().numpy()[0, 0]
                    break

            return {'prob': float(prob), 'model_type': 'GNN'}

        else:
            # 传统ML模型
            if 'LGBM' in str(model_path):
                arr = np.zeros((2048,), dtype=np.float32)
            else:
                arr = np.zeros((2048,), dtype=np.int8)

            DataStructs.ConvertToNumpyArray(fp, arr)
            X = arr.reshape(1, -1)

            if 'LGBM' in str(model_path):
                model = joblib.load(model_path)
                prob = model.predict_proba(X)[0, 1]
            else:
                X = sparse.csr_matrix(X)
                model = joblib.load(model_path)
                prob = model.predict_proba(X)[0, 1]

            return {'prob': float(prob), 'model_type': 'ML'}

    except Exception as e:
        return {'error': f'预测错误: {str(e)}'}


def save_to_database(data: dict, database_file: Path):
    """保存到数据库（CSV格式）"""
    try:
        if database_file.exists():
            df = pd.read_csv(database_file)
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        else:
            df = pd.DataFrame([data])

        database_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(database_file, index=False)
        return True, len(df)
    except Exception as e:
        return False, str(e)


def load_database(database_file: Path) -> pd.DataFrame:
    """从数据库加载所有数据"""
    try:
        if database_file.exists():
            return pd.read_csv(database_file)
        else:
            return pd.DataFrame(columns=[
                'smiles', 'canonical_smiles', 'label', 'prediction', 'probability',
                'model', 'dataset', 'confidence', 'timestamp', 'user_added'
            ])
    except Exception as e:
        st.error(f"加载数据库失败: {e}")
        return pd.DataFrame()


def count_database_statistics(df: pd.DataFrame) -> dict:
    """统计数据库信息"""
    if len(df) == 0:
        return {
            'total': 0,
            'bbb_plus': 0,
            'bbb_minus': 0,
            'recent_additions': 0
        }

    return {
        'total': len(df),
        'bbb_plus': int(df['label'].value_counts().get(1, 0)),
        'bbb_minus': int(df['label'].value_counts().get(0, 0)),
        'recent_additions': len(df[df['user_added'] == True])
    }


# ==================== 主界面 ====================

st.title("🔄 Active Learning - 新分子添加与模型更新")
st.markdown("---")

# 侧边栏
st.sidebar.header("⚙️ 配置")

# 数据集选择
dataset_options = {
    'A': 'A',
    'A,B': 'A_B',
    'A,B,C': 'A_B_C',
    'A,B,C,D': 'A_B_C_D'
}

selected_dataset_display = st.sidebar.selectbox(
    "选择数据集",
    options=list(dataset_options.keys()),
    index=1  # 默认A+B
)

selected_dataset = dataset_options[selected_dataset_display]

# 模型选择
@st.cache_data
def load_model_info():
    perf_file = PROJECT_ROOT / "outputs" / "all_16_models_performance.csv"
    if perf_file.exists():
        df = pd.read_csv(perf_file)
        return df
    return None

perf_df = load_model_info()

# 获取可用模型
def get_models_for_dataset(dataset, perf_df):
    if perf_df is None:
        MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / f"seed_0_{dataset}"
        return {
            'Random Forest': MODEL_DIR / "baseline" / "RF_seed0.joblib",
            'XGBoost': MODEL_DIR / "baseline" / "XGB_seed0.joblib",
            'LightGBM': MODEL_DIR / "baseline" / "LGBM_seed0.joblib",
        }

    dataset_df = perf_df[perf_df['Dataset'] == dataset]

    models_dict = {}
    for _, row in dataset_df.iterrows():
        MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / f"seed_0_{dataset}"

        if row['Model'] == 'RF':
            models_dict['Random Forest'] = MODEL_DIR / "baseline" / "RF_seed0.joblib"
        elif row['Model'] == 'XGB':
            models_dict['XGBoost'] = MODEL_DIR / "baseline" / "XGB_seed0.joblib"
        elif row['Model'] == 'LGBM':
            models_dict['LightGBM'] = MODEL_DIR / "baseline" / "LGBM_seed0.joblib"
        elif row['Model'] == 'GAT+SMARTS':
            if dataset in ['A_B_C', 'A_B_C_D']:
                models_dict['GAT+SMARTS'] = MODEL_DIR / "gat_finetune_bbb" / "pretrained_partial_inmemory" / "best.pt"
            else:
                models_dict['GAT+SMARTS'] = MODEL_DIR / "gat_finetune_bbb" / "pretrained_partial" / "best.pt"

    return models_dict


models_available = get_models_for_dataset(selected_dataset, perf_df)

# 模型类型映射
def get_model_type(model_name: str) -> str:
    if 'GAT' in model_name:
        return 'gnn'
    return 'rf'


selected_model = st.sidebar.selectbox(
    "选择预测模型",
    options=list(models_available.keys()),
    index=0
)

model_path = models_available[selected_model]
model_type = get_model_type(selected_model)

st.sidebar.caption(f"**数据集**: {selected_dataset_display}")
st.sidebar.caption(f"**模型**: {selected_model}")

# 显示模型性能
if perf_df is not None:
    # Clean model name for comparison (remove AUC suffix)
    model_clean = selected_model.split(' (AUC=')[0]
    model_row = perf_df[
        (perf_df['Dataset'] == selected_dataset) &
        (perf_df['Model'] == model_clean)
    ]

    if len(model_row) > 0:
        perf = model_row.iloc[0]
        st.sidebar.metric("AUC", f"{perf['AUC']:.4f}")
        st.sidebar.metric("Precision", f"{perf['Precision']:.4f}")

# 数据库文件
DATABASE_FILE = CACHE_DIR / f"new_molecules_{selected_dataset}.csv"

# 加载数据库统计
st.sidebar.markdown("---")
st.sidebar.subheader("📊 数据库统计")

db_df = load_database(DATABASE_FILE)
db_stats = count_database_statistics(db_df)

st.sidebar.metric("总分子数", db_stats['total'])
st.sidebar.metric("BBB+", db_stats['bbb_plus'])
st.sidebar.metric("BBB-", db_stats['bbb_minus'])

# 主标签页
tab1, tab2, tab3 = st.tabs([
    "📝 输入新分子",
    "📊 查看数据库",
    "🔄 重新训练"
])

# ==================== Tab 1: 输入新分子 ====================

with tab1:
    st.header("📝 输入并验证新分子")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("步骤1: 输入SMILES")

        smiles_input = st.text_area(
            "输入SMILES（每行一个）",
            placeholder="CCO\nCC(=O)OC1=CC=C(C=C)C=C1\nCCN",
            height=150,
            help="输入有效的SMILES字符串，每行一个",
            key="smiles_input"
        )

        if st.button("🔍 验证SMILES", type="primary", use_container_width=True):
            if not smiles_input.strip():
                st.warning("请输入SMILES")
            else:
                # 验证每行SMILES
                smiles_list = [s.strip() for s in smiles_input.strip().split('\n') if s.strip()]

                st.subheader("验证结果")

                valid_smiles = []
                invalid_smiles = []

                for smi in smiles_list:
                    result = validate_smiles(smi)

                    if result['valid']:
                        valid_smiles.append(result)
                    else:
                        invalid_smiles.append(result)

                # 显示验证结果
                if valid_smiles:
                    st.success(f"✅ 有效SMILES: {len(valid_smiles)}个")

                    # 显示有效SMILES详情
                    with st.expander("查看有效SMILES详情"):
                        for i, result in enumerate(valid_smiles[:20]):  # 只显示前20个
                            st.markdown(f"""
                            **{i+1}. {result['smiles']}**
                            - 分子量: {result['molecular_weight']:.2f}
                            - 原子数: {result['num_atoms']}
                            - 重原子数: {result['num_heavy_atoms']}
                            """)

                        if len(valid_smiles) > 20:
                            st.caption(f"... 还有 {len(valid_smiles) - 20} 个")

                    # 检查是否在训练数据中
                    st.subheader("步骤2: 检查训练数据")

                    not_in_train = []
                    in_train = []

                    for result in valid_smiles:
                        canonical = result['canonical_smiles']
                        check_result = check_in_training_data(canonical, selected_dataset)

                        if check_result['exists']:
                            in_train.append(result)
                        else:
                            not_in_train.append(result)

                    st.info(f"在训练数据中: {len(in_train)}个")
                    st.info(f"新分子: {len(not_in_train)}个")

                    # 预测新分子
                    if not_in_train:
                        st.subheader("步骤3: 预测新分子")

                        results = []
                        for result in not_in_train[:10]:  # 只预测前10个
                            pred_result = predict_with_model(
                                result['canonical_smiles'],
                                model_path,
                                model_type
                            )

                            if 'error' in pred_result:
                                st.error(f"预测错误: {pred_result['error']}")
                            else:
                                prob = pred_result['prob']
                                pred_label = 'BBB+' if prob >= 0.5 else 'BBB-'
                                pred_confidence = 'High' if prob > 0.8 or prob < 0.2 else 'Medium'

                                results.append({
                                    'smiles': result['smiles'],
                                    'canonical_smiles': result['canonical_smiles'],
                                    'probability': prob,
                                    'prediction': pred_label,
                                    'confidence': pred_confidence
                                })

                        if results:
                            df_results = pd.DataFrame(results)
                            st.dataframe(
                                df_results,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    'smiles': 'SMILES',
                                    'canonical_smiles': 'Canonical SMILES',
                                    'probability': st.column_config.NumberColumn("BBB+概率", format="%.4f"),
                                    'prediction': '预测',
                                    'confidence': '置信度'
                                }
                            )

                            # 批量标注
                            st.subheader("步骤4: 批量标注")

                            st.markdown("""
                            **请确认预测结果**:
                            - ✅ = BBB+ (能透过血脑屏障)
                            - ❌ = BBB- (不能透过血脑屏障)

                            如果预测正确，点击"全部确认保存"
                            如果有错误，手动修改后再保存
                            """)

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                if st.button("✅ 全部确认保存", type="primary"):
                                    # 全部标记为预测结果
                                    for _, row in df_results.iterrows():
                                        row['label'] = 1 if row['prediction'] == 'BBB+' else 0
                                        row['user_added'] = True
                                        row['timestamp'] = datetime.now().isoformat()
                                        row['dataset'] = selected_dataset
                                        row['model'] = selected_model

                                    saved, total = save_to_database(row, DATABASE_FILE)

                                    if saved:
                                        st.success(f"成功保存 {total} 个分子到数据库！")
                                        st.rerun()

                            with col2:
                                if st.button("✅ 仅保存高置信度", type="secondary"):
                                    # 只保存高置信度的
                                    high_conf = df_results[df_results['confidence'] == 'High']

                                    for _, row in high_conf.iterrows():
                                        row['label'] = 1 if row['prediction'] == 'BBB+' else 0
                                        row['user_added'] = True
                                        row['timestamp'] = datetime.now().isoformat()
                                        row['dataset'] = selected_dataset
                                        row['model'] = selected_model

                                    if len(high_conf) > 0:
                                        saved, total = save_to_database(high_conf.iloc[0], DATABASE_FILE)
                                        st.success(f"保存了 {len(high_conf)} 个高置信度分子")
                                        st.rerun()

                            with col3:
                                if st.button("❌ 放弃", type="secondary"):
                                    st.info("已取消保存")

                else:
                    st.error(f"所有SMILES都无效！发现 {len(invalid_smiles)} 个错误")

                    if invalid_smiles:
                        with st.expander("查看无效SMILES详情"):
                            for i, result in enumerate(invalid_smiles[:10]):
                                st.error(f"{i+1}. {result['smiles']} - {result['error']}")

    with col2:
        st.subheader("💡 使用说明")

        st.markdown("""
        **工作流程**:

        1. **输入SMILES** - 每行一个SMILES字符串

        2. **验证格式** - 系统自动验证SMILES有效性
           - 检查分子量
           - 检查原子数

        3. **检查数据** - 查看是否已在训练数据中
           - 已存在：跳过
           - 不存在：进行预测

        4. **预测结果** - 使用选定模型预测
           - 显示BBB+概率
           - 显示预测标签
           - 显示置信度

        5. **确认保存** - 手动确认后保存
           - 全部保存：保存所有新分子
           - 仅保存高置信度：只保存概率>0.8或<0.2的
           - 放弃：不保存

        6. **数据库更新** - 保存到CSV文件
           - 文件位置: `artifacts/active_learning_cache/new_molecules_{dataset}.csv`
           - 包含SMILES、标签、预测信息、时间戳等

        **提示**:
        - 保存的分子可以用于后续重新训练
        - 建议至少累积100个新分子后再重新训练
        """)

# ==================== Tab 2: 查看数据库 ====================

with tab2:
    st.header("📊 新分子数据库")

    # 加载数据库
    df = load_database(DATABASE_FILE)

    if len(df) == 0:
        st.info("数据库为空，请先在【输入新分子】标签页添加分子")
    else:
        # 显示统计信息
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("总分子数", db_stats['total'])

        with col2:
            st.metric("BBB+", db_stats['bbb_plus'])

        with col3:
            st.metric("BBB-", db_stats['bbb_minus'])

        st.markdown("---")

        # 筛选选项
        col1, col2 = st.columns(2)

        with col1:
            show_label = st.selectbox("显示标签", ["全部", "仅BBB+", "仅BBB-"])

        with col2:
            sort_by = st.selectbox("排序方式", ["时间戳（新→旧）", "时间戳（旧→新）"])

        # 筛选数据
        if show_label == "全部":
            filtered_df = df
        elif show_label == "仅BBB+":
            filtered_df = df[df['label'] == 1]
        else:  # 仅BBB-
            filtered_df = df[df['label'] == 0]

        # 排序
        if sort_by == "时间戳（新→旧）":
            filtered_df = filtered_df.sort_values('timestamp', ascending=False)
        else:
            filtered_df = filtered_df.sort_values('timestamp', ascending=True)

        # 显示数据表格
        st.dataframe(
            filtered_df[['timestamp', 'smiles', 'label_display', 'prediction', 'probability', 'model', 'dataset', 'confidence']].rename(columns={
                'label_display': '标签',
                'prediction': '预测',
                'probability': 'BBB+概率'
            }),
            use_container_width=True,
            hide_index=True
        )

        # 导出功能
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 导出数据库"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="下载数据库 (CSV)",
                    data=csv,
                    file_name=f"new_molecules_{selected_dataset}.csv",
                    mime="text/csv"
                )

        with col2:
            if st.button("🗑️️ 清空数据库"):
                if st.checkbox("确认清空", value=False):
                    st.error("请勾选确认框")
                else:
                    try:
                        DATABASE_FILE.unlink()
                        st.success("数据库已清空")
                        st.rerun()
                    except Exception as e:
                        st.error(f"清空失败: {e}")

# ==================== Tab 3: 重新训练 ====================

with tab3:
    st.header("🔄 重新训练模型")

    st.markdown("""
    **重新训练流程**:

    当数据库累积足够多的新分子后，可以重新训练模型以提升性能。

    **建议**:
    - 至少累积50个新分子后再重新训练
    - 定期检查模型性能是否提升
    - 保存训练日志以便对比
    """)

    # 显示当前状态
    st.subheader("📊 当前状态")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("数据集", selected_dataset_display)
        st.metric("当前模型", selected_model)

    with col2:
        st.metric("数据库分子数", db_stats['total'])
        st.metric("建议最小数量", 50)

    with col3:
        can_retrain = db_stats['total'] >= 50
        st.metric("是否可训练", "✅ 是" if can_retrain else "❌ 否")

    st.markdown("---")

    if can_retrain:
        st.success(f"✅ 数据库已有 {db_stats['total']} 个分子，可以进行重新训练！")

        st.subheader("训练配置")

        # 训练参数
        col1, col2 = st.columns(2)

        with col1:
            new_epochs = st.slider(
                "训练轮数 (新增数据)",
                min_value=10,
                max_value=100,
                value=50,
                step=10
            )

            learning_rate = st.selectbox(
                "学习率",
                options=["0.001", "0.005", "0.01", "0.1"],
                index=1
            )

        with col2:
            use_original = st.checkbox("包含原始训练数据", value=True)

            st.markdown("""
            **建议配置**:
            - 包含原始数据：在新数据+原数据上重新训练
            - 仅新数据：只用新添加的数据
            """)

        # 训练按钮
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🚀 开始重新训练", type="primary"):
                st.info("准备开始训练...")
                st.info("功能开发中，请使用命令行脚本")
                model_clean = selected_model.replace(' ', '_').lower()
                cmd = f"""python scripts/retrain_with_new_data.py \\
    --dataset {selected_dataset} \\
    --model {model_clean} \\
    --new_data artifacts/active_learning_cache/new_molecules_{selected_dataset}.csv \\
    --epochs {new_epochs} \\
    --lr {learning_rate}"""
                st.code(cmd)

        with col2:
            if st.button("📊 查看训练历史"):
                st.info("训练历史功能开发中...")

        with col3:
            if st.button("💾 下载当前数据库"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="下载数据库",
                    data=csv,
                    file_name=f"new_molecules_{selected_dataset}.csv",
                    mime="text/csv"
                )

    else:
        st.warning(f"❌ 数据库分子数不足（当前{db_stats['total']}个），建议至少50个后再重新训练")

        st.markdown("---")
        st.subheader("📝 训练脚本使用说明")

        st.code("""
# 使用独立的训练脚本重新训练
python scripts/retrain_with_new_data.py \\
    --dataset A_B \\
    --model xgboost \\
    --new_data artifacts/active_learning_cache/new_molecules_A_B.csv \\
    --epochs 100 \\
    --lr 0.01
        """)

        st.markdown("""
        **参数说明**:
        - `--dataset`: 数据集 (A, A_B, A_B_C, A_B_C_D)
        - `--model`: 模型类型 (rf, xgb, lgbm)
        - `--new_data`: 新分子���据库文件路径
        - `--epochs`: 训练轮数
        - `--lr`: 学习率
        """)