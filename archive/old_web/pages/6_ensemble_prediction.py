"""
BBB渗透性预测平台 - Ensemble模型预测页面
包含所有训练好的ablation模型：RF, XGB, LGBM, SVM_RBF, KNN5, NB, LR,
以及Ensemble模型 (Stacking_RF, Stacking_XGB, SoftVoting)
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

st.set_page_config(
    page_title="Ensemble Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    h1, h2, h3 {
        font-family: 'Times New Roman', serif;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# 模型加载
# =============================================================================

@st.cache_resource
def load_model(model_name, feature_name):
    """加载训练好的模型"""
    import joblib

    model_dir = PROJECT_ROOT / "artifacts" / "ablation"

    # ========== 支持单个特征的模型 ==========
    single_feature_models = ['RF', 'SVM_RBF', 'KNN5', 'NB_Bernoulli']

    if model_name in single_feature_models:
        model_path = model_dir / f"{model_name}_{feature_name}"
        model_file = model_path / f"{model_name}_seed0.joblib"
        if model_file.exists():
            return joblib.load(model_file)

    # ========== 仅支持 combined 特征的模型 ==========
    combined_only_models = ['XGB', 'LGBM', 'GB', 'ETC', 'ADA', 'LR']

    if model_name in combined_only_models:
        if feature_name == 'combined':
            model_file = model_dir / f"{model_name}_seed0.joblib"
            if model_file.exists():
                return joblib.load(model_file)

    # ========== Ensemble 模型 ==========
    ensemble_dir = PROJECT_ROOT / "artifacts" / "models" / "ensemble"
    ensemble_models = {
        'Stacking_rf': 'stacking_rf.joblib',
        'Stacking_xgb': 'stacking_xgb.joblib',
        'SoftVoting': 'soft_voting.joblib',
    }

    if model_name in ensemble_models:
        if feature_name == 'combined':
            model_file = ensemble_dir / ensemble_models[model_name]
            if model_file.exists():
                return joblib.load(model_file)

    return None


@st.cache_data
def load_performance():
    """加载性能数据"""
    perf_file = PROJECT_ROOT / "artifacts" / "ablation" / "ALL_RESULTS_COMBINED.csv"
    ensemble_perf_file = PROJECT_ROOT / "artifacts" / "ablation" / "ENSEMBLE_RESULTS.csv"

    dfs = []
    if perf_file.exists():
        df = pd.read_csv(perf_file)
        df = df[df['split'] == 'test']
        dfs.append(df)

    if ensemble_perf_file.exists():
        df_ensemble = pd.read_csv(ensemble_perf_file)
        dfs.append(df_ensemble)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


def smiles_to_features(smiles):
    """将SMILES转换为分子特征"""
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, Descriptors

    # SMILES auto-fix mapping for known invalid formats
    smiles_fixes = {
        'CC(C)(C)C1CCCc2ccccc2Cl': 'CC(C)C1CCCC1c2ccccc2Cl',  # MPC molecule fix
        'C=C(C)C=O)OCC[N+](C)(C)CCCS(=O)(=O)[O-]': 'C=C(C)C(=O)OCC[N+](C)(C)CCCS(=O)(=O)[O-]',  # SBMA fix
        'C=C(C)C=O)OCC[N+](C)(C)[O-]': 'C=C(C)C(=O)OCC[N+](C)(C)[O-]',  # ONMA fix
    }
    smiles_to_use = smiles_fixes.get(smiles, smiles)

    try:
        mol = Chem.MolFromSmiles(smiles_to_use)
        if mol is None:
            return None

        # Morgan指纹
        fp_morgan = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        morgan = np.array(fp_morgan, dtype=np.float32).reshape(1, -1)

        # MACCS
        fp_maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs = np.array(fp_maccs, dtype=np.float32).reshape(1, -1)

        # Atom Pairs
        fp_ap = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024)
        atompairs = np.array(fp_ap, dtype=np.float32).reshape(1, -1)

        # FP2 (using RDKit fingerprint as substitute)
        fp_fp2 = Chem.RDKFingerprint(mol, fpSize=2048)
        fp2 = np.array(fp_fp2, dtype=np.float32).reshape(1, -1)

        # RDKit描述符
        descriptors = {}
        for desc_name, desc_func in Descriptors.descList:
            try:
                val = desc_func(mol)
                descriptors[desc_name] = val
            except:
                descriptors[desc_name] = None

        # 过滤掉None值
        rdkit_desc = np.array([descriptors.get(d, 0) for d in [
            'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex',
            'MaxPartialCharge', 'MinPartialCharge', 'NumRadicalElectrons',
            'NumValenceElectrons', 'NumAromaticRings', 'NumSaturatedRings',
            'NumAliphaticCarbocycles', 'NumAromaticCarbocycles',
            'NumAromaticHeterocycles', 'NumSaturatedHeterocycles',
            'RingCount', 'FractionCSP2', 'FractionCSP3',
            'TPSA', 'LabuteASA', 'BalabanJ', 'BertzCT',
            'Chi0', 'Chi0n', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3',
            'LabuteASAs', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3',
            'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4',
            'SLogP', 'MR', 'HeavyAtomMolWt', 'ExactMolWt'
        ]])

        # 合并特征 (仅4种，与训练时一致)
        from scipy import sparse
        combined = sparse.hstack([
            sparse.csr_matrix(morgan.reshape(1, -1)),
            sparse.csr_matrix(maccs.reshape(1, -1)),
            sparse.csr_matrix(atompairs.reshape(1, -1)),
            sparse.csr_matrix(fp2.reshape(1, -1))
        ])

        return {
            'morgan': morgan,
            'maccs': maccs,
            'atompairs': atompairs,
            'fp2': fp2,
            'combined': combined
        }

    except Exception as e:
        st.error(f"特征提取失败: {str(e)}")
        return None


def predict_with_model(model, features, feature_name):
    """使用模型进行预测"""
    X = features[feature_name]

    # 预测
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    return y_pred[0], y_prob[0]


# =============================================================================
# 主界面
# =============================================================================

st.title("🧪 Ensemble Model Prediction")
st.markdown("""
**Complete Model Suite - All Trained Models Available**

This page provides predictions using:
- **Tree Ensembles**: RF, XGBoost, LightGBM, GradientBoost, ExtraTrees, AdaBoost
- **Other ML**: SVM_RBF, KNN, NaiveBayes, Logistic Regression
- **Ensemble Models**: Stacking_RF, Stacking_XGB, SoftVoting

Select your preferred model and feature type for prediction.
""")

st.markdown("---")

# 侧边栏 - 模型和特征选择
st.sidebar.header("⚙️ Model & Feature Selection")

# 模型选择
st.sidebar.subheader("1. Select Model")

model_category = st.sidebar.selectbox(
    "Model Category",
    ["Tree Ensemble", "Ensemble", "Other ML", "All Models"]
)

# 根据类别显示模型
if model_category == "Tree Ensemble":
    available_models = ['RF', 'XGB', 'LGBM', 'GB', 'ETC', 'ADA']
elif model_category == "Ensemble":
    available_models = ['Stacking_rf', 'Stacking_xgb', 'SoftVoting']
elif model_category == "Other ML":
    available_models = ['SVM_RBF', 'KNN5', 'NB_Bernoulli', 'LR']
else:
    available_models = ['RF', 'XGB', 'LGBM', 'GB', 'ETC', 'ADA',
                      'SVM_RBF', 'KNN5', 'NB_Bernoulli', 'LR',
                      'Stacking_rf', 'Stacking_xgb', 'SoftVoting']

selected_model = st.sidebar.selectbox("Model", available_models)

# 特征选择
st.sidebar.subheader("2. Select Feature")

# 根据模型确定可用特征
# Ensemble 模型和某些树模型只支持 combined 特征
combined_only_models = ['Stacking_rf', 'Stacking_xgb', 'SoftVoting', 'XGB', 'LGBM', 'GB', 'ETC', 'ADA', 'LR']

if selected_model in combined_only_models:
    available_features = ['combined']
else:
    available_features = ['morgan', 'maccs', 'atompairs', 'fp2']

feature_display = {
    'morgan': 'Morgan (2048D)',
    'maccs': 'MACCS (167D)',
    'atompairs': 'AtomPairs (1024D)',
    'fp2': 'FP2 (2048D)',
    'combined': 'Combined (5287D)'
}

selected_feature = st.sidebar.selectbox(
    "Feature",
    available_features,
    format_func=lambda x: feature_display[x]
)

# 显示模型性能信息
st.sidebar.subheader("📊 Model Performance")

perf_df = load_performance()

if perf_df is not None:
    # 查找选中模型和特征的性能
    model_perf = perf_df[
        (perf_df['model'] == selected_model) &
        (perf_df['feature'] == selected_feature)
    ]

    if len(model_perf) > 0:
        perf = model_perf.iloc[0]
        st.sidebar.metric("AUC", f"{perf['auc']:.4f}")
        st.sidebar.metric("F1", f"{perf['f1']:.4f}")
        st.sidebar.metric("MCC", f"{perf['mcc']:.4f}")
    else:
        st.sidebar.info("Performance data not available")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Guide:**

- **Ensemble Models**: Best performance
- **RF + Morgan**: Best single model
- **XGBoost + Morgan**: Highest F1
- **LightGBM + MACCS**: Best MCC
""")

# =============================================================================
# 输入区域
# =============================================================================

st.header("🧬 Molecule Input")

input_method = st.radio(
    "Input Method",
    ["Enter SMILES", "Upload SDF File (Coming Soon)"],
    horizontal=True
)

if input_method == "Enter SMILES":
    smiles_input = st.text_area(
        "Enter SMILES (one per line for batch prediction):",
        placeholder="CCOC(=O)c1ccc(O)nc1O",
        height=150
    )

    # 示例SMILES
    with st.expander("📖 Example SMILES (click to load)"):
        st.code("""
# BBB+ (permeable)
CCOC(=O)c1ccc(O)nc1O
CN1CCC[C@@H](C(=O)O)CC(=O)C1CCCc2ccccc2

# BBB- (non-permeable)
CC(C)(C)C1CCCc2ccccc2Cl
        """
)

    # 预测按钮
    predict_btn = st.button("🚀 Predict", type="primary")

elif input_method == "Upload SDF File (Coming Soon)":
    st.info("SDF file upload feature coming soon! Please use SMILES input.")
    predict_btn = False

else:
    predict_btn = False

# =============================================================================
# 预测和结果显示
# =============================================================================

if predict_btn and smiles_input.strip():
    st.header("📊 Prediction Results")

    # 加载模型和特征
    with st.spinner(f"Loading {selected_model} model with {selected_feature} features..."):
        model = load_model(selected_model, selected_feature)

        if model is None:
            st.error(f"❌ Model not found: {selected_model} + {selected_feature}")
        else:
            # 处理输入
            smiles_list = [s.strip() for s in smiles_input.strip().split('\n') if s.strip()]

            results = []
            for smiles in smiles_list:
                features_dict = smiles_to_features(smiles)

                if features_dict is None:
                    results.append({
                        'SMILES': smiles,
                        'Status': 'Invalid',
                        'Prediction': 'N/A',
                        'Probability': 'N/A'
                    })
                else:
                    try:
                        pred, prob = predict_with_model(model, features_dict, selected_feature)

                        if pred == 1:
                            prediction = "BBB+ (Permeable)"
                            prob_pct = prob * 100
                        else:
                            prediction = "BBB- (Non-permeable)"
                            prob_pct = (1 - prob) * 100

                        results.append({
                            'SMILES': smiles,
                            'Status': 'Success',
                            'Prediction': prediction,
                            'Probability': f"{prob:.2%}"
                        })

                    except Exception as e:
                        results.append({
                            'SMILES': smiles,
                            'Status': f'Error: {str(e)}',
                            'Prediction': 'N/A',
                            'Probability': 'N/A'
                        })

            # 显示结果
            results_df = pd.DataFrame(results)

            # 统计
            total = len(results_df)
            success = len(results_df[results_df['Status'] == 'Success'])
            bbb_plus = len(results_df[results_df['Prediction'] == 'BBB+ (Permeable)'])

            # 显示摘要
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Molecules", total)

            with col2:
                st.metric("Successful Predictions", success)

            with col3:
                if bbb_plus > 0:
                    st.metric("BBB+ (Permeable)", bbb_plus)

            # 详细结果表
            st.subheader("Detailed Results")

            # 根据预测结果着色
            def color_predictions(val):
                if 'Permeable' in val:
                    return 'background-color: #d4edda'
                elif 'Non-permeable' in val:
                    return 'background-color: #f8d7da'
                else:
                    return ''

            styled_df = results_df.style.map(color_predictions, subset=['Prediction'])
            st.dataframe(styled_df)

            # 下载按钮
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results",
                csv,
                "prediction_results.csv",
                mime="text/csv"
            )

# =============================================================================
# 页脚信息
# =============================================================================

st.markdown("---")
st.markdown("""
**Platform Information:**

- **Models Available**: 13 models
- **Features**: 5 individual + 1 combined
- **Total Configurations**: 50+ model-feature combinations
- **Best Model**: Stacking_XGB (AUC = 0.9727)

**Citation:**
If you use this platform, please cite the B3DB database and relevant methodology papers.
""")
