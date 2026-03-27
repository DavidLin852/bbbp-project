"""
BBB渗透性预测平台 - 主页（16��模型版本）
"""
import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="BBB Prediction Platform",
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
    .stApp {
        background-color: #f5f5f5;
    }
    h1, h2, h3 {
        font-family: 'Times New Roman', serif;
    }
    .feature-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .stat-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 主页标题
st.title("🧪 BBB Permeability Prediction Platform")
st.markdown("""
**32 Models × 4 Datasets = Complete Coverage for BBB Prediction**

This platform uses state-of-the-art machine learning models to predict blood-brain barrier (BBB) permeability for small molecules.
Includes baseline, SMARTS-enhanced, and GAT models for comprehensive prediction options.
""")

st.markdown("""---""")

# 统计信息
@st.cache_data
def load_stats():
    """加载所有模型统计信息（包括SMARTS增强模型）"""
    # 加载原始16个模型
    perf_file = Path(__file__).parent / "outputs" / "all_16_models_performance.csv"
    if perf_file.exists():
        original_df = pd.read_csv(perf_file)
    else:
        original_df = None

    # 加载SMARTS增强模型
    extended_models = []
    for dataset in ['A', 'A_B', 'A_B_C', 'A_B_C_D']:
        ext_file = Path(__file__).parent / "outputs" / f"extended_models_{dataset}_seed0.csv"
        if ext_file.exists():
            ext_df = pd.read_csv(ext_file)
            # 添加缺失的列（如果存在）
            if 'TN' not in ext_df.columns:
                ext_df['TN'] = -1
            if 'FN' not in ext_df.columns:
                ext_df['FN'] = -1
            extended_models.append(ext_df)

    if extended_models:
        extended_df = pd.concat(extended_models, ignore_index=True)
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

perf_df = load_stats()

if perf_df is not None:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="stat-box">
            <h2 style="color: #3498db;">32</h2>
            <p>Trained Models</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-box">
            <h2 style="color: #2ecc71;">4</h2>
            <p>Datasets</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # 最高AUC
        best_auc = perf_df.loc[perf_df['AUC'].idxmax()]
        st.markdown(f"""
        <div class="stat-box">
            <h2 style="color: #e74c3c;">{best_auc['AUC']:.3f}</h2>
            <p>Best AUC ({best_auc['Model']} - {best_auc['Dataset']})</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        # 最高Precision
        best_prec = perf_df.loc[perf_df['Precision'].idxmax()]
        st.markdown(f"""
        <div class="stat-box">
            <h2 style="color: #9b59b6;">{best_prec['Precision']:.3f}</h2>
            <p>Best Precision ({best_prec['Model']} - {best_prec['Dataset']})</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""---""")

# 简介
st.markdown("""
## 🎯 Platform Features

**🔬 Multi-Dataset Support**
- **A**: 846 samples (High quality) - Best for precision
- **A,B**: 3,743 samples (Recommended) - Best overall performance
- **A,B,C**: 6,203 samples (Extended) - Large scale coverage
- **A,B,C,D**: 6,244 samples (Complete) - Maximum data

**🤖 Model Diversity**
- **Random Forest / RF+SMARTS**: Fast and reliable (8 models)
- **XGBoost / XGB+SMARTS**: Best AUC performance (8 models)
- **LightGBM / LGBM+SMARTS**: Highest precision, lowest FP (8 models)
- **GAT+SMARTS**: Chemical structure awareness, highest recall (4 models)
- **GAT**: Random initialization baseline (4 models)

**🧬 SMARTS Analysis**
- Identify chemical substructures that promote/hinder BBB permeability
- Visualize structure-activity relationships
- Guide molecular design for better BBB penetration
""")

st.markdown("""---""")

# 功能卡片
st.subheader("📱 Available Pages")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🔍 Prediction</h3>
        <p>Predict BBB permeability for your molecules:</p>
        <ul>
            <li><b>4 Datasets</b>: Choose A, A,B, A,B,C, or A,B,C,D</li>
            <li><b>8 Model Types</b>: RF, RF+SMARTS, XGB, XGB+SMARTS, LGBM, LGBM+SMARTS, GAT+SMARTS, GAT</li>
            <li><b>32 Combinations</b>: Select the best model for your needs</li>
            <li><b>Single/Batch</b>: One molecule or bulk CSV upload</li>
            <li><b>Real-time Performance</b>: See AUC, Precision, FP for each model</li>
        </ul>
        <p><a href="?page=prediction" target="_self">👉 Go to Prediction Page</a></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Model Comparison</h3>
        <p>Explore all 32 models in detail:</p>
        <ul>
            <li><b>Interactive Charts</b>: Bar plots, scatter plots, heatmaps</li>
            <li><b>Side-by-side Comparison</b>: Compare by dataset or model type</li>
            <li><b>Performance Metrics</b>: AUC, Precision, Recall, F1</li>
            <li><b>Smart Recommendations</b>: Best model for each use case</li>
            <li><b>Data Export</b>: Download comparison results as CSV</li>
        </ul>
        <p><a href="?page=model_comparison" target="_self">👉 Go to Comparison Page</a></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""---""")

# 模型性能对比
st.subheader("🏆 Top Models by Dataset")

if perf_df is not None:
    datasets = ['A', 'A_B', 'A_B_C', 'A_B_C_D']
    dataset_names = {
        'A': 'A',
        'A_B': 'A,B',
        'A_B_C': 'A,B,C',
        'A_B_C_D': 'A,B,C,D'
    }

    for dataset in datasets:
        dataset_df = perf_df[perf_df['Dataset'] == dataset].sort_values('AUC', ascending=False)

        with st.expander(f"📊 {dataset_names[dataset]}"):
            col1, col2, col3, col4 = st.columns(4)

            for idx, (_, row) in enumerate(dataset_df.iterrows()):
                with [col1, col2, col3, col4][idx % 4]:
                    color = "#2ecc71" if idx == 0 else "#95a5a6"
                    st.markdown(f"""
                    <div style="padding: 10px; background: white; border-radius: 8px; border-left: 4px solid {color};">
                        <h5>{row['Model']}</h5>
                        <p style="margin: 5px 0;"><b>AUC:</b> {row['AUC']:.4f}</p>
                        <p style="margin: 5px 0;"><b>Precision:</b> {row['Precision']:.4f}</p>
                        <p style="margin: 5px 0;"><b>Recall:</b> {row['Recall']:.4f}</p>
                        <p style="margin: 5px 0;"><b>F1:</b> {row['F1']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)

st.markdown("""---""")

# 使用说明
st.subheader("📖 Quick Start Guide")

tab1, tab2, tab3 = st.tabs(["Prediction", "Model Selection", "Best Practices"])

with tab1:
    st.markdown("""
    ### How to Predict BBB Permeability:

    1. **Go to Prediction Page** (click on the sidebar or the button above)
    2. **Select Dataset**: Choose from A, A,B, A,B,C, or A,B,C,D
       - **A**: Highest precision, lowest false positives
       - **A,B**: Best overall performance (recommended)
       - **A,B,C**: Larger dataset, good balance
       - **A,B,C,D**: Maximum data coverage

    3. **Select Model**: Based on your priority
       - **RF+SMARTS**: Highest AUC (best accuracy)
       - **LGBM+SMARTS**: Highest precision (most reliable)
       - **RF**: Fastest prediction
       - **GAT+SMARTS**: Highest recall (won't miss potential BBB+)

    4. **Enter SMILES**: One per line for single prediction
    5. **Click Predict**: View results with confidence scores

    ### For Batch Prediction:
    1. Upload CSV file with 'smiles' column
    2. Select dataset and model
    3. Click batch prediction
    4. Download results as CSV
    """)

with tab2:
    st.markdown("""
    ### Model Selection Guide:

    **By Application:**
    - **Drug Discovery**: Use LGBM+SMARTS (A,B) for highest precision (0.964)
    - **Production Deployment**: Use RF+SMARTS (A,B) for best AUC (0.986)
    - **High-Throughput Screening**: Use RF for speed
    - **Exploratory Research**: Use GAT+SMARTS for chemical insights

    **By Performance Metric:**
    - **Highest AUC** (0.986): RF+SMARTS on A,B
    - **Highest Precision** (0.964): LGBM+SMARTS on A,B
    - **Highest Recall** (0.977): RF+SMARTS on A,B
    - **Highest F1** (0.958): RF+SMARTS on A,B

    **By Dataset Size:**
    - **Small & Precise** (846 samples): A group
    - **Balanced** (3,743 samples): A,B group ⭐ Recommended
    - **Large Scale** (6,203 samples): A,B,C group
    - **Maximum Coverage** (6,244 samples): A,B,C,D group
    """)

with tab3:
    st.markdown("""
    ### Best Practices:

    **For Reliable Results:**
    1. Start with A,B dataset + RF+SMARTS model (best overall)
    2. Validate predictions with multiple models
    3. Use confidence scores to filter uncertain predictions
    4. Check SMARTS analysis for chemical insights

    **For Drug Discovery:**
    1. Use A,B dataset + LGBM+SMARTS for highest precision
    2. Prioritize molecules with prediction probability > 0.8
    3. Use SMARTS analysis to understand substructure contributions
    4. Consider adjust threshold to 0.65 for more conservative predictions

    **For High-Throughput Screening:**
    1. Use RF models for fastest prediction
    2. Start with A,B dataset for good balance
    3. Use batch prediction for efficiency
    4. Export results for further analysis

    **For Novel Chemotypes:**
    1. Use GAT+SMARTS models (structure-aware)
    2. Compare predictions across multiple datasets
    3. Examine SMARTS substructure contributions
    4. Consider ensemble predictions for robustness
    """)

st.markdown("""---""")

# 页脚
st.markdown("""
<div style="text-align: center; color: #7f8c8d; margin-top: 50px;">
    <p><b>BBB Prediction Platform © 2024</b></p>
    <p>32 Models | 4 Datasets | Complete Coverage</p>
    <p><small>Machine Learning for Drug Discovery | Blood-Brain Barrier Permeability Prediction</small></p>
</div>
""", unsafe_allow_html=True)
