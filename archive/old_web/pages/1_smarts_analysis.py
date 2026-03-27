"""
SMARTS子结构��要性分析页面

显示：
1. 正贡献SMARTS（提高BBB+概率）
2. 负贡献SMARTS（降低BBB+概率）
3. SMARTS化学结构可视化
4. 统计信息
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

st.set_page_config(
    page_title="SMARTS Analysis",
    page_icon="🧬",
    layout="wide"
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
    h1, h2, h3 {
        font-family: 'Times New Roman', serif;
    }
    .smarts-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏
st.sidebar.title("🧬 SMARTS分析")

# 配置
SMARTS_FILE = PROJECT_ROOT / "assets" / "smarts" / "bbb_smarts_v1.json"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "explain" / "smarts_analysis"

st.sidebar.markdown("---")
st.sidebar.markdown("## 数据源")

# 选择数据文件 - 匹配所有smarts_importance相关文件
result_files = sorted(list(RESULTS_DIR.glob("smarts_importance*seed*.csv"))) if RESULTS_DIR.exists() else []

if result_files:
    selected_file = st.sidebar.selectbox(
        "选择分析结果",
        options=result_files,
        format_func=lambda x: x.name
    )
else:
    st.sidebar.error("未找到SMARTS分析结果")
    st.info("请先运行 `python analyze_smarts_importance.py --seed 0 --baseline`")
    st.stop()

# 加载SMARTS模式
def load_smarts_patterns(smarts_file):
    """加载SMARTS模式"""
    with open(smarts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    patterns = {}
    for item in data:
        smarts = item.get('smarts', '').replace(' ', '')
        name = item.get('name', '')
        if smarts and name:
            patterns[name] = smarts

    return patterns

# 加载数据
@st.cache_data
def load_analysis_data(file_path):
    """加载分析结果"""
    return pd.read_csv(file_path)

# 主界面
st.title("SMARTS Substructure Importance Analysis")
st.markdown("""
This page analyzes which chemical substructures (SMARTS patterns) contribute most to blood-brain barrier (BBB) permeability.

**Methodology:**
- For each SMARTS pattern, compare molecules that contain it vs those that don't
- Calculate the difference in BBB+ probability
- Positive delta: promotes BBB permeability
- Negative delta: hinders BBB permeability
""")

# 加载数据
df = load_analysis_data(selected_file)
patterns = load_smarts_patterns(SMARTS_FILE)

st.sidebar.markdown("---")
st.sidebar.markdown("## 显示选项")

top_n = st.sidebar.slider("显示Top N", min_value=5, max_value=50, value=20, step=5)
show_structures = st.sidebar.checkbox("显示化学结构", value=True)
filter_type = st.sidebar.radio(
    "筛选类型",
    ["全部", "正贡献 (促进BBB+)", "负贡献 (阻碍BBB+)"]
)

# 统计信息
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total SMARTS", len(df))

with col2:
    n_positive = (df['delta_prob'] > 0).sum()
    st.metric("Positive SMARTS", n_positive, delta=f"{n_positive/len(df)*100:.1f}%")

with col3:
    n_negative = (df['delta_prob'] < 0).sum()
    st.metric("Negative SMARTS", n_negative, delta=f"{n_negative/len(df)*100:.1f}%")

with col4:
    global_avg_bbb = df['avg_prob_with'].mean()
    st.metric("Avg BBB+ Rate", f"{global_avg_bbb:.1%}")

st.markdown("---")

# 筛选数据
if filter_type == "正贡献 (促进BBB+)":
    df_display = df[df['delta_prob'] > 0].head(top_n)
    title = f"Top {top_n} Positive SMARTS (Promote BBB+)"
elif filter_type == "负贡献 (阻碍BBB+)":
    df_display = df[df['delta_prob'] < 0].tail(top_n).sort_values('delta_prob', ascending=True)
    title = f"Top {top_n} Negative SMARTS (Hinder BBB+)"
else:
    # 显示正负各一半
    n_pos = top_n // 2
    n_neg = top_n - n_pos
    positive_part = df[df['delta_prob'] > 0].head(n_pos)
    negative_part = df[df['delta_prob'] < 0].tail(n_neg).sort_values('delta_prob', ascending=True)
    df_display = pd.concat([positive_part, negative_part])
    title = f"Top {top_n} SMARTS by Importance"

st.subheader(title)

# 显示SMARTS卡片
for idx, row in df_display.iterrows():
    smarts_name = row['smarts']
    delta = row['delta_prob']
    n_mol = row['n_molecules']
    freq = row['freq']
    pos_rate = row.get('pos_rate', 0)

    # 颜色编码
    if delta > 0:
        delta_color = "🟢"
        delta_text = f"+{delta:.3f}"
    else:
        delta_color = "🔴"
        delta_text = f"{delta:.3f}"

    # 创建卡片
    st.markdown(f"""
    <div class="smarts-card">
        <h4>{delta_color} <b>{smarts_name}</b></h4>
    </div>
    """, unsafe_allow_html=True)

    # 两列布局
    col1, col2 = st.columns([1, 2])

    with col1:
        # 显示SMARTS模式
        smarts_pattern = patterns.get(smarts_name, "N/A")
        st.code(f"SMARTS: {smarts_pattern}", language="text")

        # 统计信息
        st.markdown("**Statistics:**")
        st.markdown(f"- **Δ Probability:** {delta_text}")
        st.markdown(f"- **Molecules:** {n_mol} ({freq:.1%} of dataset)")
        if pd.notna(pos_rate):
            st.markdown(f"- **BBB+ Rate:** {pos_rate:.1%}")

    with col2:
        # 显示化学结构
        if show_structures and smarts_pattern != "N/A":
            try:
                # 尝试从SMARTS生成分子结构
                mol = Chem.MolFromSmarts(smarts_pattern)
                if mol is not None:
                    # 转换为RDKit Mol对象以便显示
                    try:
                        # 尝试创建一个简单的分子来展示SMARTS
                        mol_img = Draw.MolToImage(mol, size=(200, 150))
                        st.image(mol_img, caption=f"{smarts_name} structure", width=200)
                    except:
                        st.caption("Structure visualization not available")
                else:
                    st.caption("Could not generate structure from SMARTS")
            except Exception as e:
                st.caption(f"Structure error: {str(e)[:50]}")

    st.markdown("---")

# 详细数据表格
with st.expander("View Full Data Table"):
    st.dataframe(
        df_display[['smarts', 'delta_prob', 'n_molecules', 'freq', 'avg_prob_with', 'pos_rate']],
        column_config={
            'smarts': st.column_config.TextColumn("SMARTS Name", width="medium"),
            'delta_prob': st.column_config.NumberColumn("Δ Probability", format="%.4f"),
            'n_molecules': st.column_config.NumberColumn("Molecules", width="small"),
            'freq': st.column_config.NumberColumn("Frequency", format="%.4f"),
            'avg_prob_with': st.column_config.NumberColumn("BBB+ Prob", format="%.4f"),
            'pos_rate': st.column_config.NumberColumn("BBB+ Rate", format="%.2f")
        },
        use_container_width=True,
        hide_index=True
    )

# 下载按钮
csv = df_display.to_csv(index=False).encode('utf-8-sig')
st.download_button(
    "Download Filtered Data",
    csv,
    f"smarts_analysis_{filter_type[:20]}.csv",
    "mime=text/csv"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**说明**")
st.sidebar.caption("""
- **Positive SMARTS**: Chemical substructures that promote BBB permeability
- **Negative SMARTS**: Substructures that hinder BBB permeability
- **Δ Probability**: Difference in BBB+ probability when this SMARTS is present
- **Frequency**: How often this SMARTS appears in the dataset
""")
