"""
模型选择与对比页面 - 完整32模型版本（包含SMARTS增强模型和GAT no-pretrain模型）
支持自定义筛选和选择
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go

# 设置页面配置
st.set_page_config(
    page_title="Model Comparison - 32 Models",
    page_icon="📊",
    layout="wide"
)

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 页面标题
st.title("📊 32 Model Performance Comparison")
st.markdown("**Including SMARTS-Enhanced and GAT Models**")
st.markdown("---")

# 加载模型性能数据
@st.cache_data
def load_model_performance():
    """Load performance data for all 32 models (baseline + SMARTS-enhanced + GAT)"""
    # 加载原始16个模型
    perf_file = PROJECT_ROOT / "outputs" / "all_16_models_performance.csv"
    if perf_file.exists():
        original_df = pd.read_csv(perf_file)
    else:
        st.error("原始性能数据文件不存在��请先运行模型评估。")
        return None

    # 加载SMARTS增强模型
    extended_models = []
    for dataset in ['A', 'A_B', 'A_B_C', 'A_B_C_D']:
        ext_file = PROJECT_ROOT / "outputs" / f"extended_models_{dataset}_seed0.csv"
        if ext_file.exists():
            ext_df = pd.read_csv(ext_file)
            # 添加缺失的列（如果存在）
            if 'TN' not in ext_df.columns:
                ext_df['TN'] = -1
            if 'FN' not in ext_df.columns:
                ext_df['FN'] = -1
            extended_models.append(ext_df)

    if not extended_models:
        st.error("SMARTS增强模型数据不存在！请先运行SMARTS模型训练。")
        return original_df

    extended_df = pd.concat(extended_models, ignore_index=True)

    # 合并两个数据集
    df = pd.concat([original_df, extended_df], ignore_index=True)

    # 加载GAT模型
    gat_no_pretrain_models = []
    for dataset in ['A', 'A_B', 'A_B_C', 'A_B_C_D']:
        gat_np_file = PROJECT_ROOT / "outputs" / f"gat_no_pretrain_{dataset}_seed0.csv"
        if gat_np_file.exists():
            gat_np_df = pd.read_csv(gat_np_file)
            # 添加缺失的列（如果存在）
            if 'TN' not in gat_np_df.columns:
                gat_np_df['TN'] = -1
            if 'FN' not in gat_np_df.columns:
                gat_np_df['FN'] = -1
            gat_no_pretrain_models.append(gat_np_df)

    if gat_no_pretrain_models:
        gat_no_pretrain_df = pd.concat(gat_no_pretrain_models, ignore_index=True)
        df = pd.concat([df, gat_no_pretrain_df], ignore_index=True)

    return df

# 数据集名称映射
dataset_names = {
    'A': 'A',
    'A_B': 'A,B',
    'A_B_C': 'A,B,C',
    'A_B_C_D': 'A,B,C,D'
}

# 加载数据
df = load_model_performance()

if df is None:
    st.stop()

# 添加数据集显示名称
df['Dataset_Display'] = df['Dataset'].map(dataset_names)

# 侧边栏 - 高级筛选
st.sidebar.header("🔍 筛选选项")

# 数据集筛选
st.sidebar.subheader("选择数据集")
dataset_options = df['Dataset_Display'].unique().tolist()
selected_datasets = st.sidebar.multiselect(
    "数据集",
    options=dataset_options,
    default=dataset_options
)

# 模型类型筛选
st.sidebar.subheader("选择模型类型")
model_options = df['Model'].unique().tolist()
selected_models = st.sidebar.multiselect(
    "模型",
    options=model_options,
    default=model_options
)

# 高级筛选
st.sidebar.subheader("🎯 高级筛选")

# AUC范围
auc_range = st.sidebar.slider(
    "AUC范围",
    min_value=float(df['AUC'].min()),
    max_value=float(df['AUC'].max()),
    value=(float(df['AUC'].min()), float(df['AUC'].max())),
    step=0.01
)

# Precision范围
precision_range = st.sidebar.slider(
    "Precision范围",
    min_value=float(df['Precision'].min()),
    max_value=float(df['Precision'].max()),
    value=(float(df['Precision'].min()), float(df['Precision'].max())),
    step=0.01
)

# 应用筛选
filtered_df = df[
    (df['Dataset_Display'].isin(selected_datasets)) &
    (df['Model'].isin(selected_models)) &
    (df['AUC'] >= auc_range[0]) & (df['AUC'] <= auc_range[1]) &
    (df['Precision'] >= precision_range[0]) & (df['Precision'] <= precision_range[1])
]

if len(filtered_df) == 0:
    st.warning("没有符合条件的模型，请调整筛选条件")
    st.stop()

# 显示筛选结果数量
st.info(f"显示 {len(filtered_df)} 个模型（共32个）")

# 主要内容区域
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 按数据集对比")

    for dataset_display in selected_datasets:
        dataset_df = filtered_df[filtered_df['Dataset_Display'] == dataset_display]

        if len(dataset_df) == 0:
            continue

        st.markdown(f"### {dataset_display}")

        # 创建对比表格
        display_df = dataset_df[['Model', 'AUC', 'Precision', 'Recall', 'F1', 'FP', 'TP']].copy()
        display_df['AUC'] = display_df['AUC'].apply(lambda x: f"{x:.4f}")
        display_df['Precision'] = display_df['Precision'].apply(lambda x: f"{x:.4f}")
        display_df['Recall'] = display_df['Recall'].apply(lambda x: f"{x:.4f}")
        display_df['F1'] = display_df['F1'].apply(lambda x: f"{x:.4f}")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

with col2:
    st.subheader("🏆 筛选后的最佳模型")

    if len(filtered_df) > 0:
        # 计算各种最佳指标
        best_auc = filtered_df.loc[filtered_df['AUC'].idxmax()]
        best_precision = filtered_df.loc[filtered_df['Precision'].idxmax()]
        best_f1 = filtered_df.loc[filtered_df['F1'].idxmax()]
        best_recall = filtered_df.loc[filtered_df['Recall'].idxmax()]

        st.metric(
            "最佳 AUC",
            f"{best_auc['AUC']:.4f}",
            f"{best_auc['Model']} - {best_auc['Dataset_Display']}"
        )

        st.metric(
            "最佳 Precision",
            f"{best_precision['Precision']:.4f}",
            f"{best_precision['Model']} - {best_precision['Dataset_Display']}"
        )

        st.metric(
            "最佳 F1",
            f"{best_f1['F1']:.4f}",
            f"{best_f1['Model']} - {best_f1['Dataset_Display']}"
        )

        st.metric(
            "最佳 Recall",
            f"{best_recall['Recall']:.4f}",
            f"{best_recall['Model']} - {best_recall['Dataset_Display']}"
        )

# 可视化
st.markdown("---")
st.subheader("📊 性能可视化")

# 选择要绘制的指标
metric = st.selectbox(
    "选择指标",
    options=['AUC', 'Precision', 'Recall', 'F1'],
    index=0
)

# 创建柱状图
fig = px.bar(
    filtered_df,
    x='Dataset_Display',
    y=metric,
    color='Model',
    barmode='group',
    title=f'{metric} 对比（筛选后的模型）',
    labels={'Dataset_Display': '数据集', 'value': metric},
    height=500
)

fig.update_layout(
    xaxis_title="数据集",
    yaxis_title=metric,
    legend_title="模型",
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# 散点图 - AUC vs Recall
st.subheader("🎯 AUC vs Recall (性能 vs 召回率)")

fig_scatter = px.scatter(
    filtered_df,
    x='Recall',
    y='AUC',
    color='Model',
    symbol='Dataset_Display',
    size='F1',
    hover_data=['Precision', 'F1'],
    title="AUC vs Recall - 气泡大小表示F1分数",
    labels={'Recall': 'Recall', 'AUC': 'AUC'},
    height=500
)

fig_scatter.update_layout(
    xaxis_title="Recall (越高越好)",
    yaxis_title="AUC (越高越好)",
    legend_title="模型",
    hovermode='closest'
)

# 添加理想区域标注
fig_scatter.add_vrect(
    x0=0.9, x1=1.0,
    fillcolor="green", opacity=0.1,
    annotation_text="高Recall区域"
)

fig_scatter.add_hrect(
    y0=0.95, y1=1.0,
    fillcolor="blue", opacity=0.1,
    annotation_text="高AUC区域"
)

st.plotly_chart(fig_scatter, use_container_width=True)

# 热力图 - 所有指标
st.subheader("🌡️ 性能热力图")

# 准备热力图数据
heatmap_df = filtered_df.pivot_table(
    index='Model',
    columns='Dataset_Display',
    values='AUC',
    aggfunc='first'
)

fig_heatmap = px.imshow(
    heatmap_df,
    labels=dict(x="数据集", y="模型", color="AUC"),
    x=heatmap_df.columns,
    y=heatmap_df.index,
    color_continuous_scale='RdYlGn',
    title="AUC 热力图（筛选后的模型）",
    height=400
)

fig_heatmap.update_layout(
    xaxis_title="数据集",
    yaxis_title="模型"
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# 完整数据表
st.markdown("---")
st.subheader("📋 筛选后的完整数据")

st.dataframe(
    filtered_df[['Dataset_Display', 'Model', 'AUC', 'Precision', 'Recall', 'F1']],
    use_container_width=True,
    hide_index=True
)

# 模型推荐
st.markdown("---")
st.subheader("💡 根据筛选结果推荐")

if len(filtered_df) >= 3:
    # 从筛选结果中推荐
    col1, col2, col3 = st.columns(3)

    with col1:
        best_overall = filtered_df.loc[filtered_df['AUC'].idxmax()]
        st.markdown(f"""
        ### 🥇 综合最佳

        **推荐**: {best_overall['Model']} - {best_overall['Dataset_Display']}

        - AUC: {best_overall['AUC']:.4f}
        - Precision: {best_overall['Precision']:.4f}
        - Recall: {best_overall['Recall']:.4f}

        适用场景: 综合性能最佳
        """)

    with col2:
        best_prec = filtered_df.loc[filtered_df['Precision'].idxmax()]
        st.markdown(f"""
        ### 🥈 高Precision

        **推荐**: {best_prec['Model']} - {best_prec['Dataset_Display']}

        - Precision: {best_prec['Precision']:.4f}
        - AUC: {best_prec['AUC']:.4f}
        - Recall: {best_prec['Recall']:.4f}

        适用场景: 减少假阳性
        """)

    with col3:
        lowest_fp = filtered_df.loc[filtered_df['FP'].idxmin()]
        st.markdown(f"""
        ### 🥉 低FP

        **推荐**: {lowest_fp['Model']} - {lowest_fp['Dataset_Display']}

        - Recall: {best_recall['Recall']:.4f}
        - AUC: {best_recall['AUC']:.4f}
        - Precision: {best_recall['Precision']:.4f}

        适用场景: 高召回率
        """)

# 下载按钮
st.markdown("---")
st.subheader("📥 下载数据")

csv = filtered_df.to_csv(index=False)
st.download_button(
    label="下载筛选后的数据 (CSV)",
    data=csv,
    file_name="filtered_model_performance.csv",
    mime="text/csv"
)

# 显示所有32个模型的概览
with st.expander("🔍 查看全部32个模型"):
    st.subheader("全部32个模型列表")

    # 按数据集分组显示
    for dataset_display in dataset_names.values():
        all_dataset_df = df[df['Dataset_Display'] == dataset_display]

        st.markdown(f"### {dataset_display}")

        display_all_df = all_dataset_df[['Model', 'AUC', 'Precision', 'Recall', 'F1']].copy()
        display_all_df['AUC'] = display_all_df['AUC'].apply(lambda x: f"{x:.4f}")
        display_all_df['Precision'] = display_all_df['Precision'].apply(lambda x: f"{x:.4f}")
        display_all_df['Recall'] = display_all_df['Recall'].apply(lambda x: f"{x:.4f}")
        display_all_df['F1'] = display_all_df['F1'].apply(lambda x: f"{x:.4f}")

        st.dataframe(
            display_all_df,
            use_container_width=True,
            hide_index=True
        )
