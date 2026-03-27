"""
BBB分子生成平台 - Streamlit Web应用

功能：
1. 使用VAE或GAN生成新的BBB穿透小分子
2. 使用BBB预测模型过滤生成的分子
3. 显示分子结构和性质
4. 导出生成的分子

运行：
    streamlit run app_bbb_predict.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time

import streamlit as st
from streamlit.components.v1 import components

st.set_page_config(
    page_title="BBB分子生成平台",
    page_icon="🧬",
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
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏
st.sidebar.title("🧬 BBB分子生成")

# 配置
SEED = 0

st.sidebar.markdown("---")
st.sidebar.markdown("## 生成设置")

# 生成策略
strategy = st.sidebar.selectbox(
    "生成策略",
    options=["VAE", "GAN", "VAE + GAN"],
    index=2,
    help="VAE: 变分自编码器; GAN: 生成对抗网络; VAE+GAN: 两者结合"
)

# 分子数量
n_generate = st.sidebar.slider(
    "生成数量",
    min_value=10,
    max_value=1000,
    value=100,
    step=10,
    help="要生成的分子数量"
)

# 过滤阈值
st.sidebar.markdown("---")
st.sidebar.markdown("## 过滤设置")

min_qed = st.sidebar.slider(
    "最小QED分数",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="药物相似性阈值 (0-1, 越高越严格)"
)

min_bbb = st.sidebar.slider(
    "最小BBB概率",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="BBB渗透概率阈值"
)

max_sa = st.sidebar.slider(
    "最大SA分数",
    min_value=1.0,
    max_value=10.0,
    value=4.0,
    step=0.5,
    help="合成可及性阈值 (1-10, 越低越容易合成)"
)

check_novelty = st.sidebar.checkbox(
    "检查新颖性",
    value=True,
    help="过滤掉训练集中已有的分子"
)

# 主界面
st.title("🧬 BBB分子生成平台")
st.markdown("使用深度学习生成新的BBB穿透小分子")

# 介绍
with st.expander("ℹ️ 关于分子生成", expanded=False):
    st.markdown("""
    ### 生成方法

    本平台提供两种分子生成方法：

    1. **VAE (变分自编码器)**
       - 学习分子在潜在空间中的分布
       - 通过采样生成新分子
       - 优点: 稳定性好, 生成的分子多样性高

    2. **GAN (生成对抗网络)**
       - 生成器与判别器对抗训练
       - 使用强化学习优化BBB渗透性
       - 优点: 可以直接优化目标性质

    ### 过滤标准

    - **QED**: 定量药物相似性 (0-1), 越高越像药物
    - **BBB概率**: 模型预测的BBB渗透概率
    - **SA分数**: 合成可及性 (1-10), 越低越容易合成
    """)

# 生成按钮
col1, col2 = st.columns([1, 4])

with col1:
    generate_btn = st.button(
        "🚀 生成分子",
        type="primary",
        use_container_width=True,
    )

with col2:
    st.info(f"设置: {strategy} | 生成 {n_generate} 个分子 | QED≥{min_qed} | BBB≥{min_bbb} | SA≤{max_sa}")

# 生成结果存储
if 'generated_molecules' not in st.session_state:
    st.session_state.generated_molecules = []

if 'generation_metrics' not in st.session_state:
    st.session_state.generation_metrics = {}

# 执行生成
if generate_btn:
    with st.spinner("正在生成分子..."):
        try:
            # 导入生成模块
            from src.generation import generate_molecules, GenerationPipeline
            from src.multi_model_predictor import MultiModelPredictor, EnsembleStrategy

            # 加载BBB预测器
            bbb_predictor = MultiModelPredictor(
                seed=SEED,
                strategy=EnsembleStrategy.SOFT_VOTING,
            )

            # 生成策略映射
            strategy_map = {
                "VAE": "vae",
                "GAN": "gan",
                "VAE + GAN": "both"
            }

            # 使用简化的生成（不使用预训练模型，直接使用BBB预测器过滤）
            # 在实际使用中，需要先训练VAE/GAN模型

            # 模拟生成（因为没有预训练模型）
            st.warning("⚠️ 注意: 目前没有预训练的生成模型，使用示例分子演示")

            # 示例BBB+分子（来自B3DB）
            example_molecules = [
                "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
                "CC(C)NCC(COC1=CC=CC2=C1CCCC2)O",  // Propranolol
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  // Caffeine
                "CC(C)NCC(O)CO",  // Propranolol fragment
                "CC(=O)Oc1ccccc1C(=O)O",  // Aspirin
                "Cc1ccc(cc1)S(=O)(=O)N",  // Toluenesulfonamide
                "CC(C)NCCc1ccccc1",  // Amphetamine-like
                "Cc1ccccc1N",  // Toluidine
                "CC(=O)NCC1=CC=CC=C1",  // Phenylacetamide
                "CCCN",  // Propylamine
            ]

            # 随机选择
            np.random.seed(SEED)
            selected = np.random.choice(example_molecules, min(n_generate, len(example_molecules)), replace=True).tolist()

            # 预测BBB概率
            if bbb_predictor:
                results = bbb_predictor.predict(selected)
                bbb_probs = results.ensemble_probability
            else:
                bbb_probs = [0.5] * len(selected)

            # 计算性质
            from rdkit import Chem
            from rdkit.Chem import QED

            molecules_data = []
            for i, smi in enumerate(selected):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue

                try:
                    qed = QED.qed(mol)
                except:
                    qed = 0.0

                # 简化的SA分数
                from src.vae.molecule_vae import compute_sa_score
                sa = compute_sa_score(smi)

                molecules_data.append({
                    'SMILES': smi,
                    'BBB_prob': bbb_probs[i],
                    'QED': qed,
                    'SA': sa,
                    'passes': qed >= min_qed and bbb_probs[i] >= min_bbb and sa <= max_sa
                })

            df = pd.DataFrame(molecules_data)

            # 过滤
            filtered = df[df['passes']]

            # 保存结果
            st.session_state.generated_molecules = filtered['SMILES'].tolist() if len(filtered) > 0 else selected[:10]
            st.session_state.generation_metrics = {
                'n_generated': len(selected),
                'n_filtered': len(filtered),
                'filter_rate': len(filtered) / len(selected) if len(selected) > 0 else 0,
                'avg_bbb': filtered['BBB_prob'].mean() if len(filtered) > 0 else 0,
                'avg_qed': filtered['QED'].mean() if len(filtered) > 0 else 0,
                'avg_sa': filtered['SA'].mean() if len(filtered) > 0 else 0,
            }

            st.success(f"✅ 生成完成! 生成了 {len(selected)} 个分子, 其中 {len(filtered)} 个通过过滤")

        except Exception as e:
            st.error(f"生成失败: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

# 显示结果
if st.session_state.generated_molecules:
    st.markdown("---")
    st.subheader("📊 生成结果")

    # 指标
    metrics = st.session_state.generation_metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("生成数量", metrics.get('n_generated', 0))
    with col2:
        st.metric("过滤后数量", metrics.get('n_filtered', 0))
    with col3:
        st.metric("过滤率", f"{metrics.get('filter_rate', 0)*100:.1f}%")
    with col4:
        st.metric("平均BBB概率", f"{metrics.get('avg_bbb', 0):.2f}")

    # 显示分子
    st.markdown("### 🧪 生成的分子")

    # 创建分子数据
    display_data = []
    for i, smi in enumerate(st.session_state.generated_molecules[:20]):  # 显示前20个
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        try:
            qed = QED.qed(mol)
        except:
            qed = 0.0

        from src.vae.molecule_vae import compute_sa_score
        sa = compute_sa_score(smi)

        display_data.append({
            'ID': i + 1,
            'SMILES': smi,
            'QED': f"{qed:.2f}",
            'SA': f"{sa:.1f}",
        })

    if display_data:
        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # 分子可视化
        st.markdown("### 🧬 分子结构")

        # 选择分子
        selected_idx = st.selectbox(
            "选择分子查看结构",
            options=range(len(st.session_state.generated_molecules)),
            format_func=lambda x: f"分子 {x+1}: {st.session_state.generated_molecules[x][:30]}..."
        )

        if selected_idx is not None:
            smiles = st.session_state.generated_molecules[selected_idx]

            # 使用RDKit绘制分子
            from rdkit.Chem import Draw
            from rdkit.Chem.Draw import rdMolDraw2D

            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # 绘制
                drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()

                # 转换为streamlit显示
                import io
                img_bytes = drawer.GetDrawingText()
                st.image(img_bytes, caption=f"SMILES: {smiles}", use_container_width=False)

                # 显示性质
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("QED分数", f"{qed:.2f}")
                with col2:
                    st.metric("SA分数", f"{sa:.1f}")
                with col3:
                    st.metric("BBB概率", "需要预测")

    # 导出
    st.markdown("---")
    st.subheader("💾 导出")

    col1, col2 = st.columns(2)

    with col1:
        # CSV导出
        if st.session_state.generated_molecules:
            export_df = pd.DataFrame({
                'SMILES': st.session_state.generated_molecules,
            })
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 下载SMILES (CSV)",
                data=csv,
                file_name="generated_molecules.csv",
                mime="text/csv",
            )

    with col2:
        # 显示原始SMILES
        if st.session_state.generated_molecules:
            st.text_area(
                "SMILES列表",
                value="\n".join(st.session_state.generated_molecules[:50]),
                height=150,
            )

else:
    # 欢迎界面
    st.markdown("""
    ### 欢迎使用BBB分子生成平台!

    点击左侧的 **"🚀 生成分子"** 按钮开始生成新的BBB穿透小分子。

    #### 生成流程

    1. **选择生成策略**: VAE、GAN或两者结合
    2. **设置生成参数**: 分子数量和过滤阈值
    3. **生成分子**: 系统将生成并过滤分子
    4. **查看结果**: 浏览生成的分子结构和性质
    5. **导出结果**: 下载SMILES列表用于进一步研究

    #### 过滤标准说明

    | 指标 | 说明 | 阈值建议 |
    |------|------|----------|
    | QED | 药物相似性 | ≥ 0.5 |
    | BBB概率 | 模型预测的渗透性 | ≥ 0.7 |
    | SA | 合成难度 | ≤ 4.0 |
    """)

    # 显示示例分子
    st.markdown("### 示例BBB+分子")

    example_molecules = [
        ("咖啡因", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("普萘洛尔", "CC(C)NCC(COC1=CC=CC2=C1CCCC2)O"),
        ("阿司匹林", "CC(=O)Oc1ccccc1C(=O)O"),
    ]

    for name, smiles in example_molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            from rdkit.Chem.Draw import rdMolDraw2D
            drawer = rdMolDraw2D.MolDraw2DCairo(200, 150)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            img_bytes = drawer.GetDrawingText()
            st.image(img_bytes, caption=f"{name}: {smiles}", width=200)

# 导入RDKit（在需要时）
try:
    from rdkit import Chem
except ImportError:
    st.warning("RDKit未安装，部分功能可能不可用")
