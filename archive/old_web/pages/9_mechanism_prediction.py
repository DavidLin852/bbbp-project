"""
BBB Transport Mechanism Prediction Page

This page allows users to:
1. Input a molecule (SMILES or draw)
2. Predict its BBB permeability
3. Identify the dominant transport mechanism
4. Get recommendations for optimization

Author: BBB Prediction Project
Reference: Cornelissen et al., J. Med. Chem. 2022
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="BBB Mechanism Prediction",
    page_icon="🧬",
    layout="wide"
)

# Title and description
st.title("🧬 BBB Transport Mechanism Prediction")
st.markdown("""
This tool predicts how a molecule crosses the blood-brain barrier (BBB) and identifies
the dominant transport mechanism.

**Mechanisms:**
- **Passive Diffusion**: Molecule crosses BBB via transcellular diffusion
- **Active Influx**: Molecule is transported by SLC influx transporters
- **Active Efflux**: Molecule is pumped out by ABC efflux transporters (e.g., P-gp)
- **Mixed**: Multiple mechanisms contribute

Based on: *Cornelissen et al., J. Med. Chem. 2022, 65, 11, 8340–8360*
""")

# Sidebar
st.sidebar.header("Input Molecule")

# Input options
input_method = st.sidebar.radio(
    "Choose input method:",
    ["SMILES", "Draw Structure", "Example Molecules"]
)

# Example molecules
EXAMPLES = {
    "Aspirin (Passive)": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine (Passive)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Dopamine (Influx)": "NCCc1cc(O)c(O)cc1",
    "Morphine (Mixed)": "CN1CCC[C@H]1C(=O)OC(C)(C)C(=O)OC1=Cc2ccccc12",
    "Penicillin G (Efflux risk)": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
}

if input_method == "SMILES":
    smiles = st.sidebar.text_input(
        "Enter SMILES:",
        value="CC(=O)OC1=CC=CC=C1C(=O)O",
        help="Enter SMILES notation of the molecule"
    )

elif input_method == "Example Molecules":
    example_name = st.sidebar.selectbox("Select an example:", list(EXAMPLES.keys()))
    smiles = EXAMPLES[example_name]
    st.sidebar.info(f"SMILES: {smiles}")

elif input_method == "Draw Structure":
    st.sidebar.warning("Drawing tool requires additional setup. Using default example.")
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

# Predict button
predict_btn = st.sidebar.button("🔬 Predict Mechanism", type="primary")

# Load models
@st.cache_resource
def load_models():
    """Load trained mechanism prediction models."""
    try:
        import xgboost as xgb
        import joblib

        models_dir = project_root / "artifacts" / "models" / "mechanism"

        models = {}

        # Load BBB model
        bbb_model_path = models_dir / "bbb_model.json"
        if bbb_model_path.exists():
            models['bbb'] = xgb.XGBClassifier()
            models['bbb'].load_model(str(bbb_model_path))

        # Load mechanism models
        for mech in ['passive', 'influx']:
            mech_model_path = models_dir / f"{mech}_model.json"
            if mech_model_path.exists():
                models[mech] = xgb.XGBClassifier()
                models[mech].load_model(str(mech_model_path))

        # Load imputer
        imputer_path = models_dir / "imputer.joblib"
        if imputer_path.exists():
            models['imputer'] = joblib.load(imputer_path)

        return models if models else None

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None

# Extract features
def extract_features(smiles: str):
    """Extract features for prediction."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, MACCSkeys, DataStructs

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Physicochemical features (7)
        tpsa = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotbonds = Descriptors.NumRotatableBonds(mol)

        logd = logp
        if hbd > 0:
            logd -= 0.5 * hbd

        physicochemical = [tpsa, mw, logp, logd, hbd, hba, rotbonds]

        # MACCS keys (167)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_array = np.zeros((167,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(maccs, maccs_array)

        # Combine
        combined = np.concatenate([physicochemical, maccs_array])

        return combined, mol

    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None, None

# Main prediction logic
if predict_btn and smiles:
    # Load models
    models = load_models()

    if models is None:
        st.error("❌ Models not found. Please train models first.")
        st.info("Run: `python scripts/mechanism_training/train_robust.py`")
    else:
        # Extract features
        features, mol = extract_features(smiles)

        if features is None:
            st.error("❌ Invalid SMILES. Please check the input.")
        else:
            # Impute
            features_imp = models['imputer'].transform(features.reshape(1, -1))

            # Predict BBB
            bbb_proba = models['bbb'].predict_proba(features_imp)[0, 1]
            bbb_pred = int(bbb_proba >= 0.5)

            # Predict mechanisms
            mech_predictions = {}
            for mech_name, mech_model in models.items():
                if mech_name in ['bbb', 'imputer']:
                    continue

                try:
                    mech_proba = mech_model.predict_proba(features_imp)[0, 1]
                    mech_predictions[mech_name] = mech_proba
                except:
                    mech_predictions[mech_name] = 0.0

            # Determine dominant mechanism
            if bbb_pred == 1:
                # BBB+
                max_mech = max(mech_predictions.items(), key=lambda x: x[1])
                dominant_mech = max_mech[0]
                confidence = max_mech[1]
            else:
                # BBB-
                dominant_mech = "impermeable"
                confidence = 1 - bbb_proba

            # Display results
            st.header("Prediction Results")

            # BBB Permeability
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("BBB Permeability")
                if bbb_pred == 1:
                    st.success("✅ BBB+ (Permeable)")
                else:
                    st.error("❌ BBB- (Not Permeable)")
                st.metric("Confidence", f"{bbb_proba:.1%}")

            with col2:
                st.subheader("Dominant Mechanism")
                st.info(f"**{dominant_mech.upper()}**")
                st.metric("Confidence", f"{confidence:.1%}")

            with col3:
                st.subheader("Molecule Properties")
                try:
                    from rdkit.Chem import Descriptors
                    mw = Descriptors.MolWt(mol)
                    tpsa = Descriptors.TPSA(mol)
                    logp = Descriptors.MolLogP(mol)

                    st.metric("MW", f"{mw:.1f}")
                    st.metric("TPSA", f"{tpsa:.1f} Ų")
                    st.metric("LogP", f"{logp:.2f}")
                except:
                    pass

            # Detailed mechanism breakdown
            st.subheader("Mechanism Probabilities")

            mech_data = []
            for mech_name, prob in mech_predictions.items():
                mech_data.append({
                    "Mechanism": mech_name.upper(),
                    "Probability": f"{prob:.1%}",
                    "Confidence": prob
                })

            df_mech = pd.DataFrame(mech_data)
            st.dataframe(df_mech, use_container_width=True, hide_index=True)

            # Bar chart
            st.subheader("Mechanism Probability Distribution")
            import plotly.graph_objects as go

            fig = go.Figure(data=[
                go.Bar(
                    x=[m["Mechanism"] for m in mech_data],
                    y=[m["Confidence"] for m in mech_data],
                    marker_color=['#2ecc71' if m["Mechanism"] == dominant_mech.upper() else '#95a5a6'
                               for m in mech_data]
                )
            ])
            fig.update_layout(
                yaxis_title="Probability",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Recommendations
            st.subheader("💡 Recommendations")

            if dominant_mech == "passive":
                st.success("""
                **Good passive diffusion properties!**

                ✅ Low TPSA, moderate LogP favor BBB permeability
                ✅ Low efflux risk
                ✅ Optimal for CNS drugs

                **Maintain** current properties:
                - TPSA < 90 Ų
                - LogP 1-3
                - MW < 500 Da
                """)
            elif dominant_mech == "influx":
                st.info("""
                **Likely to use active transport!**

                ✅ May utilize SLC transporters for brain uptake
                ⚠️ Transporter expression varies between individuals
                ⚠️ Consider potential drug-drug interactions

                **Optimization tips:**
                - Consider prodrug strategies to enhance passive diffusion
                - Evaluate transporter expression in target tissue
                """)
            else:
                st.warning("""
                **Poor BBB permeability!**

                ❌ High TPSA, high MW, or unfavorable properties
                ❌ Consider structural modifications

                **Optimization suggestions:**
                - Reduce TPSA (remove polar groups)
                - Reduce MW (consider fragment-based design)
                - Optimize LogP (aim for 1-3)
                - Reduce rotatable bonds
                - Consider prodrug approach
                """)

            # Molecule visualization
            st.subheader("Molecule Structure")
            try:
                from rdkit.Chem import Draw
                img = Draw.MolToImage(mol, size=(400, 200))
                st.image(img)
            except:
                st.info("Molecule visualization not available")

else:
    # Show instructions
    st.info("👈 Enter a SMILES or select an example molecule, then click **Predict Mechanism**")

    # Show example molecules table
    st.subheader("Example Molecules")

    example_data = []
    for name, smi in EXAMPLES.items():
        example_data.append({
            "Name": name,
            "SMILES": smi
        })

    df_examples = pd.DataFrame(example_data)
    st.dataframe(df_examples, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
**References:**
1. Cornelissen et al., "Explaining Blood–Brain Barrier Permeability of Small Molecules by
   Integrated Analysis of Different Transport Mechanisms", *J. Med. Chem.* **2022**, 65, 11, 8340–8360
   DOI: 10.1021/acs.jmedchem.2c01824

2. B3DB Database: https://github.com/theochem/B3DB

**Methodology:**
- Trained on 7,805 compounds from B3DB
- XGBoost models with physicochemical + MACCS features
- Synthetic mechanism labels based on property rules
""")
