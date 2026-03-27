"""
Dimensionality Reduction Visualization - Interactive exploration
"""
import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.sparse import load_npz
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle
import json
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from io import BytesIO

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.config import Paths, DatasetConfig

# Page config
st.set_page_config(
    page_title="Dimensionality Reduction",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    h1, h2, h3 {
        font-family: 'Times New Roman', serif;
    }
    .info-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Dimensionality Reduction Visualization")
st.markdown("---")

# Sidebar
st.sidebar.header("Settings")

# Method selection
method = st.sidebar.selectbox(
    "Dimensionality Reduction Method",
    ["PCA", "t-SNE", "LDA"],
    help="Choose the dimensionality reduction method to visualize"
)

# Feature type selection
feature_type = st.sidebar.selectbox(
    "Feature Type",
    ["Morgan", "MACCS", "AtomPairs", "FP2", "Combined"],
    help="Choose which molecular features to use"
)

# Dataset split selection
split = st.sidebar.selectbox(
    "Dataset Split",
    ["test", "train", "val", "all"],
    help="Choose which dataset to display"
)

# Load data cache
@st.cache_resource
def load_background_data():
    """Load and cache background data and models."""
    from scipy import sparse

    feature_dir = project_root / "artifacts" / "features" / "seed_0_enhanced"

    # Load metadata
    meta = pd.read_csv(feature_dir / "meta.csv")

    # Load Morgan fingerprints
    X_morgan_data = np.load(feature_dir / "morgan.npz")
    if 'format' in X_morgan_data.files:
        X_morgan = sparse.csr_matrix((X_morgan_data['data'], X_morgan_data['indices'],
                                      X_morgan_data['indptr']), shape=X_morgan_data['shape'])
    else:
        X_morgan = X_morgan_data['X'] if 'X' in X_morgan_data.files else X_morgan_data[X_morgan_data.files[0]]
    X_morgan_dense = X_morgan.toarray() if sparse.issparse(X_morgan) else X_morgan

    # Load MACCS
    X_maccs_data = np.load(feature_dir / "maccs.npz")
    if 'format' in X_maccs_data.files:
        X_maccs = sparse.csr_matrix((X_maccs_data['data'], X_maccs_data['indices'],
                                    X_maccs_data['indptr']), shape=X_maccs_data['shape'])
    else:
        X_maccs = X_maccs_data['X'] if 'X' in X_maccs_data.files else X_maccs_data[X_maccs_data.files[0]]
    X_maccs_dense = X_maccs.toarray() if sparse.issparse(X_maccs) else X_maccs

    # Load AtomPairs
    X_atompairs_data = np.load(feature_dir / "atompairs.npz")
    if 'format' in X_atompairs_data.files:
        X_atompairs = sparse.csr_matrix((X_atompairs_data['data'], X_atompairs_data['indices'],
                                         X_atompairs_data['indptr']), shape=X_atompairs_data['shape'])
    else:
        X_atompairs = X_atompairs_data['X'] if 'X' in X_atompairs_data.files else X_atompairs_data[X_atompairs_data.files[0]]
    X_atompairs_dense = X_atompairs.toarray() if sparse.issparse(X_atompairs) else X_atompairs

    # Load FP2
    X_fp2_data = np.load(feature_dir / "fp2.npz")
    if 'format' in X_fp2_data.files:
        X_fp2 = sparse.csr_matrix((X_fp2_data['data'], X_fp2_data['indices'],
                                   X_fp2_data['indptr']), shape=X_fp2_data['shape'])
    else:
        X_fp2 = X_fp2_data['X'] if 'X' in X_fp2_data.files else X_fp2_data[X_fp2_data.files[0]]
    X_fp2_dense = X_fp2.toarray() if sparse.issparse(X_fp2) else X_fp2

    # Load RDKit Descriptors
    X_desc = pd.read_csv(feature_dir / "descriptors.csv", index_col=0).values

    # Load Combined Features
    X_combined_data = np.load(feature_dir / "combined_all.npz")
    if 'format' in X_combined_data.files:
        X_combined = sparse.csr_matrix((X_combined_data['data'], X_combined_data['indices'],
                                        X_combined_data['indptr']), shape=X_combined_data['shape'])
    else:
        X_combined = X_combined_data['X'] if 'X' in X_combined_data.files else X_combined_data[X_combined_data.files[0]]
    X_combined_dense = X_combined.toarray() if sparse.issparse(X_combined) else X_combined

    return {
        'meta': meta,
        'X_morgan': X_morgan_dense,
        'X_maccs': X_maccs_dense,
        'X_atompairs': X_atompairs_dense,
        'X_fp2': X_fp2_dense,
        'X_combined': X_combined_dense  # 仅包含4种特征: Morgan+MACCS+AtomPairs+FP2
    }


@st.cache_resource
def fit_reduction_models(_data_dict):
    """Fit and cache dimensionality reduction models."""
    models = {}

    # Prepare features
    X_morgan = _data_dict['X_morgan']
    X_maccs = _data_dict['X_maccs']
    X_atompairs = _data_dict['X_atompairs']
    X_fp2 = _data_dict['X_fp2']
    X_desc = _data_dict['X_desc']
    X_combined = _data_dict['X_combined']
    y = _data_dict['meta']['y_cls'].values  # Get labels for LDA

    features_dict = {
        'Morgan': X_morgan,
        'MACCS': X_maccs,
        'AtomPairs': X_atompairs,
        'FP2': X_fp2,
        'Combined': X_combined  # 仅4种特征: Morgan+MACCS+AtomPairs+FP2 = 5287维
    }

    # Fit models
    for feat_name, X in features_dict.items():
        # PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        models[f'PCA_{feat_name}'] = {
            'model': pca,
            'transformed': X_pca
        }

        # t-SNE (faster with lower perplexity and iterations for interactivity)
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            max_iter=500,
            random_state=42,
            verbose=0
        )
        X_tsne = tsne.fit_transform(X)
        models[f't-SNE_{feat_name}'] = {
            'model': tsne,
            'transformed': X_tsne
        }

        # LDA (Linear Discriminant Analysis)
        # For binary classification, LDA can only reduce to 1 dimension
        # We'll use LDA for 1st dimension and PCA for 2nd dimension
        lda = LDA(n_components=1)
        X_lda_1d = lda.fit_transform(X, y)

        # Combine LDA (1st dim) with PCA (2nd dim) for 2D visualization
        X_lda_2d = np.column_stack([X_lda_1d.flatten(), X_pca[:, 1]])

        models[f'LDA_{feat_name}'] = {
            'model': lda,
            'pca_ref': pca,  # Store PCA reference for transforming new data
            'transformed': X_lda_2d
        }

    return models


def compute_single_smiles_features(smiles):
    """Compute features for a single SMILES - supports 5 feature types (no RDKitDesc)."""
    from rdkit.Chem import MACCSkeys, rdMolDescriptors

    # SMILES auto-fix mapping for known invalid formats
    smiles_fixes = {
        'CC(C)(C)C1CCCc2ccccc2Cl': 'CC(C)C1CCCC1c2ccccc2Cl',  # MPC molecule fix
        'C=C(C)C=O)OCC[N+](C)(C)CCCS(=O)(=O)[O-]': 'C=C(C)C(=O)OCC[N+](C)(C)CCCS(=O)(=O)[O-]',  # SBMA fix
        'C=C(C)C=O)OCC[N+](C)(C)[O-]': 'C=C(C)C(=O)OCC[N+](C)(C)[O-]',  # ONMA fix
    }
    smiles_to_use = smiles_fixes.get(smiles, smiles)

    mol = Chem.MolFromSmiles(smiles_to_use)
    if mol is None:
        return None

    # Morgan fingerprint
    fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    X_morgan = np.zeros((1, 2048), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp_morgan, X_morgan[0])

    # MACCS
    fp_maccs = MACCSkeys.GenMACCSKeys(mol)
    X_maccs = np.zeros((1, 167), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp_maccs, X_maccs[0])

    # Atom Pairs
    fp_ap = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024)
    X_atompairs = np.zeros((1, 1024), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp_ap, X_atompairs[0])

    # FP2 (using RDKFingerprint)
    fp_fp2 = Chem.RDKFingerprint(mol, fpSize=2048)
    X_fp2 = np.zeros((1, 2048), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp_fp2, X_fp2[0])

    # Combined - 仅包含4种特征 (与训练时一致)
    X_combined = np.hstack([X_morgan, X_maccs, X_atompairs, X_fp2])

    return {
        'Morgan': X_morgan,
        'MACCS': X_maccs,
        'AtomPairs': X_atompairs,
        'FP2': X_fp2,
        'Combined': X_combined
    }


# Load data
with st.spinner("Loading data and fitting models..."):
    data_dict = load_background_data()
    models_dict = fit_reduction_models(data_dict)

st.success("Data and models loaded successfully!")

# =============================================================================
# Preset Molecules Visualization
# =============================================================================
st.markdown("---")
st.header("🧬 Preset Molecules - LDA Visualization")

# Preset molecule data
preset_molecules = {
    'MPC': 'C=C(C)C(=O)OCCOP(=O)([O-])OCC[N+](C)(C)C',
    'CBMA': 'C=C(C)C(O)OCC[N+](C)(C)CC(=O)[O-]',
    'CBMA-2': 'C=C(C)C(=O)OCC[N+](C)(C)CCC(=O)[O-]',
    'CBMA-3': 'C=C(C)C(=O)OCC[N+](C)(C)CCCC(=O)[O-]',
    'SBMA': 'C=C(C)C(=O)OCC[N+](C)(C)CCCS(=O)(=O)[O-]',  # Fixed: C=O) → C(=O)
    'SBMA-2': 'C=C(C)C(=O)OCC[N+](C)(C)CCS(=O)(=O)[O-]',
    'MPSMA': 'C=C(C)C(=O)OCC(O)C[N+]1(C)CCC(S(=O)(=O)[O-])CC1',
    'DMAEMA': 'C=C(C)C(=O)OCCN(C)C',
    'ONMA': 'C=C(C)C(=O)OCC[N+](C)(C)[O-]',  # Fixed: C=O) → C(=O)
    'DSC6MA': 'C=C(C)C(=O)NCCCCCCNC(=O)[C@H](N)CO',
    'Betaine': 'C[N+](C)(C)CC(=O)[O-]',
    'DMACA': 'CN(C)c1ccc(/C=C/C(=O)O)cc1',
    'GABA': 'NCCCC(=O)O',
    'GLUT': 'OCC1OC(O)C(O)C(O)C1O',
    'Tryptamine': 'NCCc1c[nH]c2ccccc12',
    'taurine': 'NCCS(=O)(=O)[O-]',
    'TryC3MA': 'C=C(C)C(=O)NCCCC(=O)NCCc1c[nH]c2ccccc12',
    'DMACAC5MA': 'C=C(C)C(=O)NCCCCCCNC(=O)/C=C/c1ccc(N(C)C)cc1',
    'GLUTC4MA': 'C=C(C)C(=O)NCCCCNC(=O)OCC1OC(O)C(O)C(O)C1O'
}

# Color palette for preset molecules
import matplotlib.cm as cm
color_palette = [
    '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
    '#1ABC9C', '#E67E22', '#34495E', '#16A085', '#D35400',
    '#7F8C8D', '#27AE60', '#8E44AD', '#C0392B', '#F1C40F',
    '#BDC3C7', '#2C3E50'
]

# Compute features for all preset molecules
preset_features = {}
preset_names_list = list(preset_molecules.keys())

st.info(f"📊 Computing features for {len(preset_molecules)} preset molecules...")
with st.spinner("Computing features for preset molecules..."):
    for idx, (name, smiles) in enumerate(preset_molecules.items()):
        features = compute_single_smiles_features(smiles)
        if features is not None:
            preset_features[name] = features

if preset_features:
    st.success(f"✅ Successfully computed features for {len(preset_features)} molecules")

    # Set default visualization parameters
    default_method = "LDA"
    default_feature = "Morgan"
    default_split = "all"

    # Get model and background data
    model_key = f"{default_method}_{default_feature}"

    if model_key in models_dict:
        model_info = models_dict[model_key]
        X_background = model_info['transformed']
        meta = data_dict['meta']

        # Filter by split
        if default_split != "all":
            mask = meta['split'] == default_split
            X_background = X_background[mask]
            labels = meta[mask]['y_cls'].values
        else:
            labels = meta['y_cls'].values

        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot background points
        bbb_plus_mask = labels == 1
        bbb_minus_mask = labels == 0

        ax.scatter(X_background[bbb_plus_mask, 0], X_background[bbb_plus_mask, 1],
                  c='#2ecc71', label='BBB+ (Background)', alpha=0.3, s=20,
                  edgecolors='none')
        ax.scatter(X_background[bbb_minus_mask, 0], X_background[bbb_minus_mask, 1],
                  c='#e74c3c', label='BBB- (Background)', alpha=0.3, s=20,
                  edgecolors='none')

        # Transform and plot preset molecules
        for idx, (name, features) in enumerate(preset_features.items()):
            X_feat = features[default_feature]

            # Transform using LDA+PCA
            X_lda_1d = model_info['model'].transform(X_feat)
            X_pca_2nd = model_info['pca_ref'].transform(X_feat)
            X_new = np.array([[X_lda_1d[0, 0], X_pca_2nd[0, 1]]])

            color = color_palette[idx % len(color_palette)]
            ax.scatter(X_new[0, 0], X_new[0, 1],
                      c=color, marker='o', s=200, label=name,
                      edgecolors='black', linewidth=1.5, zorder=10)

            # Add text annotation
            ax.annotate(name, (X_new[0, 0], X_new[0, 1]),
                       fontsize=8, fontweight='bold', ha='center',
                       color='black', xytext=(5, 5), textcoords='offset points')

        # Labels and title
        ax.set_xlabel(f'{default_method} Dimension 1', fontsize=12, fontname='Times New Roman', fontweight='bold')
        ax.set_ylabel(f'{default_method} Dimension 2', fontsize=12, fontname='Times New Roman', fontweight='bold')
        ax.set_title(f'BBB Permeability - Preset Molecules Visualization\n({default_method} + {default_feature})',
                    fontsize=14, fontname='Times New Roman', fontweight='bold')

        # Legend
        ax.legend(loc='best', fontsize=7, prop={'family': 'Times New Roman'},
                  frameon=True, shadow=True, ncol=2)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        # Display plot
        st.pyplot(fig)

        # Molecule info
        with st.expander("📋 Molecule List"):
            st.dataframe(pd.DataFrame({
                'Name': list(preset_molecules.keys()),
                'SMILES': list(preset_molecules.values()),
                'Color': color_palette[:len(preset_molecules)]
            }))

# Input section
st.header("🔬 Enter New SMILES")

col1, col2 = st.columns([3, 1])

with col1:
    new_smiles = st.text_input(
        "SMILES String",
        value="CN1CCC[C@H]1c2cccnc2",
        help="Enter a SMILES string to visualize its position in the reduced space"
    )

with col2:
    st.write("")
    st.write("")
    show_legend = st.checkbox("Show Legend", value=True)

# Example SMILES
with st.expander("💡 Example SMILES"):
    example_smiles = {
        "Caffeine (BBB+)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Dopamine (BBB+)": "NCCc1ccc(O)c(O)c1",
        "Glucose (BBB-)": "C(C1C(C(C(C(O1)O)O)O)O)O",
        "Aspirin (BBB+)": "CC(=O)Oc1ccccc1C(=O)O",
        "Atorvastatin (BBB-)": "CC(C)C1CC(C(C)C1)C",
        "Nicotine (BBB+)": "CN1CCCC1c1ccncn1"
    }

    for name, smiles in example_smiles.items():
        if st.button(name, key=f"example_{name}"):
            new_smiles = smiles
            st.rerun()

# Main visualization
if new_smiles:
    st.markdown("---")
    st.header("📈 Visualization")

    # Compute features for new SMILES
    new_features = compute_single_smiles_features(new_smiles)

    if new_features is None:
        st.error("❌ Invalid SMILES string. Please enter a valid SMILES.")
    else:
        # Get model key
        model_key = f"{method}_{feature_type}"

        # Get background data and model
        meta = data_dict['meta']
        model_info = models_dict[model_key]
        X_background = model_info['transformed']

        # Filter by split
        if split != "all":
            mask = meta['split'] == split
            X_background = X_background[mask]
            labels = meta[mask]['y_cls'].values
            smiles_background = meta[mask]['SMILES'].values
        else:
            labels = meta['y_cls'].values
            smiles_background = meta['SMILES'].values

        # Transform new SMILES
        if method == "PCA":
            X_new = model_info['model'].transform(new_features[feature_type])
        elif method == "LDA":
            # LDA gives 1D, combine with PCA's 2nd dimension
            X_lda_1d = model_info['model'].transform(new_features[feature_type])
            X_pca_2nd = model_info['pca_ref'].transform(new_features[feature_type])
            X_new = np.array([[X_lda_1d[0, 0], X_pca_2nd[0, 1]]])
        else:  # t-SNE doesn't have transform, need to refit or use approximation
            # For t-SNE, we'll use a simpler approach: add to dataset and refit
            st.warning("⚠️ t-SNE doesn't support incremental transform. Showing PCA projection instead.")
            pca_key = f"PCA_{feature_type}"
            X_background = models_dict[pca_key]['transformed']
            if split != "all":
                mask = meta['split'] == split
                X_background = X_background[mask]
            X_new = models_dict[pca_key]['model'].transform(new_features[feature_type])
            method = "PCA (t-SNE not available for new points)"

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 9))

        # Plot background points
        bbb_plus_mask = labels == 1
        bbb_minus_mask = labels == 0

        ax.scatter(X_background[bbb_plus_mask, 0], X_background[bbb_plus_mask, 1],
                  c='#2ecc71', label='BBB+ (Background)', alpha=0.4, s=30,
                  edgecolors='none')
        ax.scatter(X_background[bbb_minus_mask, 0], X_background[bbb_minus_mask, 1],
                  c='#e74c3c', label='BBB- (Background)', alpha=0.4, s=30,
                  edgecolors='none')

        # Plot new SMILES
        ax.scatter(X_new[0, 0], X_new[0, 1],
                  c='gold', marker='*', s=800, label='New Molecule',
                  edgecolors='black', linewidth=2, zorder=10)

        # Add annotation
        ax.annotate('NEW', (X_new[0, 0], X_new[0, 1]),
                   fontsize=12, fontweight='bold', ha='center', va='center',
                   color='black', xytext=(0, 0), textcoords='offset points')

        # Labels and title
        ax.set_xlabel(f'{method} Dimension 1', fontsize=12, fontname='Times New Roman')
        ax.set_ylabel(f'{method} Dimension 2', fontsize=12, fontname='Times New Roman')
        ax.set_title(f'BBB Permeability - {method} Visualization ({feature_type})',
                    fontsize=14, fontname='Times New Roman', fontweight='bold')

        if show_legend:
            ax.legend(fontsize=10, prop={'family': 'Times New Roman'}, loc='best')

        ax.grid(True, alpha=0.3)

        # Display plot
        st.pyplot(fig, dpi=150)

        # Analysis
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Method",
                method,
                help="Dimensionality reduction method"
            )

        with col2:
            st.metric(
                "Features",
                feature_type,
                help="Type of molecular features used"
            )

        with col3:
            st.metric(
                "Dataset",
                f"{split.upper()} ({len(labels)} samples)",
                help="Background dataset used for visualization"
            )

        # Position analysis
        st.markdown("---")
        st.subheader("📍 Position Analysis")

        # Calculate distances to centroids
        centroid_plus = X_background[bbb_plus_mask].mean(axis=0)
        centroid_minus = X_background[bbb_minus_mask].mean(axis=0)

        from scipy.spatial.distance import euclidean
        dist_to_plus = euclidean(X_new[0], centroid_plus)
        dist_to_minus = euclidean(X_new[0], centroid_minus)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Distance to BBB+ Centroid",
                f"{dist_to_plus:.2f}",
                help="Euclidean distance to the center of BBB+ cluster"
            )

        with col2:
            st.metric(
                "Distance to BBB- Centroid",
                f"{dist_to_minus:.2f}",
                help="Euclidean distance to the center of BBB- cluster"
            )

        # Classification based on distance
        if dist_to_plus < dist_to_minus:
            st.success(f"✅ Closer to BBB+ cluster (difference: {dist_to_minus - dist_to_plus:.2f})")
            prediction = "BBB+"
        else:
            st.error(f"❌ Closer to BBB- cluster (difference: {dist_to_plus - dist_to_minus:.2f})")
            prediction = "BBB-"

        # Molecular information
        st.markdown("---")
        st.subheader("🧪 Molecular Information")

        mol = Chem.MolFromSmiles(new_smiles)
        if mol is not None:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Molecular Weight", f"{Descriptors.MolWt(mol):.1f}")

            with col2:
                st.metric("LogP", f"{Descriptors.MolLogP(mol):.2f}")

            with col3:
                st.metric("TPSA", f"{Descriptors.TPSA(mol):.1f}")

            with col4:
                st.metric("H-Donors", f"{Descriptors.NumHDonors(mol)}")

            # SMARTS matches
            st.write("**SMARTS Pattern Matches:**")
            matched_smarts = []
            for i, (name, pattern) in enumerate(data_dict['smarts_patterns']):
                try:
                    smarts_mol = Chem.MolFromSmarts(pattern)
                    if smarts_mol and mol.HasSubstructMatch(smarts_mol):
                        matched_smarts.append(name)
                except:
                    pass

            if matched_smarts:
                st.write(", ".join(matched_smarts[:10]))
                if len(matched_smarts) > 10:
                    st.write(f"... and {len(matched_smarts) - 10} more")
            else:
                st.write("No SMARTS patterns matched")

# Instructions
st.markdown("---")
st.header("ℹ️ How to Use")

st.markdown("""
1. **Select settings** in the sidebar:
   - Choose dimensionality reduction method:
     - **PCA**: Unsupervised linear method (fast, interpretable)
     - **t-SNE**: Non-linear method (good for visualization, slower)
     - **LDA**: Supervised linear method (maximizes class separation)
   - Select feature type to visualize
   - Choose which dataset split to display as background

2. **Enter a SMILES string** for the molecule you want to analyze

3. **View the visualization**:
   - Green points: Background BBB+ molecules
   - Red points: Background BBB- molecules
   - Gold star: Your new molecule

4. **Interpret the results**:
   - If the star is closer to green points → likely BBB+
   - If the star is closer to red points → likely BBB-
   - Check the distance metrics for quantitative assessment

**Note:**
- t-SNE doesn't support transforming new data points without retraining,
  so for new SMILES input with t-SNE selected, we'll show the PCA projection instead.
- LDA uses label information to find the most discriminative directions.
  For binary classification, it reduces to 1D, so we combine it with PCA's 2nd dimension.
""")

# Footer
st.markdown("---")
st.markdown("""
<style>
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.9em;
        padding: 20px;
    }
</style>
<div class="footer">
    <p>BBB Permeability Prediction Platform - Dimensionality Reduction Visualization</p>
    <p>Using PCA and t-SNE for exploratory data analysis</p>
</div>
""", unsafe_allow_html=True)
