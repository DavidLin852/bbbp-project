# BBB Permeability Prediction Project

**Machine learning pipeline for predicting blood-brain barrier (BBB) permeability** using the B3DB classification dataset.

Combines traditional ML models (Random Forest, XGBoost, LightGBM) with Graph Neural Networks (GAT) for binary classification (BBB+ vs BBB-).

---

## 🌟 Key Features

- **Multi-model prediction platform**: RF, XGB, LGBM, GAT with/without SMARTS enhancement
- **Interactive web interface**: Streamlit-based prediction and analysis
- **SMARTS substructure analysis**: Chemical feature importance visualization
- **Complete ML pipeline**: From data preprocessing to model deployment
- **Comprehensive evaluation**: 6 publication-quality comparison charts

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create conda environment
conda env create -f environment.yml
conda activate bbb
```

### 2. Run Web Application

```bash
streamlit run app_bbb_predict.py
```

Access at: **http://localhost:8502**

**Available Pages:**
- **Home**: Model comparison and performance overview
- **Prediction**: Single/batch BBB permeability prediction
- **SMARTS Analysis**: Chemical substructure importance
- **Model Comparison**: Interactive performance analysis

### 3. Generate Model Comparison Plots

```bash
python generate_plots_from_export.py
```

Output: `outputs/model_comparison/` (6 PNG charts + summary CSV)

---

## 📊 Model Performance

Based on comprehensive evaluation (32 model-dataset combinations):

| Rank | Model | Avg AUC | Avg Precision | Avg Recall | Avg F1 |
|------|-------|---------|--------------|-----------|--------|
| 🥇 | **XGB** | 0.947 | 0.902 | 0.962 | 0.931 |
| 🥈 | **LGBM** | 0.943 | 0.924 | 0.933 | 0.928 |
| 🥉 | **RF+SMARTS** | 0.945 | 0.893 | 0.963 | 0.927 |
| 4 | RF | 0.948 | 0.893 | 0.959 | 0.925 |
| 5 | GAT+SMARTS | 0.941 | 0.901 | 0.950 | 0.925 |

### Key Findings

- **XGB baseline performs best** - SMARTS enhancement degrades tree model performance
- **SMARTS pretraining only benefits GAT** - +5.3% AUC improvement for GNN models
- **Medium dataset size optimal** - A,B (n=468) shows best performance
- **GAT+SMARTS excels on small data** - Best on dataset A (n=106) with AUC=0.948

---

## 🛠️ Core Pipeline

### Stage 1: Data Preparation
```bash
python scripts/01_prepare_splits.py --seed 0
```

**Output**: Stratified 80:10:10 train/val/test splits

### Stage 2: Feature Extraction
```bash
python scripts/02_featurize_all.py --seed 0
```

**Features**:
- Morgan fingerprints (2048-bit, radius=2)
- RDKit molecular descriptors (13 properties)
- PyTorch Geometric graphs (for GNN models)

### Stage 3: Baseline Models
```bash
python scripts/03_run_baselines.py --seed 0 --feature morgan
```

**Models**: RF, XGB, LGBM with class weighting

### Stage 4: GNN Pipeline (Optional)
```bash
python run_gnn_pipeline.py --seed 0 --step 1  # GAT baseline
python run_gnn_pipeline.py --seed 0 --step 2  # GAT + auxiliary tasks
python run_gnn_pipeline.py --seed 0 --step 3  # SMARTS pretraining
python run_gnn_pipeline.py --seed 0 --step 4  # BBB fine-tuning
```

---

## 📁 Project Structure

```
bbb_project/
├── app_bbb_predict.py              # Streamlit main application
├── run_gnn_pipeline.py             # GNN training pipeline
├── generate_plots_from_export.py   # Model comparison visualization
│
├── scripts/                        # Core pipeline (20 scripts)
│   ├── 01-06                      # Data prep → fine-tuning
│   ├── 07-13                      # Analysis & visualization
│   └── 14-15                      # Prediction & ablation
│
├── pages/                          # Streamlit pages
│   ├── 0_prediction.py
│   ├── 1_smarts_analysis.py
│   ├── 2_model_comparison.py
│   └── 3_active_learning.py
│
├── src/                            # Core modules
│   ├── config.py                   # Configuration
│   ├── baseline/                   # Baseline models
│   ├── featurize/                  # Feature extraction
│   ├── gnn/                        # GNN models
│   └── utils/                      # Utilities
│
├── outputs/model_comparison/       # Generated plots
└── tools/visualization_template/   # Standalone tool
```

---

## 📊 Dataset Information

| Dataset | Test Size | BBB+ | BBB- | Positive Rate |
|---------|-----------|------|------|---------------|
| A | 106 | 93 | 13 | 87.7% |
| A,B | 468 | 341 | 127 | 72.9% |
| A,B,C | 776 | 496 | 280 | 63.9% |
| A,B,C,D | 781 | 496 | 285 | 63.5% |

**Note**: Test set size (10% split), not total dataset size.

---

## 🎯 Usage Examples

### Predict Single Molecule

```python
# Via web interface
streamlit run app_bbb_predict.py
# → Go to "Prediction" page
# → Enter SMILES: "CCO" (ethanol)
# → Click "Predict"
```

### Batch Prediction

```python
# Via CLI
python scripts/14_predict_smiles_cli.py \
    --smiles "CCO", "c1ccccc1" \
    --model XGB \
    --dataset A,B
```

### SMARTS Analysis

```bash
# Run SMARTS importance analysis
python scripts/09_explain_smarts.py --seed 0 --baseline --top 20
```

---

## 🔬 Advanced Features

### SMARTS Enhancement

**Features**: 70 chemical substructures from `assets/smarts/bbb_smarts_v1.json`

**Results**:
- **GAT+SMARTS**: +5.3% AUC improvement ✅
- **RF/LGBM/XGB+SMARTS**: -0.5% to -2.7% AUC ❌

**Why?** Tree models already extract substructures from Morgan fingerprints; GNNs benefit from chemical prior knowledge.

### Model Comparison Visualizations

Six comprehensive charts generated by `generate_plots_from_export.py`:

1. **Performance Heatmap** - All metrics across all models
2. **AUC Barplot** - Direct model comparison
3. **AUC vs F1 Scatter** - Color=models, Shape=datasets
4. **Dataset Complexity** - Performance vs sample size
5. **Fixed Model** - Deep dive into representative models
6. **Fixed Dataset** - Model comparison per dataset

---

## 📖 Documentation

- **`CLAUDE.md`** - Comprehensive project documentation for AI assistants
- **`outputs/model_comparison/CHARTS_GUIDE.md`** - Chart explanations
- **`outputs/model_comparison/EXPORT_DATA_ANALYSIS.md`** - Performance analysis
- **`outputs/model_comparison/DATASET_SIZES_EXPLANATION.md`** - Dataset details

---

## ⚙️ Configuration

All paths and parameters in `src/config.py` (frozen dataclasses):

```python
@dataclass(frozen=True)
class Paths:
    root: Path = Path.cwd()
    data: Path = root / "data"
    # ...

@dataclass(frozen=True)
class DatasetConfig:
    filename: str = "B3DB_classification.tsv"
    # ...
```

**Benefits**: Immutable, type-safe, centralized configuration.

---

## 🐛 Common Issues

### Feature Dimension Mismatch
```python
# Error: X has 2048 features, but expects 2118
# Solution: SMARTS models need 2048 Morgan + 70 SMARTS
X = X.astype(np.float32)  # Also convert to float32 for LightGBM
```

### Missing Splits
```bash
# Error: data/splits/seed_0/test.csv not found
# Solution: Run data preparation first
python scripts/01_prepare_splits.py --seed 0
```

### Model Loading
```python
# GAT models have checkpoint structure
checkpoint = torch.load('best.pt')
model.load_state_dict(checkpoint['model'])  # Extract 'model' key!
```

---

## 🎓 Citation

If you use this code or data, please cite the B3DB database.

---

## 📄 License

This project is for research and educational purposes.

---

## 🙏 Acknowledgments

- **B3DB Database**: Blood-Brain Barrier Database for computational brain drug delivery
- **RDKit**: Open-source cheminformatics
- **PyTorch Geometric**: Graph neural network library
- **Streamlit**: Web app framework

---

**Last Updated**: 2025-01-27
**Project Status**: Production-ready ✅
**Python Version**: 3.10
**Data Version**: 2026-01-27T02-03_export.csv
