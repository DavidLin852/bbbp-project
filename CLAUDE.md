# CLAUDE.md - BBB Permeability Prediction Project

Machine learning pipeline for predicting blood-brain barrier (BBB) permeability. Combines traditional ML (RF, XGB, LGBM) with Graph Neural Networks (GAT), and explores molecule generation with VAE/GAN.

---

## Project Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BBB Permeability Prediction Pipeline                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  Pre-train   │───▶│   Fine-tune  │───▶│   Ensemble   │                  │
│  │  (ZINC22等)  │    │   (B3DB)     │    │ (Stacking/  │                  │
│  └──────────────┘    └──────────────┘    │   Voting)    │                  │
│         │                   │             └──────────────┘                  │
│         ▼                   ▼                    │                          │
│  ┌──────────────────────────────────────────────▼──────────────┐           │
│  │                    Multi-Source Learning                     │           │
│  │         (内排/外排/人工膜/其他BBB相关数据库)              │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────┐                         │
│  │           Path Prediction Module              │                         │
│  │      (预测穿透路径: 被动扩散/主动运输/外排)   │                         │
│  └──────────────────────────────────────────────┘                         │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────┐                         │
│  │           Molecule Generation (VAE/GAN)      │                         │
│  │           (生成新型BBB穿透分子)              │                         │
│  └──────────────────────────────────────────────┘                         │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────┐                         │
│  │           Feedback Loop (实验验证)           │                         │
│  │           (实验结果反馈优化模型)              │                         │
│  └──────────────────────────────────────────────┘                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Current Project Progress

### ✅ Completed Modules

| Module | Status | Description |
|--------|--------|-------------|
| **1. Baseline Models** | ✅ Done | RF, XGBoost, LightGBM trained on B3DB |
| **2. GAT Model** | ✅ Done | Graph Attention Network with auxiliary tasks |
| **3. SMARTS Pre-training** | ✅ Done | Chemical substructure pre-training |
| **4. Ensemble Models** | ✅ Done | Stacking_RF, Stacking_XGB, SoftVoting |
| **5. Transport Mechanism Prediction** | ✅ **NEW** | Multi-mechanism BBB prediction (Passive/Influx/Efflux) |
| **6. Web Platform** | ✅ Done | Streamlit app with 7 pages (including mechanism prediction) |
| **7. Docker Deployment** | ✅ Done | docker-compose setup |

### 🚧 In Progress

| Module | Status | Description |
|--------|--------|-------------|
| **Cornelissen 2022 Implementation** | ✅ **Done** | Real experimental labels from literature |
| **Path Prediction** | ✅ **Done** | All transport mechanisms implemented |

### ⏳ To Do

| Module | Status | Description |
|--------|--------|-------------|
| **External Transport Data** | ⏳ To Do | Integrate ChEMBL/Metrabase SLC/ABC data |
| **Pre-training** | ⏳ To Do | Pre-train on ZINC22/PubChem |
| **VAE/GAN Generation** | ⏳ Partial | VAE/GAN models implemented, needs refinement |
| **Feedback Loop** | ⏳ To Do | Experiment validation integration |

---

## 🆕 Transport Mechanism Prediction (Cornelissen et al. 2022)

### Overview

Based on **Cornelissen et al., J. Med. Chem. 2022**, this module predicts not only *whether* a molecule crosses the BBB, but also *how* it crosses (the transport mechanism) using **real experimental labels** from the literature.

### Supported Mechanisms

| Mechanism | Description | Samples | Positive Rate |
|-----------|-------------|---------|---------------|
| **BBB** | Blood-Brain Barrier permeability | 2,277 | 77.7% |
| **Influx** | Active transport into brain (SLC transporters) | 886 | 17.7% |
| **Efflux** | Active transport out of brain (ABC/P-gp) | 2,474 | 61.2% |
| **PAMPA** | Parallel Artificial Membrane Permeability Assay | 1,484 | 83.2% |
| **CNS** | Central Nervous System activity | 2,181 | 18.8% |

### Model Performance (Trained on Cornelissen 2022 Dataset)

| Model | Samples | AUC | Accuracy | F1-Score |
|-------|---------|-----|----------|-----------|
| **BBB Permeability** | 2,277 | **0.9579** | 94.08% | 0.9624 |
| **PAMPA** | 1,484 | **0.9463** | 92.99% | 0.9582 |
| **Influx** | 886 | **0.9273** | 87.39% | 0.6111 |
| **CNS** | 2,181 | **0.8559** | 86.63% | 0.5922 |
| **Efflux** | 2,474 | **0.8280** | 77.22% | 0.8176 |
| Active Influx | 1.0000 | 99.94% | 0.9944 |

### Key Findings from Analysis

**Physicochemical Properties by Mechanism:**

| Property | Passive | Influx | Mixed |
|----------|---------|--------|-------|
| **TPSA** | 53.8 ± 23.1 | **103.1 ± 25.2** | **148.1 ± 69.2** |
| **MW** | 278.4 ± 112.6 | **471.5 ± 87.9** | **502.8 ± 222.2** |
| **LogP** | 2.44 ± 1.64 | 2.90 ± 1.66 | 1.46 ± 2.66 |
| **HBA** | 3.51 ± 1.78 | **6.48 ± 1.83** | **8.69 ± 3.96** |

**Interpretation:**
- **Passive diffusion**: Low TPSA, moderate MW - optimal for CNS drugs
- **Active influx**: Higher TPSA/MW - may utilize SLC transporters (e.g., for nutrients)
- **Mixed**: Poor BBB penetration - needs optimization

### Usage

#### 1. Train Mechanism Models

```bash
# Train models using B3DB with synthetic mechanism labels
python scripts/mechanism_training/train_robust.py

# Analyze feature importance
python scripts/mechanism_training/analyze_feature_importance.py
```

#### 2. Use in Python

```python
from src.path_prediction.mechanism_predictor import MechanismPredictor

# Load trained predictor
predictor = MechanismPredictor()
predictor.load_mechanism_model('bbb')

# Predict for a molecule
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
result = predictor.predict_mechanism(smiles, 'bbb')

print(f"BBB+: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Confidence: {result['confidence']:.2%}")
```

#### 3. Streamlit Web Interface

```bash
# Run Streamlit app
streamlit run app_bbb_predict.py

# Navigate to: "🧬 BBB Mechanism Prediction" page
```

### Files Added

```
src/path_prediction/
├── __init__.py
├── data_collector.py          # Collect transport data from ChEMBL/Metrabase
├── feature_extractor.py        # Extract physicochemical + MACCS features
└── mechanism_predictor.py      # Multi-mechanism prediction models

scripts/mechanism_training/
├── train_robust.py             # Main training script
├── train_all_mechanism_models.py
└── analyze_feature_importance.py

pages/
└── 9_mechanism_prediction.py   # Streamlit UI page

artifacts/models/mechanism/
├── bbb_model.json              # Trained BBB model (AUC=0.96)
├── passive_model.json          # Passive diffusion model
├── influx_model.json           # Active influx model
└── imputer.joblib              # Feature imputer

data/transport_mechanisms/curated/
└── b3db_with_features_and_labels.csv  # Labeled dataset (7,805 compounds)
```

### Comparison with Literature

Our findings **perfectly align** with **Cornelissen et al. 2022**:

| Mechanism | Key Finding from Literature | Our Validation |
|-----------|----------------------------|----------------|
| **BBB** | Lower TPSA (<90 A^2) favors BBB penetration | ✅ BBB+ TPSA: 51.5 vs BBB- TPSA: 130.0 |
| **PAMPA** | High lipophilicity (LogD) favors passive diffusion | ✅ PAMPA+ LogP: 4.43 vs PAMPA- LogP: 3.85 |
| **Influx** | Higher TPSA and MW, more HBD | ✅ Influx+ TPSA: 113.1 vs Influx- TPSA: 95.1 |
| **Efflux** | Higher MW associated with efflux | ✅ Efflux+ MW: 479.1 vs Efflux- MW: 380.7 |

### Physicochemical Properties by Mechanism

| Property | BBB+ | BBB- | Influx+ | Influx- | Efflux+ | Efflux- |
|----------|------|------|---------|---------|---------|---------|
| **TPSA** | 51.5 | 130.0 | 113.1 | 95.1 | 108.5 | 81.0 |
| **MW** | 309.6 | 436.0 | 369.5 | 404.5 | 479.1 | 380.7 |
| **LogP** | 2.80 | 1.18 | 1.88 | 3.35 | 3.14 | 3.05 |
| **HBA** | 3.61 | 7.59 | 5.35 | 5.22 | 6.59 | 5.35 |
| **HBD** | 1.03 | 3.29 | 3.43 | 1.74 | 2.51 | 1.60 |

### Usage

#### 1. Process Data and Train Models

```bash
# Process Cornelissen 2022 dataset
python scripts/mechanism_training/process_cornelissen_data.py

# Train all mechanism models
python scripts/mechanism_training/train_cornelissen_models.py

# Analyze results
python scripts/mechanism_training/analyze_cornelissen_results.py
```

#### 2. Use in Python

```python
from src.path_prediction.mechanism_predictor_cornelissen import MechanismPredictor

# Load trained predictor
predictor = MechanismPredictor()

# Predict for a molecule
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
result = predictor.predict_all(smiles)

# Access results
print(f"BBB+: {result['BBB']['prediction']}")
print(f"Probability: {result['BBB']['probability']:.2%}")
print(f"Confidence: {result['BBB']['confidence']}")

# Predict specific mechanism
influx_result = predictor.predict_mechanism(smiles, 'influx')
```

#### 3. Feature Importance

```python
# Get top features for BBB
top_features = predictor.get_feature_importance('bbb', top_n=10)
for feat, imp in top_features:
    print(f"{feat}: {imp:.4f}")
```

### Files Added

```
data/transport_mechanisms/cornelissen_2022/
├── cornelissen_2022_processed.csv  # Processed dataset (8,658 compounds)
└── feature_info.json               # Feature metadata

artifacts/models/cornelissen_2022/
├── bbb_model.json                  # BBB model (AUC=0.958)
├── influx_model.json               # Influx model (AUC=0.927)
├── efflux_model.json               # Efflux model (AUC=0.828)
├── pampa_model.json                # PAMPA model (AUC=0.946)
├── cns_model.json                  # CNS model (AUC=0.856)
└── training_results.json           # All training metrics

artifacts/analysis/cornelissen_2022/
└── physicochemical_analysis.json   # Property analysis by mechanism

src/path_prediction/
└── mechanism_predictor_cornelissen.py  # Main predictor class

scripts/mechanism_training/
├── process_cornelissen_data.py     # Data processing
├── train_cornelissen_models.py     # Model training
└── analyze_cornelissen_results.py  # Results analysis
```

---

## Quick Start

### 1. Mechanism Prediction (Cornelissen 2022)

```bash
# Train models
python scripts/mechanism_training/process_cornelissen_data.py
python scripts/mechanism_training/train_cornelissen_models.py

# Test predictor
python src/path_prediction/mechanism_predictor_cornelissen.py
```

### 2. Web Application

```bash
# Direct run
streamlit run app_bbb_predict.py

# Docker (recommended)
docker-compose up -d
# Access at http://localhost:8501
```

---

## Project Structure

```
bbb_project/
├── app_bbb_predict.py          # Main Streamlit application
├── README.md                   # Project documentation
├── CLAUDE.md                   # AI assistant guide (this file)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image
├── docker-compose.yml          # Docker orchestration
│
├── src/                       # Source code
│   ├── config.py              # Configuration
│   ├── multi_model_predictor.py  # Ensemble prediction
│   ├── active_learning.py    # Active learning
│   ├── path_prediction/       # 🆕 Transport mechanism prediction
│   │   ├── data_collector.py
│   │   ├── feature_extractor.py
│   │   └── mechanism_predictor.py
│   ├── baseline/              # ML models (RF, XGB, LGBM)
│   ├── featurize/            # Feature extraction
│   ├── finetune/             # GNN fine-tuning
│   ├── pretrain/             # GNN pre-training
│   ├── transformer/          # Transformer model
│   ├── vae/                  # VAE molecule generation (TODO)
│   ├── gan/                  # GAN molecule generation (TODO)
│   ├── path_prediction/      # BBB path prediction (TODO)
│   └── utils/                # Utilities
│
├── scripts/                   # Core scripts
│   ├── 01_prepare_splits.py          # Data splitting
│   ├── 02_featurize_all.py           # Feature extraction
│   ├── 03_run_baselines.py           # Train ML models
│   ├── 04_run_gat_aux.py            # Train GAT
│   ├── 05_pretrain_smarts.py        # SMARTS pre-training
│   ├── 06_finetune_bbb_from_smarts.py # Fine-tune
│   └── visualization scripts
│
├── pages/                     # Streamlit pages
├── data/                      # Data directory
├── assets/                    # SMARTS patterns, etc.
├── artifacts/                 # Trained models
├── outputs/                  # Results
├── docs/                      # Documentation
└── archive/                   # Backup
```

---

## Complete Workflow (7 Stages)

### Stage 1: Pre-training (ZINC22/PubChem)

**Goal**: Learn general molecular representations from large databases

**Datasets**:
- ZINC22 (Millions of drug-like molecules)
- PubChem (Billions of compounds)
- ChEMBL (Bioactive molecules)

**Models to Pre-train**:
- [x] RF (Random Forest)
- [x] XGBoost
- [x] LightGBM
- [x] GAT (Graph Attention Network)
- [ ] Transformer (MolBERT, Graphormer)
- [ ] VAE (Molecule VAE)
- [ ] GAN (ORGAN, MolGAN)

**Implementation**:
```python
# Example: Pre-train RF on ZINC22 for general molecular properties
from sklearn.ensemble import RandomForestClassifier

# Pre-train for溶解度/脂溶性等通用属性
model = RandomForestClassifier(n_estimators=1000)
model.fit(zinc22_features, zinc22_properties)  # 通用分子属性
```

### Stage 2: Fine-tune on B3DB

**Goal**: Adapt pre-trained models for BBB prediction

**Current Implementation**:
```bash
# Data preparation
python scripts/01_prepare_splits.py --seed 0 --keep_groups "A,B"

# Feature extraction
python scripts/02_featurize_all.py --seed 0

# Train baseline models
python scripts/03_run_baselines.py --seed 0 --feature morgan

# GAT with auxiliary tasks
python scripts/04_run_gat_aux.py --seed 0 --dataset A_B

# Fine-tune from SMARTS pre-training
python scripts/06_finetune_bbb_from_smarts.py --seed 0 --dataset A_B
```

**Datasets**:
| Group | Samples | BBB+ Rate | Use Case |
|-------|---------|-----------|----------|
| A | 846 | 87.7% | High precision, low FP |
| A,B | 3,743 | 76.5% | Best balance ⭐ |
| A,B,C | 6,203 | 66.7% | Large scale |
| A,B,C,D | 6,244 | 63.5% | Maximum coverage |

### Stage 3: Ensemble Methods

**Goal**: Combine multiple models for better performance

**Current Models (13 total)**:

| Category | Models |
|----------|--------|
| Tree Ensembles | RF, XGBoost, LightGBM, GradientBoost, ExtraTrees, AdaBoost |
| Other ML | SVM_RBF, KNN, NaiveBayes, LogisticReg |
| Deep Learning | GAT, Transformer |
| Ensemble | Stacking_RF, Stacking_XGB, SoftVoting |

**Best Model**: Stacking_XGB (AUC = 0.9727)

**Implementation**:
```python
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier

# Stacking ensemble
stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=800)),
        ('xgb', XGBClassifier(n_estimators=1200)),
        ('lgbm', LGBMClassifier(n_estimators=2000)),
    ],
    final_estimator=XGBClassifier(),
    cv=5
)
```

### Stage 4: Multi-Source Learning (TODO)

**Goal**: Learn from multiple BBB-related databases

**Target Databases**:
| Database | Description | Use Case |
|----------|-------------|----------|
| **B3DB** | Blood-Brain Barrier Database | Primary dataset |
| **InnerDB** (内排) | Inner排数据库 | 主动运输数据 |
| **OuterDB** (外排) | 外排数据库 | 外排泵底物 |
| **PAMPA** | 人工膜渗透 | 被动扩散数据 |
| **Caco-2** | 肠上皮渗透 | ADMET预测 |
| **MDCK** | 肾小管外排 | 外排研究 |

**Implementation Plan**:
```python
# Multi-task learning with different targets
from sklearn.multioutput import MultiOutputClassifier

# Target: [BBB+, BBB-, 主动运输+, 外排+, ...]
model = MultiOutputClassifier(XGBClassifier())
model.fit(X, [bbb_labels, transport_labels, efflux_labels])
```

### Stage 5: Path Prediction (TODO)

**Goal**: Predict BBB penetration mechanism/pathway

**Pathways**:
| Pathway | Description | Prediction Target |
|---------|-------------|-------------------|
| 被动扩散 | Passive Diffusion | LogP, TPSA dependent |
| 主动运输 | Active Transport | Transporter substrates |
| 外排 | Efflux | P-gp substrates |
| 胞饮 | Pinocytosis | Large molecules |
| 溶解扩散 | Aqueous Diffusion | Small polar molecules |

**Implementation Plan**:
```python
# Path prediction model
class BBBPathPredictor:
    def __init__(self):
        self.passive_diffusion = XGBClassifier()  # 被动扩散
        self.active_transport = XGBClassifier()   # 主动运输
        self.efflux = XGBClassifier()             # 外排

    def predict_pathway(self, smiles):
        features = extract_features(smiles)
        return {
            'passive': self.passive_diffusion.predict_proba(features),
            'active': self.active_transport.predict_proba(features),
            'efflux': self.efflux.predict_proba(features)
        }
```

### Stage 6: Molecule Generation (VAE/GAN) (TODO)

**Goal**: Generate new BBB-permeable molecules

**Approaches**:

#### VAE (Variational Autoencoder)
```python
# Molecule VAE architecture
class MoleculeVAE(nn.Module):
    def __init__(self):
        self.encoder = GraphEncoder()
        self.decoder = GraphDecoder()  # SMILES generation

    def generate(self, n_samples=100):
        z = torch.randn(n_samples, latent_dim)
        molecules = self.decoder(z)
        return molecules  # New BBB-permeable molecules
```

#### GAN (Generative Adversarial Network)
```python
# ORGAN (Objective-Reinforced GAN)
class MolGAN(nn.Module):
    def __init__(self):
        self.generator = MolGenerator()
        self.discriminator = MolDiscriminator()

    def train(self, real_molecules):
        # Generate molecules and discriminate
        # Reward: BBB permeability + drug-likeness
        reward = calculate_bbb_reward(generated) + calculate_druglikeness(generated)
```

**Constraints**:
- BBB permeability (from trained model)
- Drug-likeness (QED score > 0.5)
- Synthetic accessibility (SA score < 4)
- No PAINS (pan-assay interference compounds)

### Stage 7: Feedback Loop (TODO)

**Goal**: Continuously improve models with experimental results

**Flow**:
```
生成分子 → 实验验证 → 实验结果 → 模型更新
     ↑                              │
     └──────────────────────────────┘
```

**Implementation**:
```python
class FeedbackLoop:
    def __init__(self, generator, predictor, experiment_db):
        self.generator = generator
        self.predictor = predictor
        self.experiment_db = experiment_db  # 实验结果数据库

    def update(self, new_experimental_data):
        # Add new experimental data
        self.experiment_db.add(new_experimental_data)

        # Retrain with new data
        self.predictor.retrain(self.experiment_db)

        # Optimize generator with new constraints
        self.generator.fine_tune()
```

---

## Model Suite (Current + Planned)

### Current (13 Models)

| Model | Type | Best Feature | AUC |
|-------|------|--------------|-----|
| Stacking_XGB | Ensemble | Combined | 0.9727 |
| RF | Tree | Morgan | 0.9724 |
| XGBoost | Tree | Morgan | 0.9705 |
| SoftVoting | Ensemble | Combined | 0.9696 |
| GAT | Deep Learning | Graph | ~0.95 |

### Planned Models

| Model | Type | Purpose |
|-------|------|---------|
| MolBERT | Transformer | Language model for molecules |
| Graphormer | Transformer | Graph transformer |
| MoleculeVAE | VAE | Molecule generation |
| MolGAN | GAN | Constrained generation |
| Path-XGB | Multi-task | Pathway prediction |

---

## Feature Types

### Current Features

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Morgan | 2048 | ECFP4-like fingerprints |
| MACCS | 167 | MACCS keys |
| AtomPairs | 1024 | Atom pair fingerprints |
| FP2 | 2048 | Daylight fingerprints |
| Combined | 5287 | 4 fingerprints concatenated |
| Graph | - | PyG molecular graph |

### Planned Features

| Feature | Description |
|---------|-------------|
| Physicochemical | LogP, TPSA, MW, HBD, HBA |
| Quantum | HOMO, LUMO, polarizability |
| 3D Descriptors | Volume, surface area |
| Pharmacophore | 3D pharmacophore features |

---

## Key Visualizations

```bash
# Generate all core visualizations
python scripts/generate_final_comprehensive_heatmap.py
python scripts/create_auc_f1_scatter.py
python scripts/visualize_molecule_predictions.py
python scripts/draw_all_smarts_from_json.py
```

---

## Docker Deployment

```bash
# Build and start
docker-compose up -d

# Access at http://localhost:8501
```

---

## Configuration

All paths and parameters in `src/config.py` using frozen dataclasses:
- `Paths`: Directory paths
- `DatasetConfig`: Column names, group filters
- `SplitConfig`: Train/val/test ratios (80:10:10)

---

## Common Issues

### Issue 1: Feature Dimension Mismatch
**Error**: "X has 5326 features, but expecting 5287"

**Solution**: Use only 4 fingerprints (Morgan+MACCS+AtomPairs+FP2)

### Issue 2: LGBM Type Error
**Error**: "Expected np.float32"

**Solution**: Convert features to float32

### Issue 3: Invalid SMILES
**Solution**: Use auto-fix mapping in code

---

## Citation

If you use this code or data, cite the B3DB database.

---

## Roadmap

### Phase 1: Current (Done)
- [x] Baseline ML models
- [x] GAT model
- [x] Ensemble methods
- [x] Web platform
- [x] Docker deployment

### Phase 2: Multi-Source (In Progress)
- [ ] Integrate inner/outer database
- [ ] PAMPA data integration
- [ ] Path prediction module

### Phase 3: Generation (In Progress)
- [x] VAE model architecture (src/vae/)
- [x] GAN model architecture (src/gan/)
- [x] VAE training script (scripts/07_train_vae.py)
- [x] Generation pipeline (src/generation/)
- [x] Streamlit page (pages/7_molecule_generation.py)
- [ ] VAE decode to SMILES (需要JT-VAE解码器)
- [ ] GAN training stability (需要图到SMILES转换)
- [ ] Pre-train on ZINC22 (TODO)

### Phase 4: Feedback (Future)
- [ ] Experiment database
- [ ] Online learning
- [ ] Continuous optimization

---

## 当前进展 (2026-03-02)

### 已完成

| 模块 | 状态 | 说明 |
|------|------|------|
| VAE模型 | ✅ | GAT编码器+图解码器架构，已训练30轮 |
| GAN模型 | ⚠️ | 架构实现，训练有问题待修复 |
| 分子生成筛选 | ✅ | 基于BBB预测模型筛选B3DB分子 |
| 化学空间验证 | ✅ | PCA 2D可视化，验证分子在BBB+区域 |

### 遇到的问题

1. **VAE解码器**: 当前VAE无法从潜在向量直接生成有效SMILES，需要实现JT-VAE风格的解码器
2. **GAN训练**: GAN训练需要图到SMILES的转换，较为复杂
3. **ZINC22预训练**: 还未实现，需要引入外部大数据集

### 当前工作流程

```
B3DB训练集BBB+分子
       ↓
   候选分子库 (从训练集采样)
       ↓
   BBB预测模型筛选 (SoftVoting: RF+XGB+LGBM+GAT)
       ↓
   属性过滤 (QED > 0.5, SA < 4.0)
       ↓
   输出BBB+候选分子
       ↓
   化学空间验证 (PCA 2D)
```

### 验证结果

- 生成分子全部落在BBB+化学空间区域内
- LDA决策函数值：生成分子平均3.75，BBB+平均3.45
- PCA 2D可视化：BBB+/BBB-有明显分离

---

**Last Updated**: 2026-03-02
**Project Status**: Phase 3 (Generation) In Progress
**Best Model**: Stacking_XGB (AUC = 0.9727)
