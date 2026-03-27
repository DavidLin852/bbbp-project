# Project Context

## Project Name
BBBP prediction and constrained candidate discovery with B3DB and ZINC22

## Overall Goal
This project focuses on blood-brain barrier permeability prediction (BBBP) and future constrained molecular candidate discovery.

The long-term goal is not only to build accurate prediction models, but also to extract chemically meaningful structural contributions and support future candidate selection under polymer/material-related constraints.

The final application is not unrestricted de novo design. Instead, future candidate molecules may need to remain compatible with polymerization, grafting, or further material construction, potentially involving acrylate-like motifs, reactive handles, or structures compatible with POEGMA-related systems.

## Main Objectives
1. Build reliable BBB classification models for BBB+ / BBB-.
2. Build reliable regression models for logBB.
3. Compare classical ML models, graph models, and sequence models.
4. Improve molecular representation using ZINC22 pretraining.
5. Provide structural interpretability for candidate analysis.
6. Support future constrained candidate ranking instead of unconstrained generation.

## Available Datasets

### B3DB
There are two main datasets:
- classification dataset for BBB+ / BBB-
- regression dataset for logBB

These are the core downstream supervised datasets.

### ZINC22
ZINC22 drug-like subset has already been downloaded.
It is currently not yet fully integrated into the training pipeline.
Its main purpose is for molecular structure pretraining, not direct BBB supervision.

## Current Research Direction
The intended pipeline is:

1. Standardize and clean B3DB datasets.
2. Build reproducible baseline experiments on B3DB first.
3. Compare multiple molecular representations:
   - ECFP / circular fingerprints
   - FP2
   - MACCS
   - molecular descriptors
   - graph representation
   - SMILES / sequence representation
4. Compare multiple model families:
   - RF
   - XGBoost
   - LightGBM
   - SVM
   - KNN
   - GNN
   - Transformer
5. Later consider ensemble strategies such as voting or stacking.
6. Later introduce ZINC22-based pretraining and compare pretrained vs non-pretrained models.
7. Perform interpretability analysis to identify useful structural contributions.
8. Use constrained ranking / selection for future candidate discovery.

## Important Methodological Preferences
- B3DB baseline pipeline must be stabilized before scaling ZINC22 pretraining.
- Scaffold split should be treated as the main evaluation split.
- Random split can be used only as a reference.
- Classification and regression should be treated as separate tasks first.
- Reproducibility is important.
- Code structure should remain modular and maintainable.
- The project should be runnable on the CFFF platform.

## What Has Already Been Explored
- Preliminary modeling ideas have already been discussed across classical ML, GNN, and Transformer methods.
- Ensemble ideas such as stacking/voting have already been considered.
- Multiple fingerprints and representations have already been considered.
- Clustering / visualization methods such as PCA, t-SNE, LDA, and UMAP have already been explored conceptually or partially tested.
- Mechanism-inspired grouping based on the paper
  "Explaining Blood–Brain Barrier Permeability of Small Molecules by Integrated Analysis of Different Transport Mechanisms"
  has already been considered.
- Initial observation: molecules with similar transport mechanisms do not necessarily form clearly separated clusters.

## What Is Still Pending
- Standardized B3DB preprocessing pipeline
- Reproducible baseline experiment framework
- Clean experiment configuration structure
- Stable graph data pipeline
- ZINC22 pretraining pipeline
- Proper interpretability scripts
- Candidate ranking framework under future chemistry/material constraints

## Project Constraints
- Do not redesign the entire project from scratch unless explicitly requested.
- Prefer incremental, modular progress.
- Do not assume all ZINC22 data will be used at once.
- Avoid overly complex abstractions unless they clearly improve maintainability.
- Keep code and data management clearly separated.
- Data files, model checkpoints, and large outputs should not be committed into Git.
- The project should support a workflow where code is developed locally and run on CFFF.

## Preferred Project Structure
project/
  configs/
  scripts/
  src/
    data/
    features/
    models/
    train/
    evaluate/
    explain/
    utils/
  notebooks/
  results/

## Development Style
- Prefer modular Python code.
- Prefer clear function boundaries.
- Prefer reproducible scripts over ad hoc notebook-only logic.
- Keep scripts practical and easy to run on both local environment and CFFF.
- Add concise comments only where useful.
- Avoid unnecessary overengineering.

## Current Priority
Current highest priority is:
1. Standardize B3DB preprocessing
2. Build reproducible baselines
3. Prepare the project for smooth migration to CFFF
4. Only then expand into ZINC22 pretraining