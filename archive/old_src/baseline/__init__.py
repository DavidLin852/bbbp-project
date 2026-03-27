"""
Baseline Machine Learning Models for BBB Permeability Prediction

Supports multiple ML algorithms:
- RF: Random Forest
- XGB: XGBoost
- LGBM: LightGBM
- SVM: Support Vector Machine (RBF, Linear, Poly)
- KNN: K-Nearest Neighbors
- NB: Naive Bayes (Gaussian, Bernoulli)
- LR: Logistic Regression
- MLP: Multi-Layer Perceptron
- GB: Gradient Boosting
- ADA: AdaBoost
- ETC: Extra Trees Classifier
"""

from .train_baselines import (
    train_eval_all_models,
    train_single_model,
    MODEL_CONFIGS,
    ALL_MODELS,
    TRADITIONAL_MODELS,
    QUICK_MODELS,
    ENSEMBLE_MODELS,
    SVM_MODELS,
    KNN_MODELS,
    NB_MODELS,
    NN_MODELS,
    get_available_models,
    get_model_info
)

__all__ = [
    # Training functions
    'train_eval_all_models',
    'train_single_model',

    # Model configurations
    'MODEL_CONFIGS',
    'ALL_MODELS',
    'TRADITIONAL_MODELS',
    'QUICK_MODELS',
    'ENSEMBLE_MODELS',
    'SVM_MODELS',
    'KNN_MODELS',
    'NB_MODELS',
    'NN_MODELS',

    # Utility functions
    'get_available_models',
    'get_model_info'
]
