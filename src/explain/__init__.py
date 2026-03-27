"""
Explainability module for BBB Permeability Prediction

提供多种可解释性方法：
- Grad x Input: 基于梯度的原子级归因
- SMARTS Occlusion: SMARTS子结构遮挡分析
- SHAP: SHAP值分析（新增）
"""

from __future__ import annotations

from .atom_grad import grad_x_input_atom_scores, plot_atom_attribution
from .smarts_occlusion import SMARTSOccluder, smarts_occlusion_analysis
from .shap_analysis import (
    SHAPExplainer,
    SHAPConfig,
    ModelType,
    explain_model,
    identify_toxicophores_from_smarts,
    map_shap_to_toxicophores,
    COMMON_TOXICOPHORES
)

__all__ = [
    # Grad x Input
    'grad_x_input_atom_scores',
    'plot_atom_attribution',

    # SMARTS Occlusion
    'SMARTSOccluder',
    'smarts_occlusion_analysis',

    # SHAP (新增)
    'SHAPExplainer',
    'SHAPConfig',
    'ModelType',
    'explain_model',
    'identify_toxicophores_from_smarts',
    'map_shap_to_toxicophores',
    'COMMON_TOXICOPHORES'
]
