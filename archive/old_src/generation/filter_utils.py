"""
Molecule filtering utilities.

Provides filtering and validation for generated molecules:
- Validity filtering (SMILES validity)
- Property filtering (QED, BBB, SA)
- Novelty filtering (exclude training set molecules)
- Duplicate removal
- Diversity filtering
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


@dataclass
class FilteredMolecules:
    """Result of molecule filtering."""
    original: List[str]
    valid: List[str]
    filtered: List[str]
    validity_rate: float
    filter_summary: dict


class MoleculeFilter:
    """
    Filter molecules based on multiple criteria.

    Criteria:
    - Validity: Can RDKit parse the SMILES?
    - QED: Drug-likeness score
    - BBB: BBB permeability probability
    - SA: Synthetic accessibility score
    - Novelty: Not in training set
    - Uniqueness: No duplicates
    """

    def __init__(
        self,
        min_qed: float = 0.5,
        min_bbb_prob: float = 0.7,
        max_sa_score: float = 4.0,
        training_smiles: Optional[Set[str]] = None,
        remove_duplicates: bool = True,
        check_novelty: bool = True,
    ):
        self.min_qed = min_qed
        self.min_bbb_prob = min_bbb_prob
        self.max_sa_score = max_sa_score
        self.training_smiles = training_smiles or set()
        self.remove_duplicates = remove_duplicates
        self.check_novelty = check_novelty

    def is_valid(self, smi: str) -> bool:
        """Check if SMILES is chemically valid."""
        return Chem.MolFromSmiles(smi) is not None

    def get_qed(self, smi: str) -> float:
        """Get QED score."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0.0
        try:
            return QED.qed(mol)
        except:
            return 0.0

    def get_sa_score(self, smi: str) -> float:
        """Get SA score."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 10.0

        # Simplified SA score
        from ..vae.molecule_vae import compute_sa_score
        return compute_sa_score(smi)

    def is_novel(self, smi: str) -> bool:
        """Check if molecule is novel (not in training set)."""
        if not self.check_novelty:
            return True

        # Canonicalize SMILES
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        canonical = Chem.MolToSmiles(mol, canonical=True)

        return canonical not in self.training_smiles

    def passes_filters(
        self,
        smi: str,
        qed: float,
        bbb_prob: float,
        sa_score: float,
    ) -> bool:
        """Check if molecule passes all filters."""
        return (
            qed >= self.min_qed and
            bbb_prob >= self.min_bbb_prob and
            sa_score <= self.max_sa_score
        )

    def filter(
        self,
        smiles_list: List[str],
        bbb_predictor,
        show_progress: bool = True,
    ) -> FilteredMolecules:
        """
        Filter molecules based on all criteria.

        Args:
            smiles_list: List of SMILES strings
            bbb_predictor: BBB prediction model
            show_progress: Show progress bar

        Returns:
            FilteredMolecules result
        """
        original_count = len(smiles_list)

        # Filter by validity
        valid_smiles = []
        for smi in tqdm(smiles_list, desc="Validating", disable=not show_progress):
            if self.is_valid(smi):
                valid_smiles.append(smi)

        validity_rate = len(valid_smiles) / original_count if original_count > 0 else 0.0

        # Compute properties
        qed_scores = []
        bbb_probs = []
        sa_scores = []
        novel_smiles = []

        for smi in tqdm(valid_smiles, desc="Computing properties", disable=not show_progress):
            qed = self.get_qed(smi)
            sa = self.get_sa_score(smi)

            # BBB prediction
            try:
                results = bbb_predictor.predict([smi])
                bbb_prob = float(results.ensemble_probability[0])
            except:
                bbb_prob = 0.0

            # Check novelty
            if self.is_novel(smi):
                novel_smiles.append(smi)
                qed_scores.append(qed)
                bbb_probs.append(bbb_prob)
                sa_scores.append(sa)

        # Apply property filters
        filtered = []
        for smi, qed, bbb, sa in zip(novel_smiles, qed_scores, bbb_probs, sa_scores):
            if self.passes_filters(smi, qed, bbb, sa):
                filtered.append(smi)

        # Remove duplicates
        if self.remove_duplicates:
            filtered = self._canonicalize_and_deduplicate(filtered)

        filter_summary = {
            'original': original_count,
            'valid': len(valid_smiles),
            'novel': len(novel_smiles),
            'passed_filters': len(filtered),
            'validity_rate': validity_rate,
            'avg_qed': np.mean(qed_scores) if qed_scores else 0.0,
            'avg_bbb_prob': np.mean(bbb_probs) if bbb_probs else 0.0,
            'avg_sa_score': np.mean(sa_scores) if sa_scores else 10.0,
        }

        return FilteredMolecules(
            original=smiles_list,
            valid=valid_smiles,
            filtered=filtered,
            validity_rate=validity_rate,
            filter_summary=filter_summary,
        )

    def _canonicalize_and_deduplicate(self, smiles_list: List[str]) -> List[str]:
        """Remove duplicates by canonicalizing SMILES."""
        seen = set()
        unique = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True)

            if canonical not in seen:
                seen.add(canonical)
                unique.append(smi)

        return unique


def filter_molecules(
    smiles_list: List[str],
    bbb_predictor,
    min_qed: float = 0.5,
    min_bbb_prob: float = 0.7,
    max_sa: float = 4.0,
    training_smiles: Optional[Set[str]] = None,
    remove_duplicates: bool = True,
) -> FilteredMolecules:
    """
    Convenience function for filtering molecules.

    Args:
        smiles_list: List of SMILES strings
        bbb_predictor: BBB prediction model
        min_qed: Minimum QED score
        min_bbb_prob: Minimum BBB probability
        max_sa: Maximum SA score
        training_smiles: Set of training SMILES for novelty check
        remove_duplicates: Remove duplicate molecules

    Returns:
        FilteredMolecules result
    """
    filter_obj = MoleculeFilter(
        min_qed=min_qed,
        min_bbb_prob=min_bbb_prob,
        max_sa_score=max_sa,
        training_smiles=training_smiles,
        remove_duplicates=remove_duplicates,
    )

    return filter_obj.filter(smiles_list, bbb_predictor)


def remove_duplicates(smiles_list: List[str]) -> List[str]:
    """
    Remove duplicate SMILES by canonicalization.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of unique SMILES
    """
    seen = set()
    unique = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)

        if canonical not in seen:
            seen.add(canonical)
            unique.append(smi)

    return unique


def compute_diversity(
    smiles_list: List[str],
    sample_size: int = 1000,
) -> dict:
    """
    Compute molecular diversity metrics.

    Args:
        smiles_list: List of SMILES strings
        sample_size: Maximum number of molecules to sample

    Returns:
        Dictionary with diversity metrics
    """
    if len(smiles_list) > sample_size:
        smiles_list = np.random.choice(smiles_list, sample_size, replace=False).tolist()

    # Compute fingerprints
    fps = []
    valid_smiles = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps.append(fp)
        valid_smiles.append(smi)

    if len(fps) < 2:
        return {
            'n_valid': len(valid_smiles),
            'mean_tanimoto': 0.0,
            'median_tanimoto': 0.0,
        }

    # Compute pairwise Tanimoto similarities
    n = len(fps)
    similarities = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)

    return {
        'n_valid': len(valid_smiles),
        'mean_tanimoto': np.mean(similarities),
        'median_tanimoto': np.median(similarities),
        'min_tanimoto': np.min(similarities),
        'max_tanimoto': np.max(similarities),
    }


def compute_scaffolds(smiles_list: List[str]) -> dict:
    """
    Compute Murcko scaffolds for molecules.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Dictionary with scaffold statistics
    """
    scaffolds = {}

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)

            if scaffold_smiles not in scaffolds:
                scaffolds[scaffold_smiles] = 0
            scaffolds[scaffold_smiles] += 1
        except:
            continue

    return {
        'n_unique_scaffolds': len(scaffolds),
        'most_common_scaffolds': sorted(
            scaffolds.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10],
    }


def load_training_smiles(
    split_path: Path,
    groups: str = "A,B",
) -> Set[str]:
    """
    Load training SMILES for novelty checking.

    Args:
        split_path: Path to split CSV
        groups: Groups to include

    Returns:
        Set of canonicalized SMILES
    """
    df = pd.read_csv(split_path)

    if 'SMILES' not in df.columns:
        return set()

    smiles_set = set()
    for smi in df['SMILES'].dropna():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol, canonical=True)
            smiles_set.add(canonical)

    return smiles_set
