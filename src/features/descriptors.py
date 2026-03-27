"""
Physicochemical descriptor computation using RDKit.

Supports basic, extended, and full descriptor sets.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from sklearn.preprocessing import StandardScaler


class DescriptorGenerator:
    """
    Generate molecular descriptors from SMILES.

    Supports multiple descriptor sets with different levels
    of detail.
    """

    def __init__(self, descriptor_set: Literal["basic", "extended", "all"] = "all"):
        """
        Initialize descriptor generator.

        Args:
            descriptor_set: Which descriptors to compute
                - "basic": 13 core descriptors
                - "extended": ~100 descriptors
                - "all": 200+ descriptors
        """
        self.descriptor_set = descriptor_set
        self._descriptor_list = self._get_descriptor_list()
        self.scaler = None

    def _get_descriptor_list(self):
        """Get list of (name, function) tuples for selected set."""
        # Define descriptor groups
        basic_desc = [
            ("MolWt", Descriptors.MolWt),
            ("MolLogP", Crippen.MolLogP),
            ("TPSA", rdMolDescriptors.CalcTPSA),
            ("NumHDonors", Lipinski.NumHDonors),
            ("NumHAcceptors", Lipinski.NumHAcceptors),
            ("NumRotatableBonds", Lipinski.NumRotatableBonds),
            ("HeavyAtomCount", Lipinski.HeavyAtomCount),
            ("RingCount", Lipinski.RingCount),
            ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
            ("MaxEStateIndex", Descriptors.MaxEStateIndex),
            ("MinEStateIndex", Descriptors.MinEStateIndex),
            ("BalabanJ", Descriptors.BalabanJ),
            ("BertzCT", Descriptors.BertzCT),
        ]

        # Extended descriptors (add more for detailed analysis)
        extended_desc = basic_desc + [
            ("Chi0", Descriptors.Chi0),
            ("Chi1", Descriptors.Chi1),
            ("Chi0n", Descriptors.Chi0n),
            ("Chi1n", Descriptors.Chi1n),
            ("Kappa1", Descriptors.Kappa1),
            ("Kappa2", Descriptors.Kappa2),
            ("Kappa3", Descriptors.Kappa3),
            ("LabuteASA", Descriptors.LabuteASA),
            ("PEOE_VSA1", Descriptors.PEOE_VSA1),
            ("PEOE_VSA2", Descriptors.PEOE_VSA2),
            ("SMR_VSA1", Descriptors.SMR_VSA1),
            ("SMR_VSA2", Descriptors.SMR_VSA2),
            ("SlogP_VSA1", Descriptors.SlogP_VSA1),
            ("SlogP_VSA2", Descriptors.SlogP_VSA2),
            ("NumAromaticRings", Lipinski.NumAromaticRings),
            ("NumAliphaticRings", Lipinski.NumAliphaticRings),
            ("NumHeteroatoms", Descriptors.NumHeteroatoms),
        ]

        # All descriptors (comprehensive set)
        all_desc = extended_desc + [
            ("MaxPartialCharge", Descriptors.MaxPartialCharge),
            ("MinPartialCharge", Descriptors.MinPartialCharge),
            ("MaxAbsEStateIndex", Descriptors.MaxAbsEStateIndex),
            ("MinAbsEStateIndex", Descriptors.MinAbsEStateIndex),
            ("NumValenceElectrons", Descriptors.NumValenceElectrons),
            ("MolMR", Descriptors.MolMR),
            ("HallKierAlpha", Descriptors.HallKierAlpha),
            ("Chi0v", Descriptors.Chi0v),
            ("Chi1v", Descriptors.Chi1v),
            ("Chi2v", Descriptors.Chi2v),
            ("Chi2n", Descriptors.Chi2n),
            ("Chi3n", Descriptors.Chi3n),
            ("Chi4n", Descriptors.Chi4n),
            ("Chi3v", Descriptors.Chi3v),
            ("Chi4v", Descriptors.Chi4v),
        ]

        if self.descriptor_set == "basic":
            return basic_desc
        elif self.descriptor_set == "extended":
            return extended_desc
        else:  # "all"
            return all_desc

    def compute(self, smiles_list: list[str]) -> pd.DataFrame:
        """
        Compute descriptors for SMILES list.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            DataFrame with descriptor values
        """
        rows = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append({name: np.nan for name, _ in self._descriptor_list})
                continue

            row = {}
            for name, fn in self._descriptor_list:
                try:
                    val = float(fn(mol))
                    if np.isnan(val) or np.isinf(val):
                        row[name] = np.nan
                    else:
                        row[name] = val
                except Exception:
                    row[name] = np.nan
            rows.append(row)

        return pd.DataFrame(rows)

    def fit_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit normalizer and transform descriptors.

        Args:
            df: Descriptor DataFrame

        Returns:
            Normalized DataFrame
        """
        # Handle missing values
        df_clean = df.fillna(df.median())
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(0)

        self.scaler = StandardScaler()
        normalized = self.scaler.fit_transform(df_clean)

        return pd.DataFrame(normalized, columns=df.columns)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform descriptors using fitted normalizer.

        Args:
            df: Descriptor DataFrame

        Returns:
            Normalized DataFrame
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_normalize first.")

        # Handle missing values
        df_clean = df.fillna(df.median())
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(0)

        normalized = self.scaler.transform(df_clean)

        return pd.DataFrame(normalized, columns=df.columns)

    def get_descriptor_names(self) -> list[str]:
        """Get names of descriptors in this set."""
        return [name for name, _ in self._descriptor_list]

    def get_descriptor_count(self) -> int:
        """Get number of descriptors."""
        return len(self._descriptor_list)
