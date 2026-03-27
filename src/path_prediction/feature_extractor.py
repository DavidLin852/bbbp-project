"""
Mechanism-Specific Feature Extractor

Extracts physicochemical properties and molecular fingerprints
for transport mechanism prediction.

Features:
1. Physicochemical: TPSA, MW, LogD, LogS, LogP, HBD, HBA, Rotatable Bonds
2. MACCS keys: 166-bit structural fingerprints
3. ECFP4: 1024-bit circular fingerprints
4. Combined: Physicochemical + Top-3 MACCS keys

Reference: Cornelissen et al., J. Med. Chem. 2022
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, MACCSkeys, AllChem, DataStructs
    from rdkit.Chem.Crippen import MolLogP
except ImportError:
    logger.error("RDKit not installed. Please install: pip install rdkit")
    raise


class MechanismFeatureExtractor:
    """
    Extracts features for transport mechanism prediction.

    Features types:
    - Physicochemical properties (8 features)
    - MACCS keys (166 bits)
    - ECFP4 fingerprints (1024 bits)
    - Combined features (physicochemical + top MACCS)
    """

    # MACCS key to SMARTS mapping (selected important keys)
    MACCS_KEY_NAMES = {
        8: "Four-membered ring (beta-lactam)",
        36: "Sulfur heterocycle (thiolane)",
        43: "Two amino groups connected by carbon",
        63: "Aromatic ring (>1)",
        64: "Aromatic ring (>=2)",
        96: "COO group",
        99: "Phenol",
        103: "Primary amine",
        107: "Secondary amine",
        114: "Tertiary amine",
        121: "Halogen",
        129: "Ring count",
        136: "Ring system",
        140: "Aromatic carbon count",
        144: "Carbonyl",
        154: "Heteroatom in ring",
        165: "Heterocycle",
    }

    def __init__(self):
        self.feature_names_physicochemical = [
            "TPSA",
            "MW",
            "LogP",
            "LogD",
            "HBD",
            "HBA",
            "RotBonds",
        ]

    def calculate_physicochemical_features(
        self, smiles: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Calculate physicochemical properties for SMILES.

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            Array of shape (n_samples, 7) with features:
            - TPSA: Topological polar surface area
            - MW: Molecular weight
            - LogP: Partition coefficient (octanol/water)
            - LogD: Distribution coefficient at pH 7.4
            - HBD: Hydrogen bond donors
            - HBA: Hydrogen bond acceptors
            - RotBonds: Rotatable bonds
        """
        single_input = isinstance(smiles, str)
        if single_input:
            smiles = [smiles]

        features = []

        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    # Invalid SMILES
                    features.append([np.nan] * 7)
                    continue

                # Calculate descriptors
                tpsa = Descriptors.TPSA(mol)
                mw = Descriptors.ExactMolWt(mol)
                logp = Descriptors.MolLogP(mol)
                logd = self._calculate_logd(mol)  # LogD at pH 7.4
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                rotbonds = Descriptors.NumRotatableBonds(mol)

                features.append([tpsa, mw, logp, logd, hbd, hba, rotbonds])

            except Exception as e:
                logger.debug(f"Error calculating features for {smi}: {e}")
                features.append([np.nan] * 8)

        features_array = np.array(features)

        if single_input:
            return features_array[0]
        return features_array

    def _calculate_logd(self, mol: Chem.Mol, ph: float = 7.4) -> float:
        """
        Calculate LogD (distribution coefficient) at pH 7.4.

        LogD accounts for ionization state, unlike LogP.
        Simplified calculation using ChemAxon-style approximation.

        For a more accurate calculation, consider using:
        - ChemAxon Marvin
        - ACD Labs Percepta
        - OpenBabel OBLogD
        """
        try:
            # Get LogP (neutral form)
            logp = Descriptors.MolLogP(mol)

            # Count acidic/basic groups
            from rdkit.Chem import rdMolDescriptors

            pka = rdMolDescriptors.CalcExactMolWt(mol)  # Placeholder

            # Simplified: if molecule has ionizable groups, adjust LogP
            # This is a rough approximation
            # For accurate LogD, use specialized tools

            # Check for carboxylic acids (acidic)
            acid_pattern = Chem.MolFromSmarts("C(=O)[OH]")
            acids = len(mol.GetSubstructMatches(acid_pattern))

            # Check for amines (basic)
            base_pattern = Chem.MolFromSmarts("[NH2]")
            bases = len(mol.GetSubstructMatches(base_pattern))

            # Adjust LogD based on pH and pKa
            # At pH 7.4:
            # - Acids (pKa ~4-5) are deprotonated (more hydrophilic)
            # - Bases (pKa ~9-10) are protonated (more hydrophilic)

            logd = logp
            logd -= acids * 2.0  # Acids decrease LogD
            logd -= bases * 1.0  # Bases decrease LogD

            return logd

        except Exception:
            # Fallback to LogP
            return Descriptors.MolLogP(mol)

    def get_maccs_fingerprint(
        self, smiles: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Generate MACCS keys fingerprint (167 bits from RDKit).

        Note: RDKit's MACCS implementation returns 167 bits (includes 1 redundant bit).
        The first bit (index 0) is always 0 and can be ignored if needed.

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            Array of shape (n_samples, 167)
        """
        single_input = isinstance(smiles, str)
        if single_input:
            smiles = [smiles]

        fingerprints = []

        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    fingerprints.append(np.zeros(167))
                    continue

                maccs = MACCSkeys.GenMACCSKeys(mol)
                # Convert ExplicitBitVect to numpy array
                fp_array = np.zeros((167,), dtype=np.int32)
                DataStructs.ConvertToNumpyArray(maccs, fp_array)
                fingerprints.append(fp_array)

            except Exception as e:
                logger.debug(f"Error generating MACCS for {smi}: {e}")
                fingerprints.append(np.zeros(167))

        fp_array = np.stack(fingerprints)

        if single_input:
            return fp_array[0]
        return fp_array

    def get_ecfp4_fingerprint(
        self, smiles: Union[str, List[str]], n_bits: int = 1024
    ) -> np.ndarray:
        """
        Generate ECFP4 (Morgan) fingerprint.

        Args:
            smiles: Single SMILES string or list of SMILES
            n_bits: Length of fingerprint (default 1024)

        Returns:
            Array of shape (n_samples, n_bits)
        """
        single_input = isinstance(smiles, str)
        if single_input:
            smiles = [smiles]

        fingerprints = []

        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    fingerprints.append(np.zeros(n_bits))
                    continue

                ecfp4 = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=n_bits
                )
                fp_array = np.array(ecfp4)
                fingerprints.append(fp_array)

            except Exception as e:
                logger.debug(f"Error generating ECFP4 for {smi}: {e}")
                fingerprints.append(np.zeros(n_bits))

        fp_array = np.stack(fingerprints)

        if single_input:
            return fp_array[0]
        return fp_array

    def get_combined_features(
        self,
        smiles: Union[str, List[str]],
        top_maccs_keys: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Generate combined features: physicochemical + selected MACCS keys.

        Args:
            smiles: Single SMILES string or list of SMILES
            top_maccs_keys: Indices of top MACCS keys to include
                          (default: use top 3 from training)

        Returns:
            Array of shape (n_samples, 8 + n_maccs)
        """
        # Get physicochemical features
        physicochemical = self.calculate_physicochemical_features(smiles)

        # Get all MACCS keys
        maccs_all = self.get_maccs_fingerprint(smiles)

        # Select specific MACCS keys
        if top_maccs_keys is None:
            # Default: use all MACCS keys
            maccs_selected = maccs_all
        else:
            maccs_selected = maccs_all[:, top_maccs_keys]

        # Combine
        if len(physicochemical.shape) == 1:
            combined = np.concatenate([physicochemical, maccs_selected])
        else:
            combined = np.hstack([physicochemical, maccs_selected])

        return combined

    def extract_features_for_dataset(
        self,
        df: pd.DataFrame,
        smiles_col: str = "smiles",
        feature_type: str = "combined",
        top_maccs_keys: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Extract features for a dataset.

        Args:
            df: DataFrame with SMILES
            smiles_col: Name of SMILES column
            feature_type: 'physicochemical', 'maccs', 'ecfp4', or 'combined'
            top_maccs_keys: For 'combined' features, which MACCS keys to include

        Returns:
            Feature array
        """
        smiles_list = df[smiles_col].tolist()

        if feature_type == "physicochemical":
            features = self.calculate_physicochemical_features(smiles_list)
        elif feature_type == "maccs":
            features = self.get_maccs_fingerprint(smiles_list)
        elif feature_type == "ecfp4":
            features = self.get_ecfp4_fingerprint(smiles_list)
        elif feature_type == "combined":
            features = self.get_combined_features(smiles_list, top_maccs_keys)
        else:
            raise ValueError(
                f"Unknown feature type: {feature_type}. "
                "Use 'physicochemical', 'maccs', 'ecfp4', or 'combined'"
            )

        return features

    def identify_important_maccs_keys(
        self, X: np.ndarray, y: np.ndarray, top_k: int = 10
    ) -> List[int]:
        """
        Identify most important MACCS keys using feature importance.

        Args:
            X: Feature array (must include MACCS keys)
            y: Labels
            top_k: Number of top keys to return

        Returns:
            List of MACCS key indices (sorted by importance)
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer

        # Impute missing values
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        # Train random forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_imputed, y)

        # Get feature importance
        importance = rf.feature_importances_

        # Assume MACCS keys are after physicochemical features (first 8)
        maccs_importance = importance[8:]
        top_indices = np.argsort(maccs_importance)[-top_k:][::-1]

        # Convert to original MACCS key indices (0-based)
        top_maccs_keys = [int(i) for i in top_indices]

        logger.info(f"Top {top_k} MACCS keys: {top_maccs_keys}")
        for idx in top_maccs_keys:
            logger.info(
                f"  MACCS {idx}: {self.MACC_KEY_NAMES.get(idx, 'Unknown')} - "
                f"importance: {maccs_importance[idx]:.4f}"
            )

        return top_maccs_keys

    def create_feature_dataframe(
        self, smiles: Union[str, List[str]], feature_type: str = "combined"
    ) -> pd.DataFrame:
        """
        Create a DataFrame with features and feature names.

        Useful for visualization and interpretation.

        Args:
            smiles: SMILES string or list
            feature_type: Type of features to extract

        Returns:
            DataFrame with features
        """
        single_input = isinstance(smiles, str)
        if single_input:
            smiles = [smiles]

        if feature_type == "physicochemical":
            features = self.calculate_physicochemical_features(smiles)
            columns = self.feature_names_physicochemical
        elif feature_type == "maccs":
            features = self.get_maccs_fingerprint(smiles)
            columns = [f"MACCS_{i}" for i in range(167)]
        elif feature_type == "ecfp4":
            features = self.get_ecfp4_fingerprint(smiles)
            columns = [f"ECFP4_{i}" for i in range(1024)]
        elif feature_type == "combined":
            features = self.get_combined_features(smiles)
            physicochemical_names = self.feature_names_physicochemical
            maccs_names = [f"MACCS_{i}" for i in range(167)]
            columns = physicochemical_names + maccs_names
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        df = pd.DataFrame(features, columns=columns)
        df.insert(0, "SMILES", smiles)

        return df


def main():
    """Test feature extraction."""
    # Test molecules
    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin (passive diffusion)
        "CC(C)CC1=CC=C(C=C1)C(C)C(O)=O",  # Ibuprofen (passive diffusion)
        "CC1=CN=C(C(=N1)N)N",  # Caffeine (passive diffusion)
        "CC1C(N)C(C(=O)O)C(N)C1O",  # Beta-lactam (efflux risk - MACCS8)
    ]

    extractor = MechanismFeatureExtractor()

    print("Testing MechanismFeatureExtractor")
    print("=" * 50)

    # Test physicochemical features
    print("\n1. Physicochemical Features:")
    physicochemical = extractor.calculate_physicochemical_features(test_smiles)
    print(physicochemical)

    # Test MACCS
    print("\n2. MACCS Fingerprints:")
    maccs = extractor.get_maccs_fingerprint(test_smiles)
    print(f"Shape: {maccs.shape}")
    print(f"Sample (Aspirin): {maccs[0][:20]}...")  # First 20 bits

    # Check for MACCS8 (beta-lactam)
    print("\n3. Beta-lactam Check (MACCS8):")
    for smi, fp in zip(test_smiles, maccs):
        print(f"{smi[:30]}: MACCS8 = {int(fp[7])}")

    # Test combined features
    print("\n4. Combined Features:")
    combined = extractor.get_combined_features(test_smiles)
    print(f"Shape: {combined.shape}")

    # Create DataFrame
    print("\n5. Feature DataFrame:")
    df = extractor.create_feature_dataframe(test_smiles, feature_type="physicochemical")
    print(df)


if __name__ == "__main__":
    main()
