"""
Mechanism Predictor using Cornelissen et al. 2022 dataset.

This module provides a predictor for multiple transport mechanisms:
- BBB (Blood-Brain Barrier permeability)
- Influx (Active transport into brain)
- Efflux (Active transport out of brain / P-gp substrates)
- PAMPA (Parallel Artificial Membrane Permeability Assay)
- CNS (Central Nervous System activity)

Models trained on real experimental labels from Cornelissen et al., J. Med. Chem. 2022.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from xgboost import XGBClassifier

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Paths


class MechanismPredictor:
    """
    Predictor for BBB transport mechanisms using Cornelissen 2022 dataset.

    Example:
        >>> predictor = MechanismPredictor()
        >>> result = predictor.predict("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
        >>> print(result['BBB']['prediction'])  # True/False
        >>> print(result['BBB']['probability'])  # 0.85
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize the mechanism predictor.

        Args:
            model_dir: Directory containing trained models. If None, uses default path.
        """
        if model_dir is None:
            model_dir = Paths.artifacts / "models" / "cornelissen_2022"

        self.model_dir = Path(model_dir)
        self.models = {}
        self.feature_info = None
        self.all_features = None

        # Load models
        self._load_models()

    def _load_models(self):
        """Load all trained models."""
        mechanisms = ['bbb', 'influx', 'efflux', 'pampa', 'cns']

        for mech in mechanisms:
            model_file = self.model_dir / f"{mech}_model.json"
            if model_file.exists():
                model = XGBClassifier()
                model.load_model(str(model_file))
                self.models[mech] = model

        # Load feature info
        data_dir = Paths.root / "data" / "transport_mechanisms" / "cornelissen_2022"
        info_file = data_dir / "feature_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                self.feature_info = json.load(f)

            # Combine all features
            self.all_features = (self.feature_info['physicochemical'] +
                                self.feature_info['maccs'] +
                                self.feature_info['morgan'])

    def _extract_physicochemical_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """Extract physicochemical features from a molecule."""
        if mol is None:
            return {}

        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)

        features = {
            'LogP': logp,
            'TPSA': Descriptors.TPSA(mol),
            'MW': mw,
            'HBA': Descriptors.NumHAcceptors(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'RotatableBonds': Descriptors.NumRotatableBonds(mol),
            'RingCount': Descriptors.RingCount(mol),
            'AromaticRings': Descriptors.NumAromaticRings(mol),
            'SaturatedRings': Descriptors.NumSaturatedRings(mol),
            'Heteroatoms': Descriptors.NumHeteroatoms(mol),
            'HeavyAtoms': Descriptors.HeavyAtomCount(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'MolVolume': mw / logp if logp > 0 else 0,
        }
        return features

    def _extract_maccs_fingerprints(self, mol: Chem.Mol) -> np.ndarray:
        """Extract MACCS fingerprints (167 bits)."""
        if mol is None:
            return np.zeros(167)
        try:
            maccs = MACCSkeys.GenMACCSKeys(mol)
            return np.array(maccs)
        except:
            return np.zeros(167)

    def _extract_morgan_fingerprints(self, mol: Chem.Mol, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
        """Extract Morgan fingerprints (ECFP4-like)."""
        if mol is None:
            return np.zeros(n_bits)
        try:
            morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(morgan)
        except:
            return np.zeros(n_bits)

    def _extract_features(self, smiles: str) -> np.ndarray:
        """
        Extract all features from a SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Feature vector as numpy array
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Extract features
        physicochemical = self._extract_physicochemical_features(mol)
        maccs = self._extract_maccs_fingerprints(mol)
        morgan = self._extract_morgan_fingerprints(mol, radius=2, n_bits=1024)

        # Combine features
        feature_dict = {**physicochemical}

        # Add MACCS features
        for i, val in enumerate(maccs):
            feature_dict[f'MACCS_{i}'] = int(val)

        # Add Morgan features
        for i, val in enumerate(morgan):
            feature_dict[f'Morgan_{i}'] = int(val)

        # Create feature vector in correct order
        feature_vector = []
        for feat_name in self.all_features:
            feature_vector.append(feature_dict.get(feat_name, 0))

        return np.array(feature_vector).reshape(1, -1)

    def predict_mechanism(self, smiles: str, mechanism: str) -> Dict:
        """
        Predict a specific transport mechanism for a molecule.

        Args:
            smiles: SMILES string of the molecule
            mechanism: One of 'bbb', 'influx', 'efflux', 'pampa', 'cns'

        Returns:
            Dictionary with prediction results:
            {
                'prediction': bool,  # True if positive
                'probability': float,  # Probability of positive class
                'confidence': str,  # 'High', 'Medium', or 'Low'
            }
        """
        if mechanism not in self.models:
            raise ValueError(f"Unknown mechanism: {mechanism}. Available: {list(self.models.keys())}")

        # Extract features
        X = self._extract_features(smiles)

        # Predict
        model = self.models[mechanism]
        prediction = bool(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0, 1])

        # Determine confidence
        if probability >= 0.8 or probability <= 0.2:
            confidence = 'High'
        elif probability >= 0.6 or probability <= 0.4:
            confidence = 'Medium'
        else:
            confidence = 'Low'

        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': confidence,
        }

    def predict_all(self, smiles: str) -> Dict[str, Dict]:
        """
        Predict all transport mechanisms for a molecule.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Dictionary with results for all mechanisms:
            {
                'BBB': {...},
                'Influx': {...},
                'Efflux': {...},
                'PAMPA': {...},
                'CNS': {...},
            }
        """
        results = {}
        for mech in ['bbb', 'influx', 'efflux', 'pampa', 'cns']:
            try:
                results[mech.upper()] = self.predict_mechanism(smiles, mech)
            except Exception as e:
                results[mech.upper()] = {'error': str(e)}

        return results

    def predict_batch(self, smiles_list: List[str], mechanism: str) -> List[Dict]:
        """
        Predict mechanism for a batch of molecules.

        Args:
            smiles_list: List of SMILES strings
            mechanism: Mechanism to predict

        Returns:
            List of prediction results
        """
        results = []
        for smiles in smiles_list:
            try:
                result = self.predict_mechanism(smiles, mechanism)
                result['smiles'] = smiles
                results.append(result)
            except Exception as e:
                results.append({'smiles': smiles, 'error': str(e)})

        return results

    def get_feature_importance(self, mechanism: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get most important features for a mechanism.

        Args:
            mechanism: Mechanism name
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if mechanism not in self.models:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        model = self.models[mechanism]
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        return [(self.all_features[i], importances[i]) for i in indices]

    def get_model_info(self) -> Dict:
        """
        Get information about loaded models.

        Returns:
            Dictionary with model information
        """
        # Load training results
        results_file = self.model_dir / "training_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                training_results = json.load(f)
        else:
            training_results = {}

        return {
            'models_loaded': list(self.models.keys()),
            'n_features': len(self.all_features) if self.all_features else 0,
            'training_results': training_results,
        }


def main():
    """Example usage of the MechanismPredictor."""
    print("="*80)
    print("Mechanism Predictor - Example Usage")
    print("="*80)

    # Initialize predictor
    print("\n1. Loading models...")
    predictor = MechanismPredictor()
    print(f"   Models loaded: {list(predictor.models.keys())}")

    # Example molecules
    examples = [
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Dopamine", "NCCc1cc(O)c(O)cc1"),
        ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
        ("Glucose", "OCC1OC(O)C(O)C(O)C1O"),
    ]

    # Predict for each example
    print("\n2. Predicting mechanisms for example molecules:")
    print("-" * 80)

    for name, smiles in examples:
        print(f"\n{name} ({smiles}):")
        try:
            results = predictor.predict_all(smiles)

            for mech, res in results.items():
                if 'error' not in res:
                    pred_str = "+" if res['prediction'] else "-"
                    prob_str = f"{res['probability']:.2%}"
                    conf_str = res['confidence']
                    print(f"   {mech:6s}: {pred_str}  (Prob: {prob_str}, Conf: {conf_str})")
                else:
                    print(f"   {mech:6s}: Error - {res['error']}")
        except Exception as e:
            print(f"   Error: {e}")

    # Feature importance
    print("\n3. Top 5 important features for BBB:")
    print("-" * 40)
    try:
        bbb_features = predictor.get_feature_importance('bbb', top_n=5)
        for i, (feat, imp) in enumerate(bbb_features, 1):
            print(f"   {i}. {feat:<30} {imp:.4f}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "="*80)
    print("Example Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
