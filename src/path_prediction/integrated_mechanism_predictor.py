"""
Integrated Mechanism Predictor - Multi-model ensemble prediction.

This module uses multiple models to predict transport mechanisms:
1. Cornelissen 2022 models (trained on experimental labels)
2. Property-based heuristics (rule-based)
3. Ensemble of all approaches with confidence scores

For any given molecule, it provides:
- BBB permeability prediction
- Transport mechanism probabilities (Passive/Influx/Efflux)
- Confidence scores for each prediction
- Physicochemical property analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors
from xgboost import XGBClassifier

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Paths
from src.path_prediction.mechanism_predictor_cornelissen import MechanismPredictor


@dataclass
class MechanismPrediction:
    """Data class for mechanism prediction results."""
    mechanism: str
    probability: float
    confidence: str
    prediction: bool
    supporting_evidence: List[str]


class IntegratedMechanismPredictor:
    """
    Integrated predictor for BBB transport mechanisms.

    Combines multiple prediction approaches:
    1. ML models (trained on Cornelissen 2022)
    2. Property-based heuristics
    3. Consensus voting

    Example:
        >>> predictor = IntegratedMechanismPredictor()
        >>> result = predictor.predict_mechanisms("CC(=O)OC1=CC=CC=C1C(=O)O")
        >>> print(result['BBB']['probability'])  # 0.98
        >>> print(result['Passive_Diffusion']['probability'])  # 0.85
    """

    def __init__(self):
        """Initialize the integrated predictor."""
        print("Loading integrated mechanism predictor...")

        # Load ML predictor
        self.ml_predictor = MechanismPredictor()
        print(f"  - Loaded ML models: {list(self.ml_predictor.models.keys())}")

        # Property thresholds based on Cornelissen et al. 2022
        self.thresholds = {
            'passive': {
                'tpsa_max': 90,
                'logp_min': 1.0,
                'logp_max': 3.0,
                'mw_max': 500,
            },
            'influx': {
                'tpsa_min': 100,
                'hba_min': 5,
                'hbd_min': 2,
            },
            'efflux': {
                'mw_min': 500,
                'tpsa_min': 80,
            }
        }

        print("  - Loaded property thresholds")

    def _extract_properties(self, smiles: str) -> Dict[str, float]:
        """Extract physicochemical properties."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)

        return {
            'LogP': logp,
            'TPSA': Descriptors.TPSA(mol),
            'MW': mw,
            'HBA': Descriptors.NumHAcceptors(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'RotatableBonds': Descriptors.NumRotatableBonds(mol),
            'RingCount': Descriptors.RingCount(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'MolVolume': mw / logp if logp > 0 else 0,
        }

    def _heuristic_passive_diffusion(self, props: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Heuristic prediction for passive diffusion.

        Rules based on Cornelissen et al. 2022:
        - Low TPSA (<90)
        - Moderate LogP (1-3)
        - MW < 500
        """
        evidence = []
        score = 0.0

        # TPSA score (lower is better for passive diffusion)
        if props['TPSA'] < 60:
            score += 0.4
            evidence.append(f"Very low TPSA ({props['TPSA']:.1f} A^2) favors passive diffusion")
        elif props['TPSA'] < 90:
            score += 0.3
            evidence.append(f"Low TPSA ({props['TPSA']:.1f} A^2) supports passive diffusion")
        elif props['TPSA'] < 120:
            score += 0.1
            evidence.append(f"Moderate TPSA ({props['TPSA']:.1f} A^2)")

        # LogP score (1-3 is optimal)
        if 1.0 <= props['LogP'] <= 3.0:
            score += 0.3
            evidence.append(f"Optimal LogP ({props['LogP']:.2f}) for passive diffusion")
        elif 0.0 <= props['LogP'] < 1.0 or 3.0 < props['LogP'] <= 4.0:
            score += 0.1
            evidence.append(f"Acceptable LogP ({props['LogP']:.2f})")

        # MW score
        if props['MW'] < 400:
            score += 0.2
            evidence.append(f"Low MW ({props['MW']:.1f} Da) favors passive diffusion")
        elif props['MW'] < 500:
            score += 0.1
            evidence.append(f"Moderate MW ({props['MW']:.1f} Da)")

        # HBD/HBA
        if props['HBD'] <= 2 and props['HBA'] <= 4:
            score += 0.1
            evidence.append(f"Low HBD ({props['HBD']}) and HBA ({props['HBA']})")

        return min(score, 1.0), evidence

    def _heuristic_active_influx(self, props: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Heuristic prediction for active influx.

        Rules based on Cornelissen et al. 2022:
        - High TPSA (>100)
        - High HBA (>5)
        - May utilize nutrient transporters
        """
        evidence = []
        score = 0.0

        # TPSA score (higher is better for influx)
        if props['TPSA'] > 120:
            score += 0.4
            evidence.append(f"Very high TPSA ({props['TPSA']:.1f} A^2) suggests active influx")
        elif props['TPSA'] > 100:
            score += 0.3
            evidence.append(f"High TPSA ({props['TPSA']:.1f} A^2) indicates active transport")
        elif props['TPSA'] > 80:
            score += 0.1
            evidence.append(f"Elevated TPSA ({props['TPSA']:.1f} A^2)")

        # HBA score
        if props['HBA'] > 7:
            score += 0.3
            evidence.append(f"Very high HBA ({props['HBA']}) suggests transporter-mediated uptake")
        elif props['HBA'] > 5:
            score += 0.2
            evidence.append(f"High HBA ({props['HBA']}) may utilize transporters")

        # HBD score
        if props['HBD'] > 3:
            score += 0.2
            evidence.append(f"High HBD ({props['HBD']}) associated with active transport")

        # MW score (higher MW may require active transport)
        if props['MW'] > 450:
            score += 0.1
            evidence.append(f"High MW ({props['MW']:.1f} Da) may need active transport")

        return min(score, 1.0), evidence

    def _heuristic_active_efflux(self, props: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Heuristic prediction for active efflux (P-gp substrates).

        Rules based on Cornelissen et al. 2022:
        - High MW (>500)
        - Moderate to high TPSA
        """
        evidence = []
        score = 0.0

        # MW score (high MW is primary indicator)
        if props['MW'] > 600:
            score += 0.5
            evidence.append(f"Very high MW ({props['MW']:.1f} Da) strongly suggests efflux")
        elif props['MW'] > 500:
            score += 0.4
            evidence.append(f"High MW ({props['MW']:.1f} Da) indicates efflux risk")
        elif props['MW'] > 450:
            score += 0.2
            evidence.append(f"Elevated MW ({props['MW']:.1f} Da) may be efflux substrate")

        # TPSA score
        if props['TPSA'] > 100:
            score += 0.3
            evidence.append(f"High TPSA ({props['TPSA']:.1f} A^2) associated with efflux")
        elif props['TPSA'] > 80:
            score += 0.1
            evidence.append(f"Moderate TPSA ({props['TPSA']:.1f} A^2)")

        # LogP score (moderate-high LogP favors efflux)
        if props['LogP'] > 4:
            score += 0.2
            evidence.append(f"High LogP ({props['LogP']:.2f}) increases efflux likelihood")

        return min(score, 1.0), evidence

    def predict_mechanisms(self, smiles: str) -> Dict[str, Dict]:
        """
        Predict all transport mechanisms for a molecule.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Dictionary with predictions for all mechanisms:
            {
                'BBB': {'probability': 0.95, 'confidence': 'High', ...},
                'Passive_Diffusion': {'probability': 0.85, 'confidence': 'High', ...},
                'Active_Influx': {'probability': 0.15, 'confidence': 'High', ...},
                'Active_Efflux': {'probability': 0.30, 'confidence': 'Medium', ...},
                'properties': {...},
            }
        """
        result = {}

        # Extract properties
        try:
            props = self._extract_properties(smiles)
            result['properties'] = props
        except Exception as e:
            return {'error': str(e)}

        # 1. ML predictions (from Cornelissen models)
        ml_predictions = {}
        try:
            # BBB prediction
            bbb_result = self.ml_predictor.predict_mechanism(smiles, 'bbb')
            ml_predictions['BBB'] = bbb_result['probability']

            # PAMPA (proxy for passive diffusion)
            pampa_result = self.ml_predictor.predict_mechanism(smiles, 'pampa')
            ml_predictions['PAMPA'] = pampa_result['probability']

            # Influx
            influx_result = self.ml_predictor.predict_mechanism(smiles, 'influx')
            ml_predictions['Influx'] = influx_result['probability']

            # Efflux
            efflux_result = self.ml_predictor.predict_mechanism(smiles, 'efflux')
            ml_predictions['Efflux'] = efflux_result['probability']

        except Exception as e:
            return {'error': f"ML prediction failed: {str(e)}"}

        # 2. Heuristic predictions
        heuristic_passive, passive_evidence = self._heuristic_passive_diffusion(props)
        heuristic_influx, influx_evidence = self._heuristic_active_influx(props)
        heuristic_efflux, efflux_evidence = self._heuristic_active_efflux(props)

        # 3. Combine ML and heuristic predictions
        # BBB: use ML directly
        result['BBB'] = {
            'probability': ml_predictions['BBB'],
            'prediction': ml_predictions['BBB'] > 0.5,
            'confidence': self._get_confidence(ml_predictions['BBB']),
            'supporting_evidence': self._get_bbb_evidence(props, ml_predictions['BBB']),
        }

        # Passive Diffusion: weighted average of PAMPA (ML) and heuristic
        passive_prob = 0.6 * ml_predictions['PAMPA'] + 0.4 * heuristic_passive
        result['Passive_Diffusion'] = {
            'probability': passive_prob,
            'prediction': passive_prob > 0.5,
            'confidence': self._get_confidence(passive_prob),
            'supporting_evidence': passive_evidence,
            'ml_probability': ml_predictions['PAMPA'],
            'heuristic_probability': heuristic_passive,
        }

        # Active Influx: weighted average
        influx_prob = 0.7 * ml_predictions['Influx'] + 0.3 * heuristic_influx
        result['Active_Influx'] = {
            'probability': influx_prob,
            'prediction': influx_prob > 0.5,
            'confidence': self._get_confidence(influx_prob),
            'supporting_evidence': influx_evidence,
            'ml_probability': ml_predictions['Influx'],
            'heuristic_probability': heuristic_influx,
        }

        # Active Efflux: weighted average
        efflux_prob = 0.7 * ml_predictions['Efflux'] + 0.3 * heuristic_efflux
        result['Active_Efflux'] = {
            'probability': efflux_prob,
            'prediction': efflux_prob > 0.5,
            'confidence': self._get_confidence(efflux_prob),
            'supporting_evidence': efflux_evidence,
            'ml_probability': ml_predictions['Efflux'],
            'heuristic_probability': heuristic_efflux,
        }

        # 4. Overall mechanism assessment
        result['mechanism_summary'] = self._generate_mechanism_summary(result)

        return result

    def _get_confidence(self, probability: float) -> str:
        """Get confidence level from probability."""
        if probability >= 0.8 or probability <= 0.2:
            return 'High'
        elif probability >= 0.6 or probability <= 0.4:
            return 'Medium'
        else:
            return 'Low'

    def _get_bbb_evidence(self, props: Dict[str, float], probability: float) -> List[str]:
        """Generate evidence for BBB prediction."""
        evidence = []

        if probability > 0.7:
            if props['TPSA'] < 90:
                evidence.append(f"Low TPSA ({props['TPSA']:.1f} A^2) favors BBB penetration")
            if props['MW'] < 450:
                evidence.append(f"Moderate MW ({props['MW']:.1f} Da) suitable for BBB")
            if 1.0 <= props['LogP'] <= 3.5:
                evidence.append(f"Optimal LogP ({props['LogP']:.2f}) for BBB penetration")
            if props['HBD'] <= 3:
                evidence.append(f"Low HBD ({props['HBD']}) supports BBB penetration")

        elif probability < 0.3:
            if props['TPSA'] > 90:
                evidence.append(f"High TPSA ({props['TPSA']:.1f} A^2) hinders BBB penetration")
            if props['MW'] > 500:
                evidence.append(f"High MW ({props['MW']:.1f} Da) limits BBB penetration")
            if props['HBA'] > 8:
                evidence.append(f"High HBA ({props['HBA']}) reduces BBB permeability")

        return evidence

    def _generate_mechanism_summary(self, result: Dict) -> Dict:
        """Generate overall mechanism assessment."""
        mechanisms = ['Passive_Diffusion', 'Active_Influx', 'Active_Efflux']
        probs = {m: result[m]['probability'] for m in mechanisms}

        # Find dominant mechanism
        dominant = max(probs.items(), key=lambda x: x[1])

        # Check if any mechanism is clearly dominant
        max_prob = dominant[1]
        second_max = sorted(probs.values(), reverse=True)[1] if len(probs) > 1 else 0

        if max_prob > 0.6 and (max_prob - second_max) > 0.2:
            primary_mechanism = dominant[0]
            certainty = 'High'
        elif max_prob > 0.5:
            primary_mechanism = dominant[0]
            certainty = 'Medium'
        else:
            primary_mechanism = 'Mixed/Uncertain'
            certainty = 'Low'

        return {
            'primary_mechanism': primary_mechanism,
            'certainty': certainty,
            'probability_breakdown': probs,
        }

    def predict_batch(self, smiles_list: List[str]) -> List[Dict]:
        """Predict mechanisms for a batch of molecules."""
        results = []
        for smiles in smiles_list:
            try:
                result = self.predict_mechanisms(smiles)
                result['SMILES'] = smiles
                results.append(result)
            except Exception as e:
                results.append({
                    'SMILES': smiles,
                    'error': str(e)
                })
        return results

    def print_prediction(self, smiles: str):
        """Print a formatted prediction result."""
        result = self.predict_mechanisms(smiles)

        if 'error' in result:
            print(f"Error: {result['error']}")
            return

        print("\n" + "="*80)
        print(f"Mechanism Prediction for: {smiles}")
        print("="*80)

        # Properties
        print("\nPhysicochemical Properties:")
        props = result['properties']
        print(f"  MW:   {props['MW']:.1f} Da")
        print(f"  TPSA: {props['TPSA']:.1f} A^2")
        print(f"  LogP: {props['LogP']:.2f}")
        print(f"  HBA:  {props['HBA']}")
        print(f"  HBD:  {props['HBD']}")

        # BBB
        print(f"\nBBB Permeability:")
        bbb = result['BBB']
        status = "+" if bbb['prediction'] else "-"
        print(f"  Prediction: BBB{status} (Prob: {bbb['probability']:.2%}, Conf: {bbb['confidence']})")
        if bbb['supporting_evidence']:
            print(f"  Evidence:")
            for ev in bbb['supporting_evidence'][:3]:
                print(f"    - {ev}")

        # Mechanisms
        print(f"\nTransport Mechanisms:")
        for mech in ['Passive_Diffusion', 'Active_Influx', 'Active_Efflux']:
            pred = result[mech]
            status = "+" if pred['prediction'] else "-"
            print(f"  {mech:20s}: {status} (Prob: {pred['probability']:.2%}, Conf: {pred['confidence']})")

            # Show breakdown
            if 'ml_probability' in pred:
                print(f"    └─ ML: {pred['ml_probability']:.2%}, Heuristic: {pred['heuristic_probability']:.2%}")

            # Show top evidence
            if pred['supporting_evidence']:
                for ev in pred['supporting_evidence'][:2]:
                    print(f"    - {ev}")

        # Summary
        print(f"\nOverall Assessment:")
        summary = result['mechanism_summary']
        print(f"  Primary Mechanism: {summary['primary_mechanism']}")
        print(f"  Certainty: {summary['certainty']}")

        print("="*80 + "\n")


def main():
    """Example usage."""
    print("="*80)
    print("Integrated Mechanism Predictor - Example")
    print("="*80)

    # Initialize predictor
    predictor = IntegratedMechanismPredictor()

    # Example molecules
    examples = [
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
        ("Glucose", "OCC1OC(O)C(O)C(O)C1O"),
        ("Morphine", "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"),
        ("Dopamine", "NCCc1cc(O)c(O)cc1"),
    ]

    # Predict for each
    for name, smiles in examples:
        print(f"\n{'='*80}")
        print(f"{name}: {smiles}")
        predictor.print_prediction(smiles)


if __name__ == "__main__":
    main()
