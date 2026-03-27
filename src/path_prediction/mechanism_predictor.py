"""
Multi-Mechanism BBB Permeability Predictor

Implements separate XGBoost models for each transport mechanism:
- Passive diffusion (PAMPA)
- Active influx (SLC transporters)
- Active efflux (ABC transporters)
- Endothelial BBB (integrated)

Reference: Cornelissen et al., J. Med. Chem. 2022
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
except ImportError:
    logger.error("XGBoost not installed. Install: pip install xgboost")
    raise

try:
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )
    from sklearn.impute import SimpleImputer
except ImportError:
    logger.error("scikit-learn not installed. Install: pip install scikit-learn")
    raise

from .feature_extractor import MechanismFeatureExtractor


class MechanismPredictor:
    """
    Multi-mechanism BBB permeability predictor.

    Implements separate models for:
    1. BBB: Endothelial blood-brain barrier permeability
    2. PAMPA: Passive diffusion through artificial membrane
    3. Influx: SLC transporter uptake
    4. Efflux: ABC transporter efflux

    Each model predicts binary classification:
    - 1: Permeable / Substrate / BBB+
    - 0: Impermeable / Nonsubstrate / BBB-
    """

    # Mechanism names and descriptions
    MECHANISMS = {
        "bbb": {
            "name": "Blood-Brain Barrier",
            "description": "Endothelial BBB permeability (integrated)",
            "target": "BBB+ / BBB-",
        },
        "pampa": {
            "name": "Passive Diffusion",
            "description": "PAMPA assay (passive transcellular diffusion)",
            "target": "Permeable / Impermeable",
        },
        "influx": {
            "name": "Active Influx",
            "description": "SLC transporter-mediated uptake",
            "target": "Substrate / Nonsubstrate",
        },
        "efflux": {
            "name": "Active Efflux",
            "description": "ABC transporter-mediated efflux",
            "target": "Substrate / Nonsubstrate",
        },
    }

    # Key features for each mechanism (from literature)
    KEY_FEATURES = {
        "bbb": ["TPSA"],  # Topological polar surface area
        "pampa": ["LogD"],  # Lipophilicity at pH 7.4
        "influx": ["HBD", "MACCS43", "MACCS36"],  # Hydrogen bond donors
        "efflux": ["MW", "MACCS8"],  # Molecular weight, beta-lactam
    }

    def __init__(self, models_dir: str = "artifacts/models/mechanism"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.feature_extractors = {}
        self.top_maccs_keys = {}

        # Model hyperparameters (optimized for each mechanism)
        self.params = {
            "bbb": {
                "max_depth": 7,
                "n_estimators": 200,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "eval_metric": "logloss",
                "n_jobs": -1,
            },
            "pampa": {
                "max_depth": 7,
                "n_estimators": 200,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "eval_metric": "logloss",
                "n_jobs": -1,
            },
            "influx": {
                "max_depth": 5,
                "n_estimators": 100,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "eval_metric": "logloss",
                "n_jobs": -1,
            },
            "efflux": {
                "max_depth": 5,
                "n_estimators": 100,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "eval_metric": "logloss",
                "n_jobs": -1,
            },
        }

    def train_mechanism_model(
        self,
        mechanism: str,
        df: pd.DataFrame,
        smiles_col: str = "smiles",
        label_col: str = "label",
        feature_type: str = "combined",
        test_size: float = 0.2,
        identify_top_maccs: bool = True,
    ) -> Dict:
        """
        Train a model for a specific transport mechanism.

        Args:
            mechanism: 'bbb', 'pampa', 'influx', or 'efflux'
            df: Training dataset
            smiles_col: SMILES column name
            label_col: Label column name (0/1)
            feature_type: 'physicochemical', 'maccs', 'ecfp4', or 'combined'
            test_size: Fraction of data for testing
            identify_top_maccs: Whether to identify top MACCS keys

        Returns:
            Dictionary with training results
        """
        if mechanism not in self.MECHANISMS:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        logger.info(f"\nTraining {mechanism.upper()} model...")
        logger.info(f"Dataset: {len(df)} compounds")
        logger.info(f"Positive: {df[label_col].sum()} ({df[label_col].sum()/len(df)*100:.1f}%)")
        logger.info(f"Negative: {len(df)-df[label_col].sum()} ({(len(df)-df[label_col].sum())/len(df)*100:.1f}%)")

        # Initialize feature extractor
        extractor = MechanismFeatureExtractor()
        self.feature_extractors[mechanism] = extractor

        # Extract features
        logger.info(f"Extracting {feature_type} features...")
        if feature_type == "combined":
            # First extract all features to identify top MACCS keys
            X_physicochemical = extractor.calculate_physicochemical_features(
                df[smiles_col].tolist()
            )
            X_maccs = extractor.get_maccs_fingerprint(df[smiles_col].tolist())
            X_combined = np.hstack([X_physicochemical, X_maccs])

            # Identify top MACCS keys if requested
            if identify_top_maccs and len(df) > 100:
                logger.info("Identifying top MACCS keys...")
                top_keys = extractor.identify_important_maccs_keys(
                    X_combined, df[label_col].values, top_k=3
                )
                self.top_maccs_keys[mechanism] = top_keys

                # Extract features with only top MACCS keys
                X = extractor.get_combined_features(
                    df[smiles_col].tolist(), top_maccs_keys=top_keys
                )
            else:
                # Use all features
                X = X_combined
        else:
            X = extractor.extract_features_for_dataset(
                df, smiles_col=smiles_col, feature_type=feature_type
            )

        y = df[label_col].values

        # Handle missing values
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train model
        logger.info(f"Training XGBoost model...")
        params = self.params.get(mechanism, {})
        model = xgb.XGBClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        metrics = {
            "mechanism": mechanism,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "auc_roc": roc_auc_score(y_test, y_pred_proba),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        logger.info(f"\nResults for {mechanism.upper()}:")
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1']:.4f}")

        # Feature importance
        importance = model.feature_importances_
        feature_names = self._get_feature_names(mechanism, feature_type)

        feature_importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        logger.info(f"\nTop 10 Features:")
        for _, row in feature_importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Save model
        self.models[mechanism] = model
        self._save_model(mechanism, model, imputer, feature_importance_df)

        metrics["feature_importance"] = feature_importance_df.to_dict()

        return metrics

    def _get_feature_names(self, mechanism: str, feature_type: str) -> List[str]:
        """Get feature names for a mechanism."""
        extractor = MechanismFeatureExtractor()
        physicochemical_names = extractor.feature_names_physicochemical

        if feature_type == "physicochemical":
            return physicochemical_names
        elif feature_type == "maccs":
            return [f"MACCS_{i}" for i in range(167)]
        elif feature_type == "ecfp4":
            return [f"ECFP4_{i}" for i in range(1024)]
        elif feature_type == "combined":
            maccs_keys = self.top_maccs_keys.get(mechanism, list(range(167)))
            maccs_names = [f"MACCS_{i}" for i in maccs_keys]
            return physicochemical_names + maccs_names
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def _save_model(
        self, mechanism: str, model, imputer, feature_importance_df: pd.DataFrame
    ):
        """Save model and metadata."""
        # Save XGBoost model
        model_path = self.models_dir / f"{mechanism}_model.json"
        model.save_model(str(model_path))

        # Save imputer
        import joblib

        imputer_path = self.models_dir / f"{mechanism}_imputer.joblib"
        joblib.dump(imputer, imputer_path)

        # Save feature importance
        importance_path = self.models_dir / f"{mechanism}_feature_importance.csv"
        feature_importance_df.to_csv(importance_path, index=False)

        # Save metadata
        metadata = {
            "mechanism": mechanism,
            "top_maccs_keys": self.top_maccs_keys.get(mechanism, []),
            "params": self.params.get(mechanism, {}),
        }
        metadata_path = self.models_dir / f"{mechanism}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved {mechanism} model to {model_path}")

    def load_mechanism_model(self, mechanism: str):
        """Load a trained model."""
        if mechanism not in self.MECHANISMS:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        model_path = self.models_dir / f"{mechanism}_model.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load model
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        self.models[mechanism] = model

        # Load imputer
        import joblib

        imputer_path = self.models_dir / f"{mechanism}_imputer.joblib"
        imputer = joblib.load(imputer_path)

        # Load metadata
        metadata_path = self.models_dir / f"{mechanism}_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.top_maccs_keys[mechanism] = metadata.get("top_maccs_keys", [])

        # Initialize feature extractor
        self.feature_extractors[mechanism] = MechanismFeatureExtractor()

        logger.info(f"Loaded {mechanism} model from {model_path}")

        return model, imputer

    def predict_mechanism(
        self, smiles: str, mechanism: str
    ) -> Dict[str, Union[float, str, List]]:
        """
        Predict transport mechanism for a single molecule.

        Args:
            smiles: SMILES string
            mechanism: 'bbb', 'pampa', 'influx', or 'efflux'

        Returns:
            Dictionary with prediction results
        """
        if mechanism not in self.models:
            self.load_mechanism_model(mechanism)

        model = self.models[mechanism]
        extractor = self.feature_extractors[mechanism]

        # Extract features
        top_maccs = self.top_maccs_keys.get(mechanism, None)
        features = extractor.get_combined_features(smiles, top_maccs_keys=top_maccs)

        # Reshape for single sample
        features = features.reshape(1, -1)

        # Load imputer and transform
        import joblib

        imputer_path = self.models_dir / f"{mechanism}_imputer.joblib"
        imputer = joblib.load(imputer_path)
        features = imputer.transform(features)

        # Predict
        proba = model.predict_proba(features)[0, 1]
        pred = int(proba >= 0.5)

        # Get feature contributions
        importance = model.feature_importances_
        feature_names = self._get_feature_names(mechanism, "combined")

        contributions = dict(zip(feature_names, features[0] * importance))

        # Top contributing features
        top_features = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )[:5]

        return {
            "mechanism": mechanism,
            "prediction": pred,
            "probability": float(proba),
            "confidence": float(abs(proba - 0.5) * 2),  # 0 to 1 scale
            "top_features": top_features,
        }

    def predict_all_mechanisms(self, smiles: str) -> Dict:
        """
        Predict all transport mechanisms for a molecule.

        Returns:
            Dictionary with predictions for all mechanisms
        """
        results = {"smiles": smiles, "mechanisms": {}}

        for mechanism in self.MECHANISMS.keys():
            try:
                pred = self.predict_mechanism(smiles, mechanism)
                results["mechanisms"][mechanism] = pred
            except Exception as e:
                logger.error(f"Error predicting {mechanism}: {e}")
                results["mechanisms"][mechanism] = None

        # Determine dominant mechanism
        bbb_pred = results["mechanisms"].get("bbb", {}).get("prediction", 0)

        if bbb_pred == 1:
            # BBB+: Determine most likely mechanism
            passive_prob = results["mechanisms"].get("pampa", {}).get("probability", 0)
            influx_prob = results["mechanisms"].get("influx", {}).get("probability", 0)
            efflux_prob = results["mechanisms"].get("efflux", {}).get("probability", 0)

            if passive_prob > 0.5 and passive_prob >= influx_prob:
                dominant = "passive_diffusion"
                confidence = passive_prob
            elif influx_prob > 0.5 and influx_prob > efflux_prob:
                dominant = "active_influx"
                confidence = influx_prob
            elif efflux_prob > 0.5:
                dominant = "active_efflux"
                confidence = efflux_prob
            else:
                dominant = "mixed"
                confidence = max(passive_prob, influx_prob, efflux_prob)
        else:
            # BBB-: Check for efflux risk
            efflux_pred = results["mechanisms"].get("efflux", {}).get("prediction", 0)
            if efflux_pred == 1:
                dominant = "efflux_blocked"
                confidence = results["mechanisms"]["efflux"]["probability"]
            else:
                dominant = "impermeable"
                confidence = 1 - results["mechanisms"]["bbb"]["probability"]

        results["dominant_mechanism"] = dominant
        results["confidence"] = confidence

        return results

    def batch_predict(
        self, smiles_list: List[str], mechanism: str = "bbb"
    ) -> pd.DataFrame:
        """
        Batch predict for multiple molecules.

        Args:
            smiles_list: List of SMILES
            mechanism: Mechanism to predict

        Returns:
            DataFrame with predictions
        """
        results = []

        for smi in smiles_list:
            try:
                pred = self.predict_mechanism(smi, mechanism)
                pred["smiles"] = smi
                results.append(pred)
            except Exception as e:
                logger.error(f"Error predicting {smi}: {e}")
                results.append(
                    {"smiles": smi, "error": str(e), "prediction": None, "probability": None}
                )

        return pd.DataFrame(results)

    def generate_mechanism_report(self, smiles: str) -> str:
        """
        Generate a human-readable report for a molecule.

        Args:
            smiles: SMILES string

        Returns:
            Formatted report text
        """
        results = self.predict_all_mechanisms(smiles)

        report = []
        report.append("=" * 60)
        report.append("BBB TRANSPORT MECHANISM PREDICTION REPORT")
        report.append("=" * 60)
        report.append(f"\nSMILES: {smiles}")
        report.append(f"\nDominant Mechanism: {results['dominant_mechanism'].upper()}")
        report.append(f"Confidence: {results['confidence']:.2%}\n")

        report.append("-" * 60)
        report.append("MECHANISM-SPECIFIC PREDICTIONS:")
        report.append("-" * 60)

        for mech_name, mech_data in self.MECHANISMS.items():
            pred = results["mechanisms"].get(mech_name)
            if pred is None:
                continue

            report.append(f"\n{mech_data['name']} ({mech_data['description']}):")
            report.append(f"  Prediction: {'Positive' if pred['prediction'] == 1 else 'Negative'}")
            report.append(f"  Probability: {pred['probability']:.2%}")
            report.append(f"  Confidence: {pred['confidence']:.2%}")

            if pred.get("top_features"):
                report.append(f"  Top Contributing Features:")
                for feat, contrib in pred["top_features"][:3]:
                    report.append(f"    - {feat}: {contrib:.4f}")

        report.append("\n" + "=" * 60)

        # Recommendations
        report.append("\nRECOMMENDATIONS:")
        report.append("-" * 60)

        if results["dominant_mechanism"] == "passive_diffusion":
            report.append("✓ Good BBB permeability via passive diffusion")
            report.append("✓ Low efflux risk")
            report.append("✓ Optimize: Maintain low TPSA, moderate LogP")
        elif results["dominant_mechanism"] == "active_influx":
            report.append("✓ Good BBB permeability via transporter uptake")
            report.append("✓ May utilize SLC transporters")
            report.append("⚠ Consider: Transporter expression variability")
        elif results["dominant_mechanism"] == "active_efflux":
            report.append("✗ High efflux risk (ABC transporters)")
            report.append("✗ Likely substrate for P-gp or other efflux pumps")
            report.append("✗ Recommendations:")
            report.append("  - Reduce molecular weight")
            report.append("  - Remove beta-lactam or other efflux-triggering groups")
            report.append("  - Consider efflux pump inhibitors")
        elif results["dominant_mechanism"] == "efflux_blocked":
            report.append("✗ Poor BBB permeability due to active efflux")
            report.append("✗ Consider structural modifications")
        else:
            report.append("⚠ Mixed/uncertain mechanism")
            report.append("⚠ Experimental validation recommended")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


def main():
    """Test the mechanism predictor."""
    # Test molecules
    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin (passive diffusion)
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine (passive diffusion)
    ]

    predictor = MechanismPredictor()

    print("Testing MechanismPredictor")
    print("=" * 60)

    for smi in test_smiles:
        print(f"\nSMILES: {smi}")
        print("-" * 60)

        # Predict each mechanism
        for mechanism in ["bbb", "pampa", "influx", "efflux"]:
            try:
                result = predictor.predict_mechanism(smi, mechanism)
                print(f"{mechanism.upper()}: {result['prediction']} "
                      f"(prob={result['probability']:.2f})")
            except Exception as e:
                print(f"{mechanism.upper()}: Error - {e}")

        print("-" * 60)


if __name__ == "__main__":
    main()
