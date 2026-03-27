"""
Train All Transport Mechanism Models

This script trains XGBoost models for each transport mechanism:
1. BBB: Endothelial blood-brain barrier permeability
2. PAMPA: Passive diffusion
3. Influx: SLC transporter uptake
4. Efflux: ABC transporter efflux

Usage:
    python scripts/mechanism_training/train_all_mechanism_models.py

Author: BBB Prediction Project
Reference: Cornelissen et al., J. Med. Chem. 2022
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.path_prediction.data_collector import TransportDataCollector
from src.path_prediction.mechanism_predictor import MechanismPredictor
from src.config import Paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_with_b3db_synthetic_labels():
    """
    Train models using synthetic transport labels from B3DB.

    This is the fallback approach when external transport data is unavailable.
    Uses physicochemical properties to predict likely transport mechanism.
    """
    logger.info("Training with B3DB synthetic labels...")

    # Load B3DB data
    b3db_path = Paths.data_dir / "B3DB" / "B3DB.csv"

    if not b3db_path.exists():
        logger.error(f"B3DB not found at {b3db_path}")
        logger.info("Please download B3DB from: https://github.com/theochem/B3DB")
        return None

    df_b3db = pd.read_csv(b3db_path)

    logger.info(f"Loaded B3DB: {len(df_b3db)} compounds")

    # Create synthetic labels
    collector = TransportDataCollector()
    df_synthetic = collector.create_synthetic_labels_from_b3db(str(b3db_path))

    if df_synthetic.empty:
        logger.error("Failed to create synthetic labels")
        return None

    # Train BBB model (original BBB permeability)
    logger.info("\n" + "=" * 60)
    logger.info("Training BBB permeability model...")
    logger.info("=" * 60)

    predictor = MechanismPredictor()

    # Create BBB dataset
    df_bbb = df_synthetic[["smiles", "bbb_label"]].copy()
    df_bbb = df_bbb.rename(columns={"bbb_label": "label"})
    df_bbb = df_bbb.dropna()

    if len(df_bbb) > 0:
        bbb_metrics = predictor.train_mechanism_model(
            mechanism="bbb",
            df=df_bbb,
            feature_type="combined",
            identify_top_maccs=True,
        )
        logger.info(f"BBB Model AUC: {bbb_metrics['auc_roc']:.4f}")

    # Train mechanism-specific models using synthetic labels
    for mechanism_name, mechanism_label in [
        ("pampa", "passive"),
        ("influx", "influx"),
        ("efflux", "efflux"),
    ]:
        logger.info("\n" + "=" * 60)
        logger.info(f"Training {mechanism_name.upper()} model...")
        logger.info("=" * 60)

        # Create dataset for this mechanism
        mask = df_synthetic["mechanism"] == mechanism_label
        df_mech = df_synthetic[mask].copy()

        if len(df_mech) < 50:
            logger.warning(
                f"Insufficient data for {mechanism_name}: {len(df_mech)} compounds"
            )
            continue

        # For synthetic labels, use mechanism type as positive class
        # All other mechanisms as negative class
        df_train = pd.concat(
            [
                df_synthetic[mask][["smiles"]].assign(label=1),
                df_synthetic[~mask][["smiles"]].assign(label=0),
            ]
        ).dropna()

        if len(df_train) > 0:
            metrics = predictor.train_mechanism_model(
                mechanism=mechanism_name,
                df=df_train,
                feature_type="combined",
                identify_top_maccs=True,
            )
            logger.info(f"{mechanism_name.upper()} Model AUC: {metrics['auc_roc']:.4f}")

    return predictor


def train_with_curated_datasets(data_dir: str = "data/transport_mechanisms/curated"):
    """
    Train models using curated transport mechanism datasets.

    Args:
        data_dir: Directory with curated CSV files
    """
    logger.info("Training with curated transport datasets...")

    predictor = MechanismPredictor()

    # Check for curated datasets
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"Curated data directory not found: {data_dir}")
        logger.info("Run data collection first:")
        logger.info("  python -m src.path_prediction.data_collector")
        return None

    # Train each mechanism
    for mechanism in ["pampa", "influx", "efflux"]:
        data_file = data_path / f"{mechanism}_curated.csv"

        if not data_file.exists():
            logger.warning(f"Data file not found: {data_file}")
            continue

        logger.info("\n" + "=" * 60)
        logger.info(f"Training {mechanism.upper()} model...")
        logger.info("=" * 60)

        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} compounds from {data_file}")

        if len(df) < 50:
            logger.warning(f"Insufficient data: {len(df)} compounds")
            continue

        metrics = predictor.train_mechanism_model(
            mechanism=mechanism,
            df=df,
            feature_type="combined",
            identify_top_maccs=True,
        )

        logger.info(f"{mechanism.upper()} Model AUC: {metrics['auc_roc']:.4f}")

    return predictor


def evaluate_predictor(predictor: MechanismPredictor, test_smiles: list):
    """Evaluate predictor on test set."""
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)

    for smi in test_smiles:
        logger.info(f"\nSMILES: {smi}")
        logger.info("-" * 60)

        try:
            results = predictor.predict_all_mechanisms(smi)

            logger.info(f"Dominant Mechanism: {results['dominant_mechanism']}")
            logger.info(f"Confidence: {results['confidence']:.2%}")

            for mech, pred in results["mechanisms"].items():
                if pred:
                    logger.info(
                        f"  {mech}: {pred['prediction']} "
                        f"(prob={pred['probability']:.2%})"
                    )
        except Exception as e:
            logger.error(f"Error: {e}")


def save_training_summary(predictor: MechanismPredictor, output_dir: str):
    """Save training summary to file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect metrics for all trained models
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mechanisms": {},
    }

    for mechanism in predictor.models.keys():
        # Load feature importance
        importance_file = predictor.models_dir / f"{mechanism}_feature_importance.csv"
        if importance_file.exists():
            df_importance = pd.read_csv(importance_file)
            top_features = df_importance.head(10).to_dict("records")
        else:
            top_features = []

        summary["mechanisms"][mechanism] = {
            "top_features": top_features,
            "description": predictor.MECHANISMS[mechanism]["description"],
        }

    # Save summary
    summary_file = output_path / "mechanism_models_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSaved training summary to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Train transport mechanism prediction models"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["synthetic", "curated", "both"],
        default="synthetic",
        help="Training mode: synthetic (B3DB), curated (external data), or both",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/transport_mechanisms/curated",
        help="Directory with curated transport datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mechanism_analysis",
        help="Output directory for results",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run evaluation on test set after training",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TRANSPORT MECHANISM MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    predictor = None

    # Train with synthetic labels from B3DB
    if args.mode in ["synthetic", "both"]:
        predictor = train_with_b3db_synthetic_labels()

    # Train with curated datasets
    if args.mode in ["curated", "both"]:
        predictor_curated = train_with_curated_datasets(args.data_dir)
        if predictor_curated and predictor is None:
            predictor = predictor_curated

    if predictor is None:
        logger.error("Failed to train models")
        return 1

    # Evaluate on test set
    if args.test:
        test_smiles = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC1C(N)C(C(=O)O)C(N)C1O",  # Beta-lactam (efflux risk)
            "CC(C)CC1=CC=C(C=C1)C(C)C(O)=O",  # Ibuprofen
        ]
        evaluate_predictor(predictor, test_smiles)

    # Save training summary
    save_training_summary(predictor, args.output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Models saved to: {predictor.models_dir}")
    logger.info(f"Summary saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
