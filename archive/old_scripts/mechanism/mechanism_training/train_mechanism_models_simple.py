"""
Train Mechanism Prediction Models from B3DB

Simplified training script that:
1. Loads B3DB data
2. Creates synthetic transport mechanism labels
3. Trains XGBoost models for each mechanism
4. Evaluates and saves results

Usage:
    python scripts/mechanism_training/train_mechanism_models_simple.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.path_prediction.feature_extractor import MechanismFeatureExtractor
from src.path_prediction.mechanism_predictor import MechanismPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_b3db_dataset():
    """Load and prepare B3DB dataset."""
    b3db_path = project_root / "data" / "raw" / "B3DB_classification.tsv"

    if not b3db_path.exists():
        logger.error(f"B3DB not found at {b3db_path}")
        return None

    logger.info(f"Loading B3DB from {b3db_path}")
    df = pd.read_csv(b3db_path, sep='\t')

    logger.info(f"Loaded {len(df)} compounds")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Check for SMILES column
    smiles_col = None
    for col in ['SMILES', 'smiles', 'canonical_smiles']:
        if col in df.columns:
            smiles_col = col
            break

    if smiles_col is None:
        logger.error("No SMILES column found in B3DB")
        return None

    # Check for label column
    label_col = None
    for col in ['BBB_permeable', 'label', 'class', 'BBB', 'BBB+/BBB-']:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        logger.error("No label column found in B3DB")
        return None

    # Prepare dataset
    df_clean = df[[smiles_col, label_col]].copy()
    df_clean.columns = ['smiles', 'label']
    df_clean = df_clean.dropna()

    # Convert label to binary if needed
    if df_clean['label'].dtype == 'object':
        df_clean['label'] = df_clean['label'].replace({
            'BBB+': 1,
            'BBB-': 0,
            'yes': 1, 'y': 1, 'true': 1, 'Y': 1,
            'no': 0, 'n': 0, 'false': 0, 'N': 0
        })

    df_clean['label'] = pd.to_numeric(df_clean['label'], errors='coerce')
    df_clean = df_clean.dropna()

    logger.info(f"Clean dataset: {len(df_clean)} compounds")
    logger.info(f"  BBB+: {df_clean['label'].sum()} ({df_clean['label'].sum()/len(df_clean)*100:.1f}%)")
    logger.info(f"  BBB-: {len(df_clean)-df_clean['label'].sum()} ({(len(df_clean)-df_clean['label'].sum())/len(df_clean)*100:.1f}%)")

    return df_clean


def create_synthetic_mechanism_labels(df):
    """Create synthetic transport mechanism labels using physicochemical rules."""
    logger.info("\nCreating synthetic transport mechanism labels...")

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, MACCSkeys
    except ImportError:
        logger.error("RDKit not installed")
        return None

    labeled_data = []

    for idx, row in df.iterrows():
        smiles = row['smiles']

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Calculate properties
            mw = Descriptors.MolWt(mol)
            tpsa = Descriptors.TPSA(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rotbonds = Descriptors.NumRotatableBonds(mol)

            # Get MACCS keys (167 bits from RDKit)
            maccs = MACCSkeys.GenMACCSKeys(mol)
            from rdkit.Chem import DataStructs
            maccs_array = np.zeros((167,), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(maccs, maccs_array)

            # Predict transport mechanism
            # Rules based on Cornelissen et al. 2022
            if mw < 500 and tpsa < 90 and logp > 0 and logp < 5:
                mechanism = 'passive'
            # MACCS8 (index 7, 0-based) = beta-lactam (efflux risk)
            elif maccs_array[7] == 1 or mw > 500:
                mechanism = 'efflux'
            # MACCS43 (index 42) and MACCS36 (index 35) = influx favorable
            elif maccs_array[42] == 1 or maccs_array[35] == 1:
                mechanism = 'influx'
            else:
                mechanism = 'mixed'

            labeled_data.append({
                'smiles': smiles,
                'bbb_label': row['label'],
                'mechanism': mechanism,
                'mw': mw,
                'tpsa': tpsa,
                'logp': logp,
                'hbd': hbd,
                'hba': hba,
            })

        except Exception as e:
            logger.debug(f"Error processing {smiles}: {e}")
            continue

    df_labeled = pd.DataFrame(labeled_data)

    logger.info(f"Created labels for {len(df_labeled)} compounds")
    logger.info(f"  Passive: {(df_labeled['mechanism']=='passive').sum()}")
    logger.info(f"  Influx: {(df_labeled['mechanism']=='influx').sum()}")
    logger.info(f"  Efflux: {(df_labeled['mechanism']=='efflux').sum()}")
    logger.info(f"  Mixed: {(df_labeled['mechanism']=='mixed').sum()}")

    return df_labeled


def train_mechanism_models(df_labeled):
    """Train models for each transport mechanism."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING MECHANISM PREDICTION MODELS")
    logger.info("="*60)

    predictor = MechanismPredictor()

    # 1. Train BBB permeability model
    logger.info("\n1. Training BBB permeability model...")
    df_bbb = df_labeled[['smiles', 'bbb_label']].copy()
    df_bbb = df_bbb.rename(columns={'bbb_label': 'label'})

    try:
        bbb_metrics = predictor.train_mechanism_model(
            mechanism='bbb',
            df=df_bbb,
            feature_type='combined',
            identify_top_maccs=True,
        )
        logger.info(f"  BBB Model AUC: {bbb_metrics['auc_roc']:.4f}")
    except Exception as e:
        logger.error(f"  Error training BBB model: {e}")

    # 2. Train mechanism-specific models
    for mech_name in ['passive', 'influx', 'efflux']:
        logger.info(f"\n2. Training {mech_name.upper()} mechanism model...")

        # Create binary classification dataset
        mask = df_labeled['mechanism'] == mech_name

        df_mech = pd.concat([
            df_labeled[mask][['smiles']].assign(label=1),
            df_labeled[~mask][['smiles']].assign(label=0),
        ]).dropna()

        if len(df_mech) < 100:
            logger.warning(f"  Insufficient data for {mech_name}: {len(df_mech)}")
            continue

        try:
            metrics = predictor.train_mechanism_model(
                mechanism=mech_name,
                df=df_mech,
                feature_type='combined',
                identify_top_maccs=True,
            )
            logger.info(f"  {mech_name.upper()} Model AUC: {metrics['auc_roc']:.4f}")
        except Exception as e:
            logger.error(f"  Error training {mech_name} model: {e}")

    return predictor


def test_predictions(predictor, df_labeled):
    """Test predictions on sample molecules."""
    logger.info("\n" + "="*60)
    logger.info("TESTING PREDICTIONS")
    logger.info("="*60)

    # Select test samples
    test_samples = df_labeled.sample(min(5, len(df_labeled)), random_state=42)

    for idx, row in test_samples.iterrows():
        smiles = row['smiles']
        bbb_label = int(row['bbb_label'])
        mechanism = row['mechanism']

        logger.info(f"\nSMILES: {smiles[:50]}...")
        logger.info(f"  True BBB: {'BBB+' if bbb_label == 1 else 'BBB-'}")
        logger.info(f"  True Mechanism: {mechanism}")

        try:
            # Predict BBB
            bbb_pred = predictor.predict_mechanism(smiles, 'bbb')
            logger.info(f"  Pred BBB: {'BBB+' if bbb_pred['prediction'] == 1 else 'BBB-'} "
                       f"(prob={bbb_pred['probability']:.2f})")

            # Predict mechanism
            mech_pred = predictor.predict_mechanism(smiles, mechanism)
            logger.info(f"  Pred {mechanism}: {'Positive' if mech_pred['prediction'] == 1 else 'Negative'} "
                       f"(prob={mech_pred['probability']:.2f})")

        except Exception as e:
            logger.error(f"  Prediction error: {e}")


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("BBB TRANSPORT MECHANISM PREDICTION - MODEL TRAINING")
    logger.info("="*60)

    # 1. Load B3DB data
    df_bbb = create_b3db_dataset()
    if df_bbb is None:
        logger.error("Failed to load B3DB data")
        return 1

    # 2. Create synthetic mechanism labels
    df_labeled = create_synthetic_mechanism_labels(df_bbb)
    if df_labeled is None:
        logger.error("Failed to create mechanism labels")
        return 1

    # Save labeled dataset
    output_dir = project_root / "data" / "transport_mechanisms" / "curated"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "b3db_with_mechanism_labels.csv"
    df_labeled.to_csv(output_file, index=False)
    logger.info(f"\nSaved labeled dataset to: {output_file}")

    # 3. Train models
    predictor = train_mechanism_models(df_labeled)

    # 4. Test predictions
    test_predictions(predictor, df_labeled)

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Models saved to: {project_root / 'artifacts' / 'models' / 'mechanism'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
