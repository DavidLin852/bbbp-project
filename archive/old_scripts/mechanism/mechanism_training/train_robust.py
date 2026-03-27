"""
Robust Training Script for Mechanism Prediction Models

Simplified version that handles edge cases and provides clear error messages.
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_b3db():
    """Load B3DB and prepare for training."""
    b3db_path = project_root / "data" / "raw" / "B3DB_classification.tsv"

    if not b3db_path.exists():
        logger.error(f"B3DB not found at {b3db_path}")
        return None

    logger.info(f"Loading B3DB from {b3db_path}")
    df = pd.read_csv(b3db_path, sep='\t')

    # Select relevant columns
    df_clean = df[['SMILES', 'BBB+/BBB-']].copy()
    df_clean.columns = ['smiles', 'label']

    # Convert labels
    df_clean['label'] = df_clean['label'].map({'BBB+': 1, 'BBB-': 0})
    df_clean = df_clean.dropna()

    logger.info(f"Loaded {len(df_clean)} compounds")
    logger.info(f"  BBB+: {df_clean['label'].sum()} ({df_clean['label'].sum()/len(df_clean)*100:.1f}%)")

    return df_clean


def create_features_and_labels(df):
    """Create features and mechanism labels."""
    logger.info("Creating features...")

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, MACCSkeys, DataStructs
    except ImportError:
        logger.error("RDKit not installed")
        return None, None

    features_list = []
    mechanism_labels = []

    for idx, row in df.iterrows():
        smiles = row['smiles']
        label = row['label']

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Physicochemical features (7 features)
            tpsa = Descriptors.TPSA(mol)
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rotbonds = Descriptors.NumRotatableBonds(mol)

            # Simple LogD approximation
            logd = logp
            if hbd > 0:
                logd -= 0.5 * hbd

            physicochemical = [tpsa, mw, logp, logd, hbd, hba, rotbonds]

            # MACCS keys (167 bits)
            maccs = MACCSkeys.GenMACCSKeys(mol)
            maccs_array = np.zeros((167,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(maccs, maccs_array)

            # Combine features
            combined = np.concatenate([physicochemical, maccs_array])

            # Determine mechanism
            if mw < 500 and tpsa < 90:
                mechanism = 'passive'
            elif maccs_array[7] == 1:  # MACCS8 = beta-lactam
                mechanism = 'efflux'
            elif maccs_array[42] == 1 or maccs_array[35] == 1:  # MACCS43 or MACCS36
                mechanism = 'influx'
            else:
                mechanism = 'mixed'

            features_list.append(combined)
            mechanism_labels.append({
                'smiles': smiles,
                'bbb_label': label,
                'mechanism': mechanism
            })

        except Exception as e:
            logger.debug(f"Error processing {smiles}: {e}")
            continue

    X = np.array(features_list)
    df_labels = pd.DataFrame(mechanism_labels)

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels: {len(df_labels)} compounds")

    return X, df_labels


def train_models(X, df_labels):
    """Train XGBoost models for each mechanism."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING XGBOOST MODELS")
    logger.info("="*60)

    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.impute import SimpleImputer

    models_dir = project_root / "artifacts" / "models" / "mechanism"
    models_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Train BBB model
    logger.info("\n1. Training BBB permeability model...")
    y = df_labels['bbb_label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    model = xgb.XGBClassifier(
        max_depth=7,
        n_estimators=200,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    model.fit(X_train_imp, y_train)

    y_pred_proba = model.predict_proba(X_test_imp)[:, 1]
    y_pred = model.predict(X_test_imp)

    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  F1: {f1:.4f}")

    # Save model
    model_path = models_dir / "bbb_model.json"
    model.save_model(str(model_path))
    logger.info(f"  Saved to: {model_path}")

    results['bbb'] = {'auc': auc, 'accuracy': acc, 'f1': f1}

    # Train mechanism-specific models
    for mech in ['passive', 'influx', 'efflux']:
        logger.info(f"\n2. Training {mech.upper()} mechanism model...")

        y_mech = (df_labels['mechanism'] == mech).astype(int).values

        if y_mech.sum() < 50:
            logger.warning(f"  Insufficient positive samples: {y_mech.sum()}")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_mech, test_size=0.2, random_state=42, stratify=y_mech
        )

        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        model = xgb.XGBClassifier(
            max_depth=5,
            n_estimators=100,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()  # Handle class imbalance
        )

        model.fit(X_train_imp, y_train, verbose=False)

        y_pred_proba = model.predict_proba(X_test_imp)[:, 1]
        y_pred = model.predict(X_test_imp)

        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(f"  AUC: {auc:.4f}")
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  F1: {f1:.4f}")

        # Save model
        model_path = models_dir / f"{mech}_model.json"
        model.save_model(str(model_path))
        logger.info(f"  Saved to: {model_path}")

        results[mech] = {'auc': auc, 'accuracy': acc, 'f1': f1}

    return results, imputer


def test_predictions(X, df_labels):
    """Test predictions on sample molecules."""
    logger.info("\n" + "="*60)
    logger.info("TESTING PREDICTIONS")
    logger.info("="*60)

    import xgboost as xgb
    from sklearn.impute import SimpleImputer
    import joblib

    models_dir = project_root / "artifacts" / "models" / "mechanism"

    # Load BBB model
    bbb_model_path = models_dir / "bbb_model.json"
    if not bbb_model_path.exists():
        logger.error("BBB model not found")
        return

    bbb_model = xgb.XGBClassifier()
    bbb_model.load_model(str(bbb_model_path))

    imputer_path = models_dir / "imputer.joblib"
    if imputer_path.exists():
        imputer = joblib.load(imputer_path)
    else:
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X)

    # Test samples
    test_indices = np.random.choice(len(df_labels), size=min(5, len(df_labels)), replace=False)

    for idx in test_indices:
        row = df_labels.iloc[idx]
        smiles = row['smiles']
        true_bbb = row['bbb_label']
        true_mech = row['mechanism']

        logger.info(f"\nSMILES: {smiles[:50]}...")
        logger.info(f"  True BBB: {'BBB+' if true_bbb == 1 else 'BBB-'}")
        logger.info(f"  True Mechanism: {true_mech}")

        try:
            x_feature = X[idx:idx+1]
            x_imp = imputer.transform(x_feature)

            bbb_proba = bbb_model.predict_proba(x_imp)[0, 1]
            bbb_pred = int(bbb_proba >= 0.5)

            logger.info(f"  Pred BBB: {'BBB+' if bbb_pred == 1 else 'BBB-'} (prob={bbb_proba:.2f})")

        except Exception as e:
            logger.error(f"  Error: {e}")


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("BBB TRANSPORT MECHANISM - ROBUST TRAINING")
    logger.info("="*60)

    # 1. Load data
    df = load_and_prepare_b3db()
    if df is None:
        return 1

    # 2. Create features
    X, df_labels = create_features_and_labels(df)
    if X is None:
        return 1

    # Save labeled dataset
    output_dir = project_root / "data" / "transport_mechanisms" / "curated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "b3db_with_features_and_labels.csv"
    df_labels.to_csv(output_file, index=False)
    logger.info(f"\nSaved labeled dataset to: {output_file}")

    # 3. Train models
    results, imputer = train_models(X, df_labels)

    # Save imputer
    import joblib
    imputer_path = project_root / "artifacts" / "models" / "mechanism" / "imputer.joblib"
    joblib.dump(imputer, imputer_path)
    logger.info(f"\nSaved imputer to: {imputer_path}")

    # 4. Test predictions
    test_predictions(X, df_labels)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    for model_name, metrics in results.items():
        logger.info(f"{model_name.upper()}:")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")

    logger.info("\n" + "="*60)
    logger.info("COMPLETE!")
    logger.info("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
