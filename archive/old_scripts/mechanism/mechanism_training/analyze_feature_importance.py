"""
Analyze Feature Importance for Transport Mechanisms

Identifies the most important features (physicochemical + MACCS keys)
for each transport mechanism.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Analyze feature importance for trained models."""
    logger.info("="*60)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*60)

    # Load labeled dataset
    data_file = project_root / "data" / "transport_mechanisms" / "curated" / "b3db_with_features_and_labels.csv"
    df = pd.read_csv(data_file)

    logger.info(f"Loaded {len(df)} compounds")

    # Mechanism distribution
    logger.info("\nMechanism Distribution:")
    for mech in ['passive', 'influx', 'efflux', 'mixed']:
        count = (df['mechanism'] == mech).sum()
        logger.info(f"  {mech}: {count} ({count/len(df)*100:.1f}%)")

    # Analyze physicochemical properties by mechanism
    logger.info("\n" + "="*60)
    logger.info("PHYSICOCHEMICAL PROPERTIES BY MECHANISM")
    logger.info("="*60)

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, MACCSkeys, DataStructs
    except ImportError:
        logger.error("RDKit not installed")
        return 1

    # Collect statistics
    stats = []

    for mech in ['passive', 'influx', 'efflux', 'mixed']:
        df_mech = df[df['mechanism'] == mech].head(500)  # Sample for speed

        properties = {
            'TPSA': [],
            'MW': [],
            'LogP': [],
            'HBD': [],
            'HBA': [],
        }

        for smiles in df_mech['smiles']:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                properties['TPSA'].append(Descriptors.TPSA(mol))
                properties['MW'].append(Descriptors.MolWt(mol))
                properties['LogP'].append(Descriptors.MolLogP(mol))
                properties['HBD'].append(Descriptors.NumHDonors(mol))
                properties['HBA'].append(Descriptors.NumHAcceptors(mol))
            except:
                continue

        logger.info(f"\n{mech.upper()}:")
        for prop_name, values in properties.items():
            if values:
                logger.info(f"  {prop_name}: {np.mean(values):.2f} ± {np.std(values):.2f}")

    # Analyze MACCS key frequencies by mechanism
    logger.info("\n" + "="*60)
    logger.info("TOP MACCS KEYS BY MECHANISM")
    logger.info("="*60)

    # MACCS key descriptions (selected)
    MACCS_DESCRIPTIONS = {
        7: "Four-membered ring (beta-lactam)",
        35: "Sulfur heterocycle",
        42: "Two amino groups on same carbon",
        63: "Aromatic (>1 ring)",
        96: "Carboxylic acid",
        103: "Primary amine",
        107: "Secondary amine",
        114: "Tertiary amine",
        121: "Halogen",
        140: "Carbonyl",
    }

    for mech in ['passive', 'influx', 'efflux', 'mixed']:
        df_mech = df[df['mechanism'] == mech].head(500)

        # Count MACCS key occurrences
        maccs_counts = np.zeros(167)

        for smiles in df_mech['smiles']:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_array = np.zeros((167,), dtype=np.int32)
                DataStructs.ConvertToNumpyArray(maccs, maccs_array)

                maccs_counts += maccs_array
            except:
                continue

        # Get top 5 most common keys
        top_indices = np.argsort(maccs_counts)[-5:][::-1]

        logger.info(f"\n{mech.upper()} - Top 5 MACCS Keys:")
        for idx in top_indices:
            freq = int(maccs_counts[idx])
            pct = freq / len(df_mech) * 100
            desc = MACCS_DESCRIPTIONS.get(idx, f"MACCS key {idx}")
            logger.info(f"  MACCS {idx:3.0f}: {desc:40s} ({freq:3d}/{len(df_mech)}, {pct:5.1f}%)")

    # Compare key differences between mechanisms
    logger.info("\n" + "="*60)
    logger.info("KEY DIFFERENCES BETWEEN MECHANISMS")
    logger.info("="*60)

    # Passive vs Efflux
    logger.info("\nPassive Diffusion vs Active Efflux:")

    passive_maccs = np.zeros(167)
    efflux_maccs = np.zeros(167)

    for smiles in df[df['mechanism'] == 'passive']['smiles'].head(500):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_array = np.zeros((167,), dtype=np.int32)
                DataStructs.ConvertToNumpyArray(maccs, maccs_array)
                passive_maccs += maccs_array
        except:
            pass

    for smiles in df[df['mechanism'] == 'efflux']['smiles']:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_array = np.zeros((167,), dtype=np.int32)
                DataStructs.ConvertToNumpyArray(maccs, maccs_array)
                efflux_maccs += maccs_array
        except:
            pass

    # Find keys with large differences
    diff = passive_maccs / max(1, len(df[df['mechanism'] == 'passive'])) - \
           efflux_maccs / max(1, len(df[df['mechanism'] == 'efflux']))

    logger.info("\nMore common in PASSIVE:")
    for idx in np.argsort(diff)[-5:][::-1]:
        if diff[idx] > 0.05:  # At least 5% difference
            desc = MACCS_DESCRIPTIONS.get(idx, f"MACCS {idx}")
            logger.info(f"  MACCS {idx:3.0f}: {desc:40s} (+{diff[idx]*100:.1f}%)")

    logger.info("\nMore common in EFFLUX:")
    for idx in np.argsort(diff)[:5]:
        if diff[idx] < -0.05:
            desc = MACCS_DESCRIPTIONS.get(idx, f"MACCS {idx}")
            logger.info(f"  MACCS {idx:3.0f}: {desc:40s} ({diff[idx]*100:.1f}%)")

    # Save analysis results
    output_dir = project_root / "outputs" / "mechanism_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / "feature_importance_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("BBB Transport Mechanism - Feature Importance Analysis\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total compounds: {len(df)}\n\n")
        f.write("Mechanism Distribution:\n")
        for mech in ['passive', 'influx', 'efflux', 'mixed']:
            count = (df['mechanism'] == mech).sum()
            f.write(f"  {mech}: {count} ({count/len(df)*100:.1f}%)\n")

    logger.info(f"\nSaved summary to: {summary_file}")

    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
