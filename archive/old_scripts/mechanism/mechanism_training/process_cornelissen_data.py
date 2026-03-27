"""
Process Cornelissen et al. 2022 dataset for mechanism prediction.

This script:
1. Loads the 20220103_table.csv dataset
2. Extracts features (physicochemical + MACCS fingerprints)
3. Prepares train/test splits for each mechanism
4. Saves processed data for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Paths


def extract_physicochemical_features(mol):
    """Extract physicochemical features from a molecule."""
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


def extract_maccs_fingerprints(mol):
    """Extract MACCS fingerprints (167 bits)."""
    try:
        maccs = MACCSkeys.GenMACCSKeys(mol)
        return np.array(maccs)
    except:
        return np.zeros(167)


def extract_morgan_fingerprints(mol, radius=2, n_bits=2048):
    """Extract Morgan fingerprints (ECFP4-like)."""
    try:
        morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(morgan)
    except:
        return np.zeros(n_bits)


def extract_all_features(smiles):
    """Extract all features from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Extract features
    physicochemical = extract_physicochemical_features(mol)
    maccs = extract_maccs_fingerprints(mol)
    morgan = extract_morgan_fingerprints(mol, radius=2, n_bits=1024)

    # Combine features
    feature_dict = {
        'SMILES': smiles,
        **physicochemical,
    }

    # Add MACCS features
    for i, val in enumerate(maccs):
        feature_dict[f'MACCS_{i}'] = int(val)

    # Add Morgan features
    for i, val in enumerate(morgan):
        feature_dict[f'Morgan_{i}'] = int(val)

    return feature_dict


def main():
    """Main processing function."""
    print("="*80)
    print("Processing Cornelissen et al. 2022 Dataset")
    print("="*80)

    # Create output directory
    data_dir = Paths.root / "data"
    output_dir = data_dir / "transport_mechanisms" / "cornelissen_2022"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n1. Loading dataset...")
    input_file = Paths.root / "20220103_table.csv"
    df = pd.read_csv(input_file)
    print(f"   Total samples: {len(df)}")

    # Dataset info
    print("\n2. Dataset info:")
    print("   Mechanism      | Samples | Positive Rate")
    print("   " + "-"*45)
    for mech in ['Influx', 'Efflux', 'PAMPA', 'BBB', 'CNS']:
        col = f'Status {mech}'
        count = df[col].notna().sum()
        if count > 0:
            pos_rate = df[col].mean() * 100
            print(f"   {mech:13s} | {count:7d} | {pos_rate:5.1f}%")

    # Extract features
    print("\n3. Extracting features...")
    features_list = []
    invalid_smiles = []

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"   Processing {idx}/{len(df)}...")

        smiles = row['ParentSmiles']
        features = extract_all_features(smiles)

        if features is None:
            invalid_smiles.append((idx, smiles))
            continue

        # Add labels
        for mech in ['Influx', 'Efflux', 'PAMPA', 'BBB', 'CNS']:
            col = f'Status {mech}'
            if pd.notna(row[col]):
                features[f'label_{mech}'] = int(row[col])
            else:
                features[f'label_{mech}'] = np.nan

        # Add train/test split
        for mech in ['Influx', 'Efflux', 'PAMPA', 'BBB', 'CNS']:
            split_col = f'Testrain {mech}' if mech != 'CNS' else f'Testin CNS'
            if split_col in df.columns and pd.notna(row[split_col]):
                features[f'split_{mech}'] = row[split_col]
            else:
                features[f'split_{mech}'] = np.nan

        features_list.append(features)

    if invalid_smiles:
        print(f"\n   Warning: {len(invalid_smiles)} invalid SMILES found")

    # Create feature DataFrame
    print("\n4. Creating feature matrix...")
    features_df = pd.DataFrame(features_list)
    print(f"   Valid samples: {len(features_df)}")

    # Identify feature columns
    physicochemical_cols = [c for c in features_df.columns if c not in
                           ['SMILES'] + [c for c in features_df.columns if c.startswith('label_')] +
                           [c for c in features_df.columns if c.startswith('split_')] and
                           not c.startswith('MACCS_') and not c.startswith('Morgan_')]

    maccs_cols = [c for c in features_df.columns if c.startswith('MACCS_')]
    morgan_cols = [c for c in features_df.columns if c.startswith('Morgan_')]

    print(f"\n   Feature categories:")
    print(f"   - Physicochemical: {len(physicochemical_cols)}")
    print(f"   - MACCS: {len(maccs_cols)}")
    print(f"   - Morgan: {len(morgan_cols)}")
    print(f"   - Total: {len(physicochemical_cols) + len(maccs_cols) + len(morgan_cols)}")

    # Save processed data
    print("\n5. Saving processed data...")
    output_file = output_dir / "cornelissen_2022_processed.csv"
    features_df.to_csv(output_file, index=False)
    print(f"   Saved to: {output_file}")

    # Save feature lists
    import json
    feature_info = {
        'physicochemical': physicochemical_cols,
        'maccs': maccs_cols,
        'morgan': morgan_cols,
    }
    info_file = output_dir / "feature_info.json"
    with open(info_file, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"   Feature info saved to: {info_file}")

    # Summary statistics
    print("\n6. Summary statistics:")
    for mech in ['Influx', 'Efflux', 'PAMPA', 'BBB', 'CNS']:
        label_col = f'label_{mech}'
        split_col = f'split_{mech}'

        # Count samples with labels
        labeled = features_df[label_col].notna().sum()
        if labeled == 0:
            continue

        train_count = ((features_df[split_col] == 'Train') & (features_df[label_col].notna())).sum()
        test_count = ((features_df[split_col] == 'Test') & (features_df[label_col].notna())).sum()

        pos_rate = features_df[label_col].mean() * 100

        print(f"\n   {mech}:")
        print(f"     Total labeled: {labeled}")
        print(f"     Train: {train_count}, Test: {test_count}")
        print(f"     Positive rate: {pos_rate:.1f}%")

    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)


if __name__ == "__main__":
    main()
