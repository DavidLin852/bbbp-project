"""
Explore B3DB dataset to understand transport mechanisms.

This script analyzes the B3DB dataset to find patterns and clues
about transport mechanisms based on molecular properties.
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


def extract_features(smiles):
    """Extract features from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'MW': Descriptors.MolWt(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'RotatableBonds': Descriptors.NumRotatableBonds(mol),
        'RingCount': Descriptors.RingCount(mol),
        'AromaticRings': Descriptors.NumAromaticRings(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
    }


def load_b3db_data():
    """Load B3DB classification dataset."""
    b3db_file = Paths.data_raw / "B3DB_classification.tsv"
    df = pd.read_csv(b3db_file, sep='\t')

    # Convert BBB+/BBB- to binary
    df['BBB_binary'] = (df['BBB+/BBB-'] == 'BBB+').astype(int)

    return df


def analyze_b3db_by_group():
    """Analyze B3DB properties by data group."""
    print("="*80)
    print("B3DB Dataset Analysis by Group")
    print("="*80)

    # Load data
    df = load_b3db_data()

    print(f"\nTotal samples: {len(df)}")
    print(f"\nGroup distribution:")
    print(df['group'].value_counts().sort_index())

    # Extract features for all molecules
    print("\nExtracting features...")
    features_list = []
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"   Processing {idx}/{len(df)}...")

        feats = extract_features(row['SMILES'])
        if feats is not None:
            features_list.append({**feats, **row.to_dict()})

    features_df = pd.DataFrame(features_list)
    print(f"Valid samples: {len(features_df)}")

    # Analyze by group
    print("\n" + "="*80)
    print("Analysis by Data Group")
    print("="*80)

    for group in sorted(features_df['group'].unique()):
        df_group = features_df[features_df['group'] == group]

        bbb_plus = df_group[df_group['BBB_binary'] == 1]
        bbb_minus = df_group[df_group['BBB_binary'] == 0]

        print(f"\nGroup {group}:")
        print(f"  Total: {len(df_group)}")
        print(f"  BBB+: {len(bbb_plus)} ({len(bbb_plus)/len(df_group)*100:.1f}%)")
        print(f"  BBB-: {len(bbb_minus)} ({len(bbb_minus)/len(df_group)*100:.1f}%)")

        # Properties comparison
        print(f"\n  Properties (BBB+ vs BBB-):")
        print(f"  {'Property':<15} {'BBB+':>10} {'BBB-':>10} {'Diff':>10}")
        print(f"  {'-'*50}")

        for prop in ['TPSA', 'MW', 'LogP', 'HBA', 'HBD']:
            if prop in df_group.columns:
                pos_mean = bbb_plus[prop].mean()
                neg_mean = bbb_minus[prop].mean()
                diff = pos_mean - neg_mean
                print(f"  {prop:<15} {pos_mean:>10.2f} {neg_mean:>10.2f} {diff:>+10.2f}")

    # Analyze comments field for mechanism hints
    print("\n" + "="*80)
    print("Analyzing 'comments' Field for Mechanism Clues")
    print("="*80)

    # Check if comments column exists
    if 'comments' in features_df.columns:
        # Get unique non-null comments
        comments = features_df['comments'].dropna().unique()

        # Look for keywords related to mechanisms
        keywords = {
            'efflux': ['efflux', 'p-gp', 'pgp', 'export', 'pump'],
            'influx': ['influx', 'transport', 'carrier', 'uptake'],
            'passive': ['passive', 'diffusion'],
            'metabolism': ['metabol', 'cyp', 'enzyme'],
        }

        found_comments = []
        for comment in comments:
            comment_str = str(comment).lower()
            for category, kw_list in keywords.items():
                if any(kw in comment_str for kw in kw_list):
                    found_comments.append((category, comment))
                    break

        if found_comments:
            print(f"\nFound {len(found_comments)} comments with mechanism keywords:")
            for category, comment in found_comments[:20]:  # Show first 20
                print(f"  [{category.upper()}] {comment}")
        else:
            print("\nNo obvious mechanism keywords found in comments.")

    # Reference analysis
    print("\n" + "="*80)
    print("Reference Analysis")
    print("="*80)

    if 'reference' in features_df.columns:
        ref_counts = features_df['reference'].value_counts()
        print(f"\nTop 10 most common references:")
        for ref, count in ref_counts.head(10).items():
            print(f"  {ref}: {count} samples")

    # Lipinski Rule of 5 analysis
    print("\n" + "="*80)
    print("Lipinski Rule of 5 Analysis")
    print("="*80)

    def check_lipinski(row):
        violations = 0
        if row['MW'] > 500:
            violations += 1
        if row['LogP'] > 5:
            violations += 1
        if row['HBA'] > 10:
            violations += 1
        if row['HBD'] > 5:
            violations += 1
        return violations

    features_df['lipinski_violations'] = features_df.apply(check_lipinski, axis=1)

    print("\nLipinski violations by BBB status:")
    print(f"{'Violations':<12} {'BBB+':>10} {'BBB-':>10}")
    print("-" * 35)

    for viol in range(5):
        bbb_plus_count = len(features_df[(features_df['BBB_binary'] == 1) &
                                        (features_df['lipinski_violations'] == viol)])
        bbb_minus_count = len(features_df[(features_df['BBB_binary'] == 0) &
                                         (features_df['lipinski_violations'] == viol)])
        print(f"{viol:<12} {bbb_plus_count:>10} {bbb_minus_count:>10}")

    # CNS drug-likeness (Golden Triangle)
    print("\n" + "="*80)
    print("CNS Drug-Likeness (Golden Triangle Criteria)")
    print("="*80)

    # Golden Triangle: TPSA < 90, MW < 450, LogP 2-4
    features_df['golden_triangle'] = (
        (features_df['TPSA'] < 90) &
        (features_df['MW'] < 450) &
        (features_df['LogP'] >= 2) &
        (features_df['LogP'] <= 4)
    )

    gt_bbb_plus = len(features_df[(features_df['BBB_binary'] == 1) &
                                  features_df['golden_triangle']])
    gt_bbb_minus = len(features_df[(features_df['BBB_binary'] == 0) &
                                   features_df['golden_triangle']])

    print(f"\nGolden Triangle compliance:")
    print(f"  BBB+: {gt_bbb_plus} / {len(features_df[features_df['BBB_binary'] == 1])} "
          f"({gt_bbb_plus/len(features_df[features_df['BBB_binary'] == 1])*100:.1f}%)")
    print(f"  BBB-: {gt_bbb_minus} / {len(features_df[features_df['BBB_binary'] == 0])} "
          f"({gt_bbb_minus/len(features_df[features_df['BBB_binary'] == 0])*100:.1f}%)")

    # Save results
    output_dir = Paths.root / "outputs" / "b3db_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "b3db_with_features.csv"
    features_df.to_csv(output_file, index=False)
    print(f"\nFeatures saved to: {output_file}")

    return features_df


def compare_b3db_vs_cornelissen():
    """Compare B3DB with Cornelissen 2022 dataset."""
    print("\n" + "="*80)
    print("Comparison: B3DB vs Cornelissen 2022")
    print("="*80)

    # Load B3DB
    b3db_df = load_b3db_data()

    # Load Cornelissen
    cornelissen_file = Paths.root / "data" / "transport_mechanisms" / "cornelissen_2022" / "cornelissen_2022_processed.csv"
    if cornelissen_file.exists():
        cornelissen_df = pd.read_csv(cornelissen_file)

        print(f"\nDataset Comparison:")
        print(f"{'Metric':<30} {'B3DB':>15} {'Cornelissen':>15}")
        print("-" * 65)
        print(f"{'Total samples':<30} {len(b3db_df):>15} {len(cornelissen_df):>15}")
        print(f"{'BBB+ rate':<30} {(b3db_df['BBB_binary'].mean()*100):>15.1f}% "
              f"{(cornelissen_df['label_BBB'].mean()*100):>15.1f}%")

        # Check overlapping SMILES
        b3db_smiles = set(b3db_df['SMILES'].tolist())
        cornelissen_smiles = set(cornelissen_df['SMILES'].tolist())

        overlap = b3db_smiles & cornelissen_smiles

        print(f"\nOverlapping molecules: {len(overlap)}")

        if len(overlap) > 0:
            print(f"\nThis is useful for cross-validation!")
            print(f"We can use B3DB to test models trained on Cornelissen 2022.")
    else:
        print("\nCornelissen 2022 dataset not found.")


def suggest_mechanism_labels():
    """
    Suggest potential mechanism labels for B3DB based on properties.

    Using rules from Cornelissen et al. 2022:
    - Passive diffusion: Low TPSA (<90), moderate LogP (1-3)
    - Active influx: High TPSA (>100), high HBA
    - Active efflux: High MW (>500)
    """
    print("\n" + "="*80)
    print("Suggested Mechanism Labels for B3DB")
    print("="*80)

    # Load B3DB with features
    output_file = Paths.root / "outputs" / "b3db_analysis" / "b3db_with_features.csv"
    if output_file.exists():
        df = pd.read_csv(output_file)

        # Apply rules to suggest mechanisms
        df['suggested_passive'] = (
            (df['TPSA'] < 90) &
            (df['LogP'] >= 1) &
            (df['LogP'] <= 3) &
            (df['MW'] < 500)
        ).astype(int)

        df['suggested_influx'] = (
            (df['TPSA'] > 100) &
            (df['HBA'] > 5)
        ).astype(int)

        df['suggested_efflux'] = (
            (df['MW'] > 500)
        ).astype(int)

        print("\nSuggested mechanism distribution:")
        print(f"{'Mechanism':<20} {'BBB+':>10} {'BBB-':>10}")
        print("-" * 45)

        for mech in ['suggested_passive', 'suggested_influx', 'suggested_efflux']:
            mech_name = mech.replace('suggested_', '').title()
            bbb_plus = df[df['BBB_binary'] == 1][mech].sum()
            bbb_minus = df[df['BBB_binary'] == 0][mech].sum()
            print(f"{mech_name:<20} {bbb_plus:>10} {bbb_minus:>10}")

        # Save with suggestions
        output_file2 = Paths.root / "outputs" / "b3db_analysis" / "b3db_with_mechanism_suggestions.csv"
        df.to_csv(output_file2, index=False)
        print(f"\nSaved with mechanism suggestions to: {output_file2}")

        print("\nNote: These are heuristic labels based on property rules.")
        print("For true mechanism labels, experimental validation is required.")


def main():
    """Main analysis function."""
    print("="*80)
    print("Exploring B3DB Dataset for Transport Mechanisms")
    print("="*80)

    # Analyze B3DB
    df = analyze_b3db_by_group()

    # Compare with Cornelissen
    compare_b3db_vs_cornelissen()

    # Suggest mechanism labels
    suggest_mechanism_labels()

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the generated CSV files in outputs/b3db_analysis/")
    print("2. Use suggested mechanism labels as a starting point")
    print("3. Cross-reference with literature for experimental validation")
    print("4. Consider merging B3DB with Cornelissen 2022 for larger training set")


if __name__ == "__main__":
    main()
