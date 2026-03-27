"""
Merge B3DB and Cornelissen 2022 datasets.

This script:
1. Finds overlapping molecules between B3DB and Cornelissen 2022
2. Transfers mechanism labels from Cornelissen to B3DB
3. Creates a combined dataset with both BBB and mechanism labels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Paths


def main():
    """Main merge function."""
    print("="*80)
    print("Merging B3DB and Cornelissen 2022 Datasets")
    print("="*80)

    # Load B3DB
    print("\n1. Loading B3DB dataset...")
    b3db_file = Paths.root / "outputs" / "b3db_analysis" / "b3db_with_features.csv"
    if not b3db_file.exists():
        print(f"   Error: {b3db_file} not found.")
        print(f"   Please run explore_b3db_mechanisms.py first.")
        return

    b3db_df = pd.read_csv(b3db_file)
    print(f"   Loaded {len(b3db_df)} molecules from B3DB")

    # Load Cornelissen 2022
    print("\n2. Loading Cornelissen 2022 dataset...")
    cornelissen_file = Paths.root / "data" / "transport_mechanisms" / "cornelissen_2022" / "cornelissen_2022_processed.csv"
    if not cornelissen_file.exists():
        print(f"   Error: {cornelissen_file} not found.")
        print(f"   Please run process_cornelissen_data.py first.")
        return

    cornelissen_df = pd.read_csv(cornelissen_file)
    print(f"   Loaded {len(cornelissen_df)} molecules from Cornelissen 2022")

    # Find overlapping molecules
    print("\n3. Finding overlapping molecules...")
    b3db_smiles = set(b3db_df['SMILES'].tolist())
    cornelissen_smiles = set(cornelissen_df['SMILES'].tolist())

    overlap_smiles = b3db_smiles & cornelissen_smiles
    print(f"   Found {len(overlap_smiles)} overlapping molecules")

    # Create merged dataset
    print("\n4. Creating merged dataset...")
    merged_df = b3db_df.copy()

    # Add mechanism columns (initialize with NaN)
    for mech in ['Influx', 'Efflux', 'PAMPA', 'CNS']:
        merged_df[f'label_{mech}'] = np.nan
        merged_df[f'split_{mech}'] = np.nan

    # Transfer labels for overlapping molecules
    labeled_count = 0
    for smiles in overlap_smiles:
        # Get B3DB row
        b3db_idx = merged_df[merged_df['SMILES'] == smiles].index
        if len(b3db_idx) == 0:
            continue

        # Get Cornelissen row
        cornelissen_row = cornelissen_df[cornelissen_df['SMILES'] == smiles]
        if len(cornelissen_row) == 0:
            continue
        cornelissen_row = cornelissen_row.iloc[0]

        # Transfer labels
        for mech in ['Influx', 'Efflux', 'PAMPA', 'CNS']:
            label_col = f'label_{mech}'
            split_col = f'split_{mech}'

            if pd.notna(cornelissen_row[label_col]):
                merged_df.loc[b3db_idx, label_col] = cornelissen_row[label_col]
                merged_df.loc[b3db_idx, split_col] = cornelissen_row[split_col]

        labeled_count += 1

    print(f"   Transferred labels for {labeled_count} molecules")

    # Add dataset source column
    merged_df['dataset_source'] = 'B3DB'
    merged_df.loc[merged_df['SMILES'].isin(overlap_smiles), 'dataset_source'] = 'B3DB+Cornelissen'

    # Add heuristic mechanism labels (for non-overlapping molecules)
    print("\n5. Adding heuristic mechanism labels...")
    merged_df['heuristic_passive'] = (
        (merged_df['TPSA'] < 90) &
        (merged_df['LogP'] >= 1) &
        (merged_df['LogP'] <= 3) &
        (merged_df['MW'] < 500)
    ).astype(int)

    merged_df['heuristic_influx'] = (
        (merged_df['TPSA'] > 100) &
        (merged_df['HBA'] > 5)
    ).astype(int)

    merged_df['heuristic_efflux'] = (
        (merged_df['MW'] > 500)
    ).astype(int)

    # Statistics
    print("\n6. Merged dataset statistics:")
    print(f"   Total molecules: {len(merged_df)}")
    print(f"   From B3DB only: {len(merged_df[merged_df['dataset_source'] == 'B3DB'])}")
    print(f"   From B3DB+Cornelissen: {len(merged_df[merged_df['dataset_source'] == 'B3DB+Cornelissen'])}")

    print("\n   Mechanism labels (experimental from Cornelissen):")
    for mech in ['Influx', 'Efflux', 'PAMPA', 'CNS']:
        col = f'label_{mech}'
        count = merged_df[col].notna().sum()
        if count > 0:
            pos_rate = merged_df[merged_df[col].notna()][col].mean() * 100
            print(f"   {mech}: {count} labeled ({pos_rate:.1f}% positive)")

    print("\n   BBB labels:")
    bbb_plus = (merged_df['BBB_binary'] == 1).sum()
    bbb_minus = (merged_df['BBB_binary'] == 0).sum()
    print(f"   BBB+: {bbb_plus} ({bbb_plus/len(merged_df)*100:.1f}%)")
    print(f"   BBB-: {bbb_minus} ({bbb_minus/len(merged_df)*100:.1f}%)")

    # Save merged dataset
    print("\n7. Saving merged dataset...")
    output_dir = Paths.root / "data" / "transport_mechanisms" / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "b3db_cornelissen_merged.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"   Saved to: {output_file}")

    # Save overlap list
    overlap_file = output_dir / "overlapping_smiles.txt"
    with open(overlap_file, 'w') as f:
        for smiles in overlap_smiles:
            f.write(f"{smiles}\n")
    print(f"   Overlap list saved to: {overlap_file}")

    # Save statistics
    stats = {
        'total_b3db': len(b3db_df),
        'total_cornelissen': len(cornelissen_df),
        'overlap_count': len(overlap_smiles),
        'merged_total': len(merged_df),
        'mechanism_labels': {},
    }

    for mech in ['Influx', 'Efflux', 'PAMPA', 'CNS']:
        col = f'label_{mech}'
        count = merged_df[col].notna().sum()
        if count > 0:
            stats['mechanism_labels'][mech] = {
                'count': int(count),
                'positive_rate': float(merged_df[merged_df[col].notna()][col].mean())
            }

    import json
    stats_file = output_dir / "merge_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   Statistics saved to: {stats_file}")

    print("\n" + "="*80)
    print("Merge Complete!")
    print("="*80)

    print("\nNext steps:")
    print("1. Use merged dataset for training:")
    print("   - Overlap molecules have experimental mechanism labels")
    print("   - Use these as validation/test set")
    print("")
    print("2. Training strategies:")
    print("   - Option A: Train on Cornelissen, test on B3DB overlap")
    print("   - Option B: Train on merged dataset (experimental + heuristic)")
    print("   - Option C: Train on Cornelissen, fine-tune on B3DB")
    print("")
    print("3. Files generated:")
    print(f"   - {output_file}")
    print(f"   - {overlap_file}")
    print(f"   - {stats_file}")


if __name__ == "__main__":
    main()
