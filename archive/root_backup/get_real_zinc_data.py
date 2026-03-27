"""
Get REAL ZINC20 Data
"""
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

print("="*80)
print("Getting REAL ZINC Data")
print("="*80)

print("""
Current data is NOT from ZINC20!
- Source: BBB dataset (4,611 molecules)
- All are drug-like molecules
- Limited diversity

Let's try to get REAL ZINC data...
""")

# Try ZINC12 (most stable)
print("\nTrying ZINC12 drug-like subset...")
try:
    url = "http://zinc.docking.org/db/bysubset/drug/like/1/1/1_smi.ows"
    response = requests.get(url, stream=True, timeout=120)

    if response.status_code == 200:
        smiles_list = []
        for line in tqdm(response.iter_lines(decode_unicode=True)):
            if line:
                parts = line.strip().split()
                if len(parts) >= 2:
                    smiles_list.append({'SMILES': parts[0], 'ZINC_ID': parts[1]})

            if len(smiles_list) >= 50000:  # Get 50k molecules
                break

        print(f"\nSuccessfully downloaded {len(smiles_list):,} molecules from ZINC12!")

        # Save
        output_dir = Path("data/zinc20")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "zinc12_druglike_50k.csv"

        df = pd.DataFrame(smiles_list)
        df.to_csv(output_file, index=False)

        print(f"Saved to: {output_file}")
        print("\nNow you can use REAL ZINC data:")
        print(f"python pretrain_zinc20.py --step download --num-molecules {len(smiles_list)}")

        # Show some samples
        print("\nSample molecules:")
        for i in range(min(5, len(smiles_list))):
            print(f"  {smiles_list[i]['ZINC_ID']}: {smiles_list[i]['SMILES'][:60]}...")

    else:
        print(f"Download failed: HTTP {response.status_code}")

except Exception as e:
    print(f"Error: {e}")
    print("\nFalling back to alternative methods...")

    print("""
Alternative ways to get ZINC data:

1. Install zincdownloader:
   pip install zincdownloader

2. Download from ChEMBL:
   wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLv33/chembl_33.smi.gz

3. Use your own SMILES file
    """)

print("\n"+"="*80)
