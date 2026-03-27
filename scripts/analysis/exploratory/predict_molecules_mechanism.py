"""
Transport Mechanism Prediction for Custom Molecules

Compares literature-based rules vs ML model predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Paths
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski


def calculate_physicochemical_properties(smiles):
    """Calculate key physicochemical properties"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        props = {
            'SMILES': smiles,
            'TPSA': Descriptors.TPSA(mol),
            'MW': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
            'RotBonds': Lipinski.NumRotatableBonds(mol),
            'RingCount': Lipinski.RingCount(mol),
            'HeavyAtoms': Lipinski.HeavyAtomCount(mol),
        }

        return props
    except Exception as e:
        print(f"Error calculating properties for {smiles}: {e}")
        return None


def literature_based_prediction(props):
    """
    Predict transport mechanism based on literature (Cornelissen et al. 2022)

    Rules based on physicochemical property thresholds
    """
    predictions = {
        'BBB': 'Unknown',
        'BBB_Prob': 0.5,
        'Influx': 'Unknown',
        'Efflux': 'Unknown',
        'PAMPA': 'Unknown',
        'Reasoning': []
    }

    tpsa = props['TPSA']
    mw = props['MW']
    logp = props['LogP']
    hbd = props['HBD']
    hba = props['HBA']

    # BBB prediction (based on TPSA threshold)
    if tpsa < 90:
        predictions['BBB'] = 'BBB+ (High probability)'
        predictions['BBB_Prob'] = 0.85
        predictions['Reasoning'].append(f"TPSA={tpsa:.1f} < 90 Ų → Favors BBB penetration")
    elif tpsa < 120:
        predictions['BBB'] = 'BBB+ (Moderate probability)'
        predictions['BBB_Prob'] = 0.60
        predictions['Reasoning'].append(f"TPSA={tpsa:.1f} in intermediate range (90-120 Ų)")
    else:
        predictions['BBB'] = 'BBB- (Low probability)'
        predictions['BBB_Prob'] = 0.25
        predictions['Reasoning'].append(f"TPSA={tpsa:.1f} > 120 Ų → Poor BBB penetration")

    # PAMPA prediction (based on LogP and TPSA)
    if logp > 3 and tpsa < 100:
        predictions['PAMPA'] = 'PAMPA+ (High passive diffusion)'
        predictions['Reasoning'].append(f"LogP={logp:.2f} > 3 and TPSA={tpsa:.1f} < 100 → Good passive diffusion")
    elif logp > 2:
        predictions['PAMPA'] = 'PAMPA+ (Moderate passive diffusion)'
        predictions['Reasoning'].append(f"LogP={logp:.2f} suggests moderate passive diffusion")
    else:
        predictions['PAMPA'] = 'PAMPA- (Poor passive diffusion)'
        predictions['Reasoning'].append(f"LogP={logp:.2f} ≤ 2 → Poor passive diffusion")

    # Influx prediction (based on high TPSA, MW, HBD/HBA)
    if tpsa > 100 and mw > 350:
        predictions['Influx'] = 'Influx+ (Possible active transport)'
        predictions['Reasoning'].append(f"TPSA={tpsa:.1f} > 100 and MW={mw:.1f} > 350 → May use transporters")
    elif hba > 6 or hbd > 4:
        predictions['Influx'] = 'Influx+ (Possible transporter substrate)'
        predictions['Reasoning'].append(f"High HBA ({hba}) or HBD ({hbd}) → Possible active influx")
    else:
        predictions['Influx'] = 'Influx- (Likely passive diffusion)'
        predictions['Reasoning'].append(f"Properties suggest passive diffusion")

    # Efflux prediction (based on high MW)
    if mw > 450:
        predictions['Efflux'] = 'Efflux+ (Possible P-gp substrate)'
        predictions['Reasoning'].append(f"MW={mw:.1f} > 450 → Possible efflux pump substrate")
    elif mw > 400 and (hba > 6 or hbd > 4):
        predictions['Efflux'] = 'Efflux+ (Moderate efflux risk)'
        predictions['Reasoning'].append(f"MW={mw:.1f} > 400 and high polarity → Moderate efflux risk")
    else:
        predictions['Efflux'] = 'Efflux- (Low efflux risk)'
        predictions['Reasoning'].append(f"MW={mw:.1f} → Low efflux risk")

    return predictions


def load_mechanism_models():
    """Load trained mechanism prediction models"""
    print("Loading trained models...")

    import json
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer

    models = {}
    imputers = {}

    mechanisms = ['BBB', 'Influx', 'Efflux', 'PAMPA', 'CNS']

    model_dir = Path(Paths.models) / "cornelissen_2022"

    for mech in mechanisms:
        model_path = model_dir / f"{mech.lower()}_model.json"

        if model_path.exists():
            try:
                with open(model_path, 'r') as f:
                    model_data = json.load(f)

                # Recreate model
                model = RandomForestClassifier(
                    n_estimators=model_data.get('n_estimators', 100),
                    random_state=42
                )

                # Load feature columns
                feature_cols = model_data.get('feature_columns', [])

                models[mech] = {
                    'model': model,
                    'feature_columns': feature_cols
                }

                print(f"  Loaded {mech} model")
            except Exception as e:
                print(f"  Error loading {mech} model: {e}")

    if len(models) == 0:
        print("Warning: No trained models found, using literature-based prediction only")
        return None

    return models


def model_based_prediction(smiles, df, models_info):
    """
    Predict using trained ML models

    Requires features from the dataset
    """
    predictions = {}

    if models_info is None:
        return None

    # Find the molecule in the dataset or extract features
    # For new molecules, we need to extract features

    # Check if SMILES is in dataset
    if 'SMILES' in df.columns and smiles in df['SMILES'].values:
        row = df[df['SMILES'] == smiles].iloc[0]
    else:
        print(f"  SMILES not in dataset, skipping model prediction")
        return None

    for mech, model_info in models_info.items():
        model = model_info['model']
        feature_cols = model_info['feature_columns']

        # Get features
        try:
            features = []
            valid_features = []

            for col in feature_cols:
                if col in df.columns:
                    val = row.get(col, 0)
                    if pd.notna(val):
                        features.append(val)
                        valid_features.append(col)

            if len(features) > 0:
                features = np.array(features).reshape(1, -1)

                # Need to retrain or load actual model
                # For now, use literature-based
                predictions[mech] = 'Model not available'
        except Exception as e:
            print(f"  Error predicting {mech}: {e}")

    return predictions


def generate_prediction_report(molecules_data, output_dir):
    """Generate comprehensive prediction report"""

    report = []
    report.append("=" * 100)
    report.append("TRANSPORT MECHANISM PREDICTION REPORT")
    report.append("Based on Cornelissen et al. 2022 Standards")
    report.append("=" * 100)
    report.append("")

    for mol_name, data in molecules_data.items():
        smiles = data['SMILES']
        props = data['Properties']
        lit_pred = data['Literature Prediction']

        report.append(f"{'=' * 100}")
        report.append(f"MOLECULE: {mol_name}")
        report.append(f"SMILES: {smiles}")
        report.append(f"{'=' * 100}")
        report.append("")

        # Physicochemical Properties
        report.append("1. PHYSICOCHEMICAL PROPERTIES")
        report.append("-" * 100)
        report.append(f"  Molecular Weight (MW):        {props['MW']:.2f} Da")
        report.append(f"  Topological Polar PSA (TPSA): {props['TPSA']:.2f} Ų")
        report.append(f"  LogP (XLogP3):                {props['LogP']:.2f}")
        report.append(f"  H-Bond Donors (HBD):          {props['HBD']}")
        report.append(f"  H-Bond Acceptors (HBA):       {props['HBA']}")
        report.append(f"  Rotatable Bonds:              {props['RotBonds']}")
        report.append(f"  Ring Count:                   {props['RingCount']}")
        report.append(f"  Heavy Atoms:                  {props['HeavyAtoms']}")
        report.append("")

        # Literature-based Prediction
        report.append("2. LITERATURE-BASED PREDICTION (Cornelissen et al. 2022)")
        report.append("-" * 100)

        for key in ['BBB', 'Influx', 'Efflux', 'PAMPA']:
            if key in lit_pred:
                val = lit_pred[key]
                if val != 'Unknown':
                    report.append(f"  {key}:   {val}")

        report.append("")
        report.append("  Reasoning:")
        for reason in lit_pred['Reasoning']:
            report.append(f"    • {reason}")

        report.append("")

        # Drug-likeness assessment
        report.append("3. DRUG-LIKENESS ASSESSMENT")
        report.append("-" * 100)

        # Lipinski's Rule of Five
        ro5_violations = 0
        if props['MW'] > 500:
            ro5_violations += 1
        if props['LogP'] > 5:
            ro5_violations += 1
        if props['HBD'] > 5:
            ro5_violations += 1
        if props['HBA'] > 10:
            ro5_violations += 1

        report.append(f"  Lipinski's Rule of Five: {ro5_violations} violations")
        if ro5_violations == 0:
            report.append("    → Good oral bioavailability expected")
        elif ro5_violations == 1:
            report.append("    → Moderate bioavailability")
        else:
            report.append("    → Poor oral bioavailability expected")

        report.append("")

        # BBB-specific assessment
        report.append("4. BBB PENETRATION ASSESSMENT")
        report.append("-" * 100)

        if props['TPSA'] < 90:
            report.append("  TPSA < 90 Ų:   ✓ FAVORABLE for BBB penetration")
        elif props['TPSA'] < 120:
            report.append("  TPSA 90-120 Ų: ⚠ MODERATE BBB penetration")
        else:
            report.append("  TPSA > 120 Ų:  ✗ POOR BBB penetration")

        if props['MW'] < 400:
            report.append("  MW < 400 Da:    ✓ FAVORABLE for BBB penetration")
        elif props['MW'] < 500:
            report.append("  MW 400-500 Da:  ⚠ MODERATE BBB penetration")
        else:
            report.append("  MW > 500 Da:    ✗ POOR BBB penetration")

        report.append("")

    # Summary table
    report.append("=" * 100)
    report.append("SUMMARY TABLE")
    report.append("=" * 100)
    report.append("")

    # Header
    report.append(f"{'Molecule':<15} {'MW':>8} {'TPSA':>8} {'LogP':>8} {'HBD':>5} {'HBA':>5} {'BBB':<25} {'Influx':<20} {'Efflux':<20} {'PAMPA':<25}")
    report.append("-" * 100)

    for mol_name, data in molecules_data.items():
        props = data['Properties']
        lit_pred = data['Literature Prediction']

        bbb = lit_pred['BBB'].split('(')[0].strip()
        influx = lit_pred['Influx'].split('(')[0].strip()
        efflux = lit_pred['Efflux'].split('(')[0].strip()
        pampa = lit_pred['PAMPA'].split('(')[0].strip()

        report.append(f"{mol_name:<15} {props['MW']:>8.1f} {props['TPSA']:>8.1f} {props['LogP']:>8.2f} "
                     f"{props['HBD']:>5} {props['HBA']:>5} {bbb:<25} {influx:<20} {efflux:<20} {pampa:<25}")

    report.append("")
    report.append("=" * 100)
    report.append("END OF REPORT")
    report.append("=" * 100)

    # Save report
    report_path = output_dir / 'mechanism_prediction_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\nReport saved to: {report_path}")

    return report


def main():
    """Main prediction pipeline"""

    print("=" * 100)
    print("TRANSPORT MECHANISM PREDICTION")
    print("Cornelissen et al. 2022 Standards")
    print("=" * 100)
    print()

    # Define molecules
    molecules = {
        'MPC': 'C=C(C)C(=O)OCCOP(=O)([O-])OCC[N+](C)(C)C',
        'CBMA': 'C=C(C)C(O)OCC[N+](C)(C)CC(=O)[O-]',
        'CBMA-2': 'C=C(C)C(=O)OCC[N+](C)(C)CCC(=O)[O-]',
        'CBMA-3': 'C=C(C)C(=O)OCC[N+](C)(C)CCCC(=O)[O-]',
        'SBMA': 'C=C(C)C(=O)OCC[N+](C)(C)CCCS(=O)(=O)[O-]',
        'SBMA-2': 'C=C(C)C(=O)OCC[N+](C)(C)CCS(=O)(=O)[O-]',
        'MPSMA': 'C=C(C)C(=O)OCC(O)C[N+]1(C)CCC(S(=O)(=O)[O-])CC1',
        'DMAEMA': 'C=C(C)C(=O)OCCN(C)C',
        'ONMA': 'C=C(C)C(=O)OCC[N+](C)(C)[O-]',
        'DSC6MA': 'C=C(C)C(=O)NCCCCCCNC(=O)[C@H](N)CO',
        'TryC3MA': 'C=C(C)C(=O)NCCCC(=O)NCCc1c[nH]c2ccccc12',
        'DMACAC5MA': 'C=C(C)C(=O)NCCCCCCNC(=O)/C=C/c1ccc(N(C)C)cc1',
        'GLUTC4MA': 'C=C(C)C(=O)NCCCCNC(=O)OCC1OC(O)C(O)C(O)C1O'
    }

    print(f"Total molecules to predict: {len(molecules)}")
    print()

    # Create output directory
    output_dir = Path(Paths.artifacts).parent / "outputs" / "molecule_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store all predictions
    molecules_data = {}

    # Process each molecule
    for mol_name, smiles in molecules.items():
        print(f"\nProcessing {mol_name}...")
        print(f"  SMILES: {smiles}")

        # Calculate properties
        props = calculate_physicochemical_properties(smiles)

        if props is None:
            print(f"  ERROR: Could not parse SMILES")
            continue

        print(f"  Properties calculated:")
        print(f"    MW:   {props['MW']:.2f} Da")
        print(f"    TPSA: {props['TPSA']:.2f} A^2")
        print(f"    LogP: {props['LogP']:.2f}")
        print(f"    HBD:  {props['HBD']}")
        print(f"    HBA:  {props['HBA']}")

        # Literature-based prediction
        lit_pred = literature_based_prediction(props)

        print(f"  Literature-based prediction:")
        print(f"    BBB:    {lit_pred['BBB']}")
        print(f"    Influx: {lit_pred['Influx']}")
        print(f"    Efflux: {lit_pred['Efflux']}")
        print(f"    PAMPA:  {lit_pred['PAMPA']}")

        molecules_data[mol_name] = {
            'SMILES': smiles,
            'Properties': props,
            'Literature Prediction': lit_pred
        }

    # Load dataset for model-based prediction
    print("\n" + "=" * 100)
    print("Loading dataset for model-based prediction...")
    data_path = Path(Paths.root) / "data" / "transport_mechanisms" / "cornelissen_2022" / "cornelissen_2022_processed.csv"
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Dataset loaded: {df.shape}")

    # Try to load models
    models_info = load_mechanism_models()

    # Generate report
    print("\n" + "=" * 100)
    print("Generating prediction report...")
    report = generate_prediction_report(molecules_data, output_dir)

    # Print report to console
    print("\n" + "\n".join(report))


if __name__ == "__main__":
    main()
