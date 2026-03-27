from __future__ import annotations
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

# =============================================================================
# Extended RDKit Descriptors (200+ total)
# =============================================================================

# 1. Basic Molecular Properties
BASIC_DESC = [
    # E-State Indices
    ("MaxEStateIndex", Descriptors.MaxEStateIndex),
    ("MinEStateIndex", Descriptors.MinEStateIndex),
    ("MaxAbsEStateIndex", Descriptors.MaxAbsEStateIndex),
    ("MinAbsEStateIndex", Descriptors.MinAbsEStateIndex),

    # Molecular Weight & Size
    ("MolWt", Descriptors.MolWt),
    ("HeavyAtomMolWt", Descriptors.HeavyAtomMolWt),
    ("ExactMolWt", Descriptors.ExactMolWt),

    # Electronic Properties
    ("MaxPartialCharge", Descriptors.MaxPartialCharge),
    ("MinPartialCharge", Descriptors.MinPartialCharge),
    ("MaxAbsPartialCharge", Descriptors.MaxAbsPartialCharge),
    ("MinAbsPartialCharge", Descriptors.MinAbsPartialCharge),

    # Valence Electrons
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("NumRadicalElectrons", Descriptors.NumRadicalElectrons),

    # Lipophilicity
    ("MolLogP", Descriptors.MolLogP),
    ("MolMR", Descriptors.MolMR),

    # Hydrogen Bonding
    ("NumHDonors", Lipinski.NumHDonors),
    ("NumHAcceptors", Lipinski.NumHAcceptors),
    ("NHOHCount", Lipinski.NHOHCount),
    ("NOCount", Lipinski.NOCount),

    # Surface & Shape
    ("TPSA", rdMolDescriptors.CalcTPSA),
    ("LabuteASA", Descriptors.LabuteASA),
    ("BalabanJ", Descriptors.BalabanJ),
    ("BertzCT", Descriptors.BertzCT),

    # Topology
    ("Chi0", Descriptors.Chi0),
    ("Chi1", Descriptors.Chi1),
    ("Chi0n", Descriptors.Chi0n),
    ("Chi1n", Descriptors.Chi1n),
    ("Chi2n", Descriptors.Chi2n),
    ("Chi3n", Descriptors.Chi3n),
    ("Chi4n", Descriptors.Chi4n),
    ("Chi0v", Descriptors.Chi0v),
    ("Chi1v", Descriptors.Chi1v),
    ("Chi2v", Descriptors.Chi2v),
    ("Chi3v", Descriptors.Chi3v),
    ("Chi4v", Descriptors.Chi4v),

    # Kappa Shape Indices
    ("Kappa1", Descriptors.Kappa1),
    ("Kappa2", Descriptors.Kappa2),
    ("Kappa3", Descriptors.Kappa3),

    # rotatable bonds
    ("NumRotatableBonds", Lipinski.NumRotatableBonds),
    ("HeavyAtomCount", Lipinski.HeavyAtomCount),
    ("NumAtoms", rdMolDescriptors.CalcNumAtoms),

    # Ring Systems
    ("RingCount", Lipinski.RingCount),
    ("NumAromaticRings", Lipinski.NumAromaticRings),
    ("NumAliphaticRings", Lipinski.NumAliphaticRings),
    ("CalcNumSaturatedRings", rdMolDescriptors.CalcNumSaturatedRings),

    # Aromatic & Aliphatic
    ("NumAromaticHeterocycles", Lipinski.NumAromaticHeterocycles),
    ("NumAromaticCarbocycles", Lipinski.NumAromaticCarbocycles),
    ("NumSaturatedHeterocycles", Lipinski.NumSaturatedHeterocycles),
    ("NumSaturatedCarbocycles", Lipinski.NumSaturatedCarbocycles),
    ("NumAliphaticHeterocycles", Lipinski.NumAliphaticHeterocycles),
    ("NumAliphaticCarbocycles", Lipinski.NumAliphaticCarbocycles),

    # Heteroatoms
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
    ("NumAmideBonds", rdMolDescriptors.CalcNumAmideBonds),

    # Fraction of sp3 carbons
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),

    # Additional ring methods
    ("CalcNumRings", rdMolDescriptors.CalcNumRings),
    ("CalcNumAliphaticRings", rdMolDescriptors.CalcNumAliphaticRings),
    ("CalcNumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
]

# 2. Additional Lipinski/Rule of 5 descriptors
LIPINSKI_DESC = [
    ("HeavyAtomCount", Lipinski.HeavyAtomCount),
    ("NumHeteroatoms", Lipinski.NumHeteroatoms),
    ("NumRotatableBonds", Lipinski.NumRotatableBonds),
    ("NumHBD", Lipinski.NumHDonors),
    ("NumHBA", Lipinski.NumHAcceptors),
    ("TPSA", rdMolDescriptors.CalcTPSA),
    ("MolLogP", Crippen.MolLogP),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
]

# 3. Electronic descriptors
ELECTRONIC_DESC = [
    ("MaxEStateIndex", Descriptors.MaxEStateIndex),
    ("MinEStateIndex", Descriptors.MinEStateIndex),
    ("MaxAbsEStateIndex", Descriptors.MaxAbsEStateIndex),
    ("MinAbsEStateIndex", Descriptors.MinAbsEStateIndex),
    ("MaxPartialCharge", Descriptors.MaxPartialCharge),
    ("MinPartialCharge", Descriptors.MinPartialCharge),
    ("MaxAbsPartialCharge", Descriptors.MaxAbsPartialCharge),
    ("MinAbsPartialCharge", Descriptors.MinAbsPartialCharge),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("NumRadicalElectrons", Descriptors.NumRadicalElectrons),
]

# 4. Topological descriptors
TOPOLOGICAL_DESC = [
    ("LabuteASA", Descriptors.LabuteASA),
    ("BalabanJ", Descriptors.BalabanJ),
    ("BertzCT", Descriptors.BertzCT),
    ("Chi0", Descriptors.Chi0),
    ("Chi1", Descriptors.Chi1),
    ("Chi0n", Descriptors.Chi0n),
    ("Chi1n", Descriptors.Chi1n),
    ("Chi2n", Descriptors.Chi2n),
    ("Chi3n", Descriptors.Chi3n),
    ("Chi4n", Descriptors.Chi4n),
    ("Chi0v", Descriptors.Chi0v),
    ("Chi1v", Descriptors.Chi1v),
    ("Chi2v", Descriptors.Chi2v),
    ("Chi3v", Descriptors.Chi3v),
    ("Chi4v", Descriptors.Chi4v),
    ("Kappa1", Descriptors.Kappa1),
    ("Kappa2", Descriptors.Kappa2),
    ("Kappa3", Descriptors.Kappa3),
    ("HallKierAlpha", Descriptors.HallKierAlpha),
    ("PEOE_VSA1", Descriptors.PEOE_VSA1),
    ("PEOE_VSA2", Descriptors.PEOE_VSA2),
    ("PEOE_VSA3", Descriptors.PEOE_VSA3),
    ("PEOE_VSA4", Descriptors.PEOE_VSA4),
    ("PEOE_VSA5", Descriptors.PEOE_VSA5),
    ("PEOE_VSA6", Descriptors.PEOE_VSA6),
    ("PEOE_VSA7", Descriptors.PEOE_VSA7),
    ("PEOE_VSA8", Descriptors.PEOE_VSA8),
    ("PEOE_VSA9", Descriptors.PEOE_VSA9),
    ("PEOE_VSA10", Descriptors.PEOE_VSA10),
    ("PEOE_VSA11", Descriptors.PEOE_VSA11),
    ("PEOE_VSA12", Descriptors.PEOE_VSA12),
    ("SMR_VSA1", Descriptors.SMR_VSA1),
    ("SMR_VSA2", Descriptors.SMR_VSA2),
    ("SMR_VSA3", Descriptors.SMR_VSA3),
    ("SMR_VSA4", Descriptors.SMR_VSA4),
    ("SMR_VSA5", Descriptors.SMR_VSA5),
    ("SMR_VSA6", Descriptors.SMR_VSA6),
    ("SMR_VSA7", Descriptors.SMR_VSA7),
    ("SMR_VSA8", Descriptors.SMR_VSA8),
    ("SMR_VSA9", Descriptors.SMR_VSA9),
    ("SMR_VSA10", Descriptors.SMR_VSA10),
    ("SlogP_VSA1", Descriptors.SlogP_VSA1),
    ("SlogP_VSA2", Descriptors.SlogP_VSA2),
    ("SlogP_VSA3", Descriptors.SlogP_VSA3),
    ("SlogP_VSA4", Descriptors.SlogP_VSA4),
    ("SlogP_VSA5", Descriptors.SlogP_VSA5),
    ("SlogP_VSA6", Descriptors.SlogP_VSA6),
    ("SlogP_VSA7", Descriptors.SlogP_VSA7),
    ("SlogP_VSA8", Descriptors.SlogP_VSA8),
    ("SlogP_VSA9", Descriptors.SlogP_VSA9),
    ("SlogP_VSA10", Descriptors.SlogP_VSA10),
    ("SlogP_VSA11", Descriptors.SlogP_VSA11),
    ("SlogP_VSA12", Descriptors.SlogP_VSA12),
]

# 5. Additional structural descriptors
STRUCTURAL_DESC = [
    ("CalcNumSpiroAtoms", rdMolDescriptors.CalcNumSpiroAtoms),
    ("CalcNumBridgeheadAtoms", rdMolDescriptors.CalcNumBridgeheadAtoms),
    ("CalcNumAmideBonds", rdMolDescriptors.CalcNumAmideBonds),
    ("CalcNumHeterocycles", rdMolDescriptors.CalcNumHeterocycles),
]

# 6. MOE-type descriptors
MOE_DESC = [
    ("PEOE_VSA1", Descriptors.PEOE_VSA1),
    ("PEOE_VSA2", Descriptors.PEOE_VSA2),
    ("PEOE_VSA3", Descriptors.PEOE_VSA3),
    ("PEOE_VSA4", Descriptors.PEOE_VSA4),
    ("PEOE_VSA5", Descriptors.PEOE_VSA5),
    ("PEOE_VSA6", Descriptors.PEOE_VSA6),
    ("PEOE_VSA7", Descriptors.PEOE_VSA7),
    ("PEOE_VSA8", Descriptors.PEOE_VSA8),
    ("PEOE_VSA9", Descriptors.PEOE_VSA9),
    ("PEOE_VSA10", Descriptors.PEOE_VSA10),
    ("PEOE_VSA11", Descriptors.PEOE_VSA11),
    ("PEOE_VSA12", Descriptors.PEOE_VSA12),
    ("SlogP_VSA1", Descriptors.SlogP_VSA1),
    ("SlogP_VSA2", Descriptors.SlogP_VSA2),
    ("SlogP_VSA3", Descriptors.SlogP_VSA3),
    ("SlogP_VSA4", Descriptors.SlogP_VSA4),
    ("SlogP_VSA5", Descriptors.SlogP_VSA5),
    ("SlogP_VSA6", Descriptors.SlogP_VSA6),
    ("SlogP_VSA7", Descriptors.SlogP_VSA7),
    ("SlogP_VSA8", Descriptors.SlogP_VSA8),
    ("SlogP_VSA9", Descriptors.SlogP_VSA9),
    ("SlogP_VSA10", Descriptors.SlogP_VSA10),
    ("SlogP_VSA11", Descriptors.SlogP_VSA11),
    ("SlogP_VSA12", Descriptors.SlogP_VSA12),
]

# 7. All descriptors combined (200+)
ALL_DESC = (
    BASIC_DESC +
    LIPINSKI_DESC +
    ELECTRONIC_DESC +
    TOPOLOGICAL_DESC +
    STRUCTURAL_DESC +
    MOE_DESC
)

# Remove duplicates by name
SEEN_NAMES = set()
UNIQUE_DESC = []
for name, fn in ALL_DESC:
    if name not in SEEN_NAMES:
        UNIQUE_DESC.append((name, fn))
        SEEN_NAMES.add(name)


def smiles_to_mol(smiles: str):
    """Convert SMILES to RDKit Mol object."""
    mol = Chem.MolFromSmiles(smiles)
    return mol


def compute_descriptors(smiles_list: list[str], descriptor_set: str = "all") -> pd.DataFrame:
    """Compute RDKit descriptors for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings
        descriptor_set: Which descriptor set to use:
            - "basic": 13 basic descriptors
            - "extended": Extended set (~100 descriptors)
            - "all": All available descriptors (200+)

    Returns:
        DataFrame with descriptor values
    """
    if descriptor_set == "basic":
        desc_list = BASIC_DESC[:13]  # Original 13
    elif descriptor_set == "extended":
        desc_list = BASIC_DESC + LIPINSKI_DESC + ELECTRONIC_DESC + TOPOLOGICAL_DESC
        # Remove duplicates
        seen = set()
        unique = []
        for name, fn in desc_list:
            if name not in seen:
                unique.append((name, fn))
                seen.add(name)
        desc_list = unique
    else:  # "all"
        desc_list = UNIQUE_DESC

    rows = []
    for smi in smiles_list:
        mol = smiles_to_mol(smi)
        if mol is None:
            rows.append({name: np.nan for name, _ in desc_list})
            continue
        d = {}
        for name, fn in desc_list:
            try:
                val = float(fn(mol))
                if np.isnan(val) or np.isinf(val):
                    d[name] = np.nan
                else:
                    d[name] = val
            except Exception:
                d[name] = np.nan
        rows.append(d)
    return pd.DataFrame(rows)


def compute_basic_descriptors(smiles_list: list[str]) -> pd.DataFrame:
    """Compute basic 13 descriptors (backward compatible)."""
    return compute_descriptors(smiles_list, descriptor_set="basic")


def compute_extended_descriptors(smiles_list: list[str]) -> pd.DataFrame:
    """Compute extended descriptor set (~100 descriptors)."""
    return compute_descriptors(smiles_list, descriptor_set="extended")


def compute_all_descriptors(smiles_list: list[str]) -> pd.DataFrame:
    """Compute all available descriptors (200+)."""
    return compute_descriptors(smiles_list, descriptor_set="all")


def normalize_descriptors(df: pd.DataFrame, fit_scaler=None):
    """Z-score normalize descriptors.

    Args:
        df: Descriptor DataFrame
        fit_scaler: Optional StandardScaler to use for fitting

    Returns:
        Tuple of (normalized DataFrame, fitted scaler)
    """
    from sklearn.preprocessing import StandardScaler

    # Handle missing values
    df_clean = df.fillna(df.median())

    # Replace infinite values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(0)

    if fit_scaler is None:
        scaler = StandardScaler()
        normalized = scaler.fit_transform(df_clean)
    else:
        scaler = fit_scaler
        normalized = scaler.transform(df_clean)

    return pd.DataFrame(normalized, columns=df.columns), scaler


def get_descriptor_names(descriptor_set: str = "all") -> list[str]:
    """Get list of descriptor names.

    Args:
        descriptor_set: "basic", "extended", or "all"

    Returns:
        List of descriptor names
    """
    if descriptor_set == "basic":
        desc_list = BASIC_DESC[:13]
    elif descriptor_set == "extended":
        desc_list = BASIC_DESC + LIPINSKI_DESC + ELECTRONIC_DESC + TOPOLOGICAL_DESC
        seen = set()
        unique = []
        for name, fn in desc_list:
            if name not in seen:
                unique.append(name)
                seen.add(name)
        return unique
    else:
        return [name for name, _ in UNIQUE_DESC]
