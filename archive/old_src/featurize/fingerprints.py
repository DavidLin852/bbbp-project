from __future__ import annotations
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from scipy import sparse

def morgan_fp_matrix(smiles_list: list[str], radius: int, n_bits: int):
    """Morgan (ECFP) fingerprint as sparse matrix.

    Args:
        smiles_list: List of SMILES strings
        radius: Morgan radius (default 2 for ECFP4)
        n_bits: Number of bits (default 2048)

    Returns:
        scipy sparse CSR matrix (N, n_bits)
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            arr = np.zeros((n_bits,), dtype=np.int8)
            fps.append(arr)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    X = np.vstack(fps)
    return sparse.csr_matrix(X)


def maccs_keys_matrix(smiles_list: list[str], n_bits: int = 167):
    """MACCS Keys fingerprint as sparse matrix.

    MACCS structural keys capture 166+ specific substructures
    commonly used in molecular similarity searching.

    Args:
        smiles_list: List of SMILES strings
        n_bits: Number of bits (167 for MACCS keys)

    Returns:
        scipy sparse CSR matrix (N, 167)
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            arr = np.zeros((n_bits,), dtype=np.int8)
            fps.append(arr)
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    X = np.vstack(fps)
    return sparse.csr_matrix(X)


def atom_pairs_matrix(smiles_list: list[str], n_bits: int = 1024, max_distance: int = 3):
    """Atom Pairs fingerprint as sparse matrix.

    Atom pairs encode the presence of atom types at specific
    topological distances.

    Args:
        smiles_list: List of SMILES strings
        n_bits: Number of bits (default 1024)
        max_distance: Maximum topological distance (default 3)

    Returns:
        scipy sparse CSR matrix (N, n_bits)
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            arr = np.zeros((n_bits,), dtype=np.int8)
            fps.append(arr)
            continue
        fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    X = np.vstack(fps)
    return sparse.csr_matrix(X)


def fp2_matrix(smiles_list: list[str], n_bits: int = 2048):
    """Daylight-type FP2 fingerprint as sparse matrix.

    Uses RDKit's RDKit fingerprint as a substitute for FP2.

    Args:
        smiles_list: List of SMILES strings
        n_bits: Number of bits (default 2048)

    Returns:
        scipy sparse CSR matrix (N, n_bits)
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            arr = np.zeros((n_bits,), dtype=np.int8)
            fps.append(arr)
            continue
        # Use RDKit fingerprint as FP2 substitute
        fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    X = np.vstack(fps)
    return sparse.csr_matrix(X)


def all_fingerprints_matrix(smiles_list: dict[str, list[str]]) -> dict[str, sparse.csr_matrix]:
    """Compute all fingerprint types.

    Args:
        smiles_list: Dictionary mapping fingerprint name to SMILES list
                    {'morgan': [...], 'maccs': [...], 'atompairs': [...], 'fp2': [...]}

    Returns:
        Dictionary of sparse matrices for each fingerprint type
    """
    results = {}
    for fp_type, smi_list in smiles_list.items():
        if fp_type == 'morgan':
            results['morgan'] = morgan_fp_matrix(smi_list, radius=2, n_bits=2048)
        elif fp_type == 'maccs':
            results['maccs'] = maccs_keys_matrix(smi_list)
        elif fp_type == 'atompairs':
            results['atompairs'] = atom_pairs_matrix(smi_list)
        elif fp_type == 'fp2':
            results['fp2'] = fp2_matrix(smi_list)
    return results


def combine_fingerprints(morgan: sparse.csr_matrix,
                          maccs: sparse.csr_matrix = None,
                          atompairs: sparse.csr_matrix = None,
                          fp2: sparse.csr_matrix = None) -> sparse.csr_matrix:
    """Combine multiple fingerprint types into a single matrix.

    Args:
        morgan: Morgan fingerprint matrix (required)
        maccs: MACCS keys matrix (optional)
        atompairs: Atom pairs matrix (optional)
        fp2: FP2 matrix (optional)

    Returns:
        Combined sparse CSR matrix
    """
    matrices = [morgan]
    if maccs is not None:
        matrices.append(maccs)
    if atompairs is not None:
        matrices.append(atompairs)
    if fp2 is not None:
        matrices.append(fp2)

    return sparse.hstack(matrices)
