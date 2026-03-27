"""
Molecular fingerprint generation.

Supports:
- Morgan/ECFP fingerprints
- MACCS keys
- Atom pairs
- FP2 (RDKit)
- Combined fingerprints
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from scipy import sparse


class FingerprintGenerator:
    """
    Generate molecular fingerprints from SMILES.

    Supports multiple fingerprint types and can compute
    individual or combined features.
    """

    def __init__(
        self,
        morgan_radius: int = 2,
        morgan_bits: int = 2048,
        maccs_bits: int = 167,
        atom_pairs_bits: int = 1024,
        atom_pairs_max_dist: int = 3,
        fp2_bits: int = 2048,
    ):
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits
        self.maccs_bits = maccs_bits
        self.atom_pairs_bits = atom_pairs_bits
        self.atom_pairs_max_dist = atom_pairs_max_dist
        self.fp2_bits = fp2_bits

    def morgan_fp_matrix(self, smiles_list: list[str]) -> sparse.csr_matrix:
        """Morgan (ECFP) fingerprint as sparse matrix.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            scipy sparse CSR matrix (N, morgan_bits)
        """
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                arr = np.zeros((self.morgan_bits,), dtype=np.int8)
                fps.append(arr)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.morgan_radius, nBits=self.morgan_bits
            )
            arr = np.zeros((self.morgan_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        X = np.vstack(fps)
        return sparse.csr_matrix(X)

    def maccs_keys_matrix(self, smiles_list: list[str]) -> sparse.csr_matrix:
        """MACCS Keys fingerprint as sparse matrix.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            scipy sparse CSR matrix (N, 167)
        """
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                arr = np.zeros((self.maccs_bits,), dtype=np.int8)
                fps.append(arr)
                continue
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((self.maccs_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        X = np.vstack(fps)
        return sparse.csr_matrix(X)

    def atom_pairs_matrix(self, smiles_list: list[str]) -> sparse.csr_matrix:
        """Atom Pairs fingerprint as sparse matrix.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            scipy sparse CSR matrix (N, atom_pairs_bits)
        """
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                arr = np.zeros((self.atom_pairs_bits,), dtype=np.int8)
                fps.append(arr)
                continue
            fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(
                mol, nBits=self.atom_pairs_bits
            )
            arr = np.zeros((self.atom_pairs_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        X = np.vstack(fps)
        return sparse.csr_matrix(X)

    def fp2_matrix(self, smiles_list: list[str]) -> sparse.csr_matrix:
        """Daylight-type FP2 fingerprint as sparse matrix.

        Uses RDKit's RDKFingerprint as FP2 substitute.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            scipy sparse CSR matrix (N, fp2_bits)
        """
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                arr = np.zeros((self.fp2_bits,), dtype=np.int8)
                fps.append(arr)
                continue
            fp = Chem.RDKFingerprint(mol, fpSize=self.fp2_bits)
            arr = np.zeros((self.fp2_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        X = np.vstack(fps)
        return sparse.csr_matrix(X)

    def compute(
        self,
        smiles_list: list[str],
        fingerprint_type: Literal[
            "morgan", "maccs", "atom_pairs", "fp2", "combined"
        ] = "morgan",
    ) -> sparse.csr_matrix:
        """
        Compute specified fingerprint type.

        Args:
            smiles_list: List of SMILES strings
            fingerprint_type: Type of fingerprint to compute

        Returns:
            Feature matrix (sparse CSR)
        """
        if fingerprint_type == "morgan":
            return self.morgan_fp_matrix(smiles_list)
        elif fingerprint_type == "maccs":
            return self.maccs_keys_matrix(smiles_list)
        elif fingerprint_type == "atom_pairs":
            return self.atom_pairs_matrix(smiles_list)
        elif fingerprint_type == "fp2":
            return self.fp2_matrix(smiles_list)
        elif fingerprint_type == "combined":
            return self._compute_combined(smiles_list)
        else:
            raise ValueError(f"Unknown fingerprint type: {fingerprint_type}")

    def _compute_combined(self, smiles_list: list[str]) -> sparse.csr_matrix:
        """Combine all fingerprint types."""
        matrices = [
            self.morgan_fp_matrix(smiles_list),
            self.maccs_keys_matrix(smiles_list),
            self.atom_pairs_matrix(smiles_list),
            self.fp2_matrix(smiles_list),
        ]
        return sparse.hstack(matrices)

    def get_combined_dim(self) -> int:
        """Get dimension of combined fingerprint."""
        return (
            self.morgan_bits
            + self.maccs_bits
            + self.atom_pairs_bits
            + self.fp2_bits
        )
