"""
SMILES Tokenizer for Transformer-based Molecular Models

Implements character-level tokenization for SMILES strings with support for:
- Single-character tokens (atoms, bonds, parentheses)
- Multi-character tokens (Cl, Br, Si, etc.)
- Special tokens (PAD, UNK, SOS, EOS)
"""

from __future__ import annotations
import re
from typing import List, Dict, Optional, Set
from collections import Counter
import pickle


class SMILESTokenizer:
    """Character-level tokenizer for SMILES strings.

    Tokenization rules:
    - Single atoms: C, N, O, S, P, F, I, etc.
    - Multi-character atoms: Cl, Br, Si, Se, etc.
    - Bonds: -, =, #, $
    - Branches: (, )
    - Rings: 1-9, %10-%99
    - Aromatic: c, n, o, s
    - Special: [H], [NH+], etc.
    """

    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"  # Start of sequence
    EOS_TOKEN = "<EOS>"  # End of sequence

    # Multi-character atomic symbols
    MULTI_CHAR_ATOMS = {
        'Cl', 'Br', 'Si', 'Se', 'As', 'Te', 'B', 'Al', 'Ga', 'In', 'Tl',
        'Sn', 'Pb', 'Bi', 'Po', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Be',
        'Mg', 'Ca', 'Sr', 'Ba', 'Ra', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
        'Co', 'Ni', 'Cu', 'Zn', 'Ag', 'Cd', 'Hg', 'Au', 'Pt', 'Pd', 'Rh',
        'Ru', 'Ir', 'Os', 'Re', 'W', 'Ta', 'Hf', 'Zr', 'Y', 'La', 'Ac',
        'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
        'Tm', 'Yb', 'Lu', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
        'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
    }

    def __init__(self, vocab_size: Optional[int] = None, min_freq: int = 1):
        """Initialize tokenizer.

        Args:
            vocab_size: Maximum vocabulary size (None for unlimited)
            min_freq: Minimum frequency for a token to be included
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self.token_counts: Counter = Counter()

        # Initialize with special tokens
        self._add_special_tokens()

    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        for token in special_tokens:
            self._add_token(token)

    def _add_token(self, token: str):
        """Add a single token to vocabulary."""
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize a SMILES string into tokens.

        Args:
            smiles: SMILES string

        Returns:
            List of tokens
        """
        tokens = []
        i = 0

        while i < len(smiles):
            # Check for two-character atom
            if i + 1 < len(smiles):
                two_char = smiles[i:i+2]
                if two_char in self.MULTI_CHAR_ATOMS:
                    tokens.append(two_char)
                    i += 2
                    continue

            # Check for ring numbers >9 (%10, %11, etc.)
            if smiles[i] == '%' and i + 2 < len(smiles):
                tokens.append(smiles[i:i+3])
                i += 3
                continue

            # Single character token
            tokens.append(smiles[i])
            i += 1

        return tokens

    def build_vocab(self, smiles_list: List[str]):
        """Build vocabulary from a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings
        """
        # Count tokens
        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            self.token_counts.update(tokens)

        # Add tokens to vocabulary
        for token, count in self.token_counts.most_common(self.vocab_size):
            if count >= self.min_freq and token not in self.token_to_idx:
                self._add_token(token)

    def encode(self, smiles: str, add_special_tokens: bool = True) -> List[int]:
        """Encode a SMILES string to token indices.

        Args:
            smiles: SMILES string
            add_special_tokens: Whether to add SOS and EOS tokens

        Returns:
            List of token indices
        """
        tokens = self.tokenize(smiles)

        if add_special_tokens:
            tokens = [self.SOS_TOKEN] + tokens + [self.EOS_TOKEN]

        indices = []
        for token in tokens:
            idx = self.token_to_idx.get(token, self.token_to_idx[self.UNK_TOKEN])
            indices.append(idx)

        return indices

    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token indices to SMILES string.

        Args:
            indices: List of token indices
            skip_special_tokens: Whether to skip special tokens

        Returns:
            SMILES string
        """
        tokens = []
        for idx in indices:
            token = self.idx_to_token.get(idx, self.UNK_TOKEN)
            if skip_special_tokens and token in [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
                continue
            tokens.append(token)

        return ''.join(tokens)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_idx)

    @property
    def pad_token_id(self) -> int:
        """Return PAD token index."""
        return self.token_to_idx[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        """Return UNK token index."""
        return self.token_to_idx[self.UNK_TOKEN]

    @property
    def sos_token_id(self) -> int:
        """Return SOS token index."""
        return self.token_to_idx[self.SOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        """Return EOS token index."""
        return self.token_to_idx[self.EOS_TOKEN]

    def save(self, path: str):
        """Save tokenizer to file.

        Args:
            path: Path to save tokenizer
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'token_to_idx': self.token_to_idx,
                'idx_to_token': self.idx_to_token,
                'token_counts': self.token_counts,
                'vocab_size': self.vocab_size,
                'min_freq': self.min_freq
            }, f)

    @classmethod
    def load(cls, path: str) -> 'SMILESTokenizer':
        """Load tokenizer from file.

        Args:
            path: Path to load tokenizer from

        Returns:
            SMILESTokenizer instance
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        tokenizer = cls(vocab_size=data['vocab_size'], min_freq=data['min_freq'])
        tokenizer.token_to_idx = data['token_to_idx']
        tokenizer.idx_to_token = data['idx_to_token']
        tokenizer.token_counts = data['token_counts']

        return tokenizer

    def get_vocab_stats(self) -> Dict[str, any]:
        """Get vocabulary statistics.

        Returns:
            Dictionary with vocabulary statistics
        """
        return {
            'vocab_size': len(self),
            'total_tokens': sum(self.token_counts.values()),
            'unique_tokens': len(self.token_counts),
            'most_common': self.token_counts.most_common(10)
        }


def create_tokenizer_from_data(
    smiles_list: List[str],
    vocab_size: Optional[int] = None,
    min_freq: int = 1
) -> SMILESTokenizer:
    """Create a tokenizer from a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        vocab_size: Maximum vocabulary size
        min_freq: Minimum token frequency

    Returns:
        Trained SMILESTokenizer
    """
    tokenizer = SMILESTokenizer(vocab_size=vocab_size, min_freq=min_freq)
    tokenizer.build_vocab(smiles_list)
    return tokenizer


def collate_smiles_batch(
    smiles_list: List[str],
    labels: List,
    tokenizer: SMILESTokenizer,
    max_length: Optional[int] = None
) -> Dict[str, any]:
    """Collate a batch of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        labels: List of labels
        tokenizer: SMILESTokenizer instance
        max_length: Maximum sequence length (None for auto)

    Returns:
        Dictionary with input_ids, attention_mask, and labels
    """
    import torch

    # Encode all SMILES
    encoded = [tokenizer.encode(smiles) for smiles in smiles_list]

    # Determine max length
    if max_length is None:
        max_length = max(len(ids) for ids in encoded)

    # Pad sequences
    input_ids = []
    attention_mask = []

    for ids in encoded:
        length = len(ids)
        if length < max_length:
            # Pad
            pad_length = max_length - length
            padded_ids = ids + [tokenizer.pad_token_id] * pad_length
            mask = [1] * length + [0] * pad_length
        else:
            # Truncate
            padded_ids = ids[:max_length]
            mask = [1] * max_length

        input_ids.append(padded_ids)
        attention_mask.append(mask)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.float)
    }
