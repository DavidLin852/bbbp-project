"""
Molecule generation pipeline.

End-to-end pipeline for generating BBB-permeable molecules
using VAE and/or GAN models.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..config import GenerationConfig, Paths
from ..multi_model_predictor import MultiModelPredictor, EnsembleStrategy
from ..vae import MoleculeVAE, generate_molecules as vae_generate
from ..gan import MolGAN, generate_with_gan as gan_generate
from .filter_utils import (
    MoleculeFilter,
    FilteredMolecules,
    compute_diversity,
    compute_scaffolds,
    load_training_smiles,
)


@dataclass
class GenerationResult:
    """Result from molecule generation pipeline."""
    generated: List[str]  # All generated molecules
    filtered: List[str]  # Molecules passing filters
    n_generated: int
    n_filtered: int
    filter_rate: float
    metrics: dict
    generation_time: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        df = pd.DataFrame({
            'SMILES': self.filtered,
            'idx': range(len(self.filtered)),
        })
        return df

    def save(self, output_dir: Path):
        """Save results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save filtered molecules
        df = self.to_dataframe()
        df.to_csv(output_dir / "generated_molecules.csv", index=False)

        # Save metrics
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Save all generated (for analysis)
        with open(output_dir / "all_generated.txt", 'w') as f:
            f.write('\n'.join(self.generated))


class GenerationPipeline:
    """
    End-to-end molecule generation pipeline.

    Integrates VAE/GAN models with filtering and evaluation.
    """

    def __init__(
        self,
        vae_model: Optional[MoleculeVAE] = None,
        gan_model: Optional[MolGAN] = None,
        bbb_predictor: Optional[MultiModelPredictor] = None,
        cfg: GenerationConfig = GenerationConfig(),
        device: Optional[torch.device] = None,
    ):
        self.vae_model = vae_model
        self.gan_model = gan_model
        self.bbb_predictor = bbb_predictor
        self.cfg = cfg
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load training SMILES for novelty check
        self.training_smiles: set = set()
        if cfg.check_novelty:
            self.training_smiles = self._load_training_smiles()

    def _load_training_smiles(self) -> set:
        """Load training SMILES for novelty checking."""
        paths = Paths()
        train_path = paths.data_splits / "seed_0_full" / "train.csv"

        if train_path.exists():
            return load_training_smiles(train_path)
        return set()

    def generate(
        self,
        n_generate: Optional[int] = None,
        strategy: Optional[str] = None,
        show_progress: bool = True,
    ) -> GenerationResult:
        """
        Generate molecules.

        Args:
            n_generate: Number of molecules to generate
            strategy: Generation strategy ("vae", "gan", "both")
            show_progress: Show progress bar

        Returns:
            GenerationResult
        """
        n_generate = n_generate or self.cfg.n_generate
        strategy = strategy or self.cfg.strategy

        start_time = time.time()

        # Generate molecules
        if strategy == "vae":
            generated = self._generate_with_vae(n_generate)
        elif strategy == "gan":
            generated = self._generate_with_gan(n_generate)
        elif strategy == "both":
            # Generate with both and combine
            n_each = n_generate // 2
            vae_gen = self._generate_with_vae(n_each)
            gan_gen = self._generate_with_gan(n_generate - n_each)
            generated = vae_gen + gan_gen
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        generation_time = time.time() - start_time

        # Filter molecules
        filter_result = self._filter_molecules(
            generated,
            show_progress=show_progress,
        )

        # Compute metrics
        metrics = self._compute_metrics(filter_result.filtered)

        result = GenerationResult(
            generated=generated,
            filtered=filter_result.filtered,
            n_generated=len(generated),
            n_filtered=len(filter_result.filtered),
            filter_rate=len(filter_result.filtered) / len(generated) if generated else 0.0,
            metrics=metrics,
            generation_time=generation_time,
        )

        # Save results
        result.save(self.cfg.output_dir)

        return result

    def _generate_with_vae(self, n_samples: int) -> List[str]:
        """Generate using VAE."""
        if self.vae_model is None:
            raise ValueError("VAE model not loaded")

        self.vae_model.eval()
        with torch.no_grad():
            generated = vae_generate(
                self.vae_model,
                n_samples=n_samples,
                device=self.device,
            )
        return generated

    def _generate_with_gan(self, n_samples: int) -> List[str]:
        """Generate using GAN."""
        if self.gan_model is None:
            raise ValueError("GAN model not loaded")

        with torch.no_grad():
            generated = gan_generate(
                self.gan_model,
                n_samples=n_samples,
                device=self.device,
            )
        return generated

    def _filter_molecules(
        self,
        smiles_list: List[str],
        show_progress: bool = True,
    ) -> FilteredMolecules:
        """Filter generated molecules."""
        if self.bbb_predictor is None:
            raise ValueError("BBB predictor not loaded")

        filter_obj = MoleculeFilter(
            min_qed=self.cfg.min_qed,
            min_bbb_prob=self.cfg.min_bbb_prob,
            max_sa_score=self.cfg.max_sa_score,
            training_smiles=self.training_smiles,
            remove_duplicates=self.cfg.remove_duplicates,
        )

        return filter_obj.filter(smiles_list, self.bbb_predictor, show_progress)

    def _compute_metrics(self, smiles_list: List[str]) -> dict:
        """Compute metrics on filtered molecules."""
        diversity = compute_diversity(smiles_list)
        scaffolds = compute_scaffolds(smiles_list)

        return {
            'n_filtered': len(smiles_list),
            'diversity': diversity,
            'scaffolds': scaffolds,
        }


def create_pipeline(
    vae_path: Optional[Path] = None,
    gan_path: Optional[Path] = None,
    bbb_predictor_path: Optional[Path] = None,
    cfg: GenerationConfig = GenerationConfig(),
    device: Optional[torch.device] = None,
) -> GenerationPipeline:
    """
    Create generation pipeline with loaded models.

    Args:
        vae_path: Path to VAE model checkpoint
        gan_path: Path to GAN model checkpoint
        bbb_predictor_path: Path to BBB predictor
        cfg: Generation configuration
        device: Device to use

    Returns:
        GenerationPipeline instance
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load VAE
    vae_model = None
    if vae_path and vae_path.exists():
        # Load VAE checkpoint
        checkpoint = torch.load(vae_path, map_location=device, weights_only=False)
        # Initialize model from checkpoint
        # vae_model = ...
        pass

    # Load GAN
    gan_model = None
    if gan_path and gan_path.exists():
        # Load GAN checkpoint
        checkpoint = torch.load(gan_path, map_location=device, weights_only=False)
        # Initialize model from checkpoint
        # gan_model = ...
        pass

    # Load BBB predictor
    bbb_predictor = None
    if bbb_predictor_path is None:
        # Use default predictor
        bbb_predictor = MultiModelPredictor(
            seed=0,
            strategy=EnsembleStrategy.SOFT_VOTING,
        )

    return GenerationPipeline(
        vae_model=vae_model,
        gan_model=gan_model,
        bbb_predictor=bbb_predictor,
        cfg=cfg,
        device=device,
    )


def generate_molecules(
    n_generate: int = 1000,
    strategy: str = "both",
    vae_path: Optional[Path] = None,
    gan_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    min_qed: float = 0.5,
    min_bbb_prob: float = 0.7,
    max_sa: float = 4.0,
) -> GenerationResult:
    """
    Main interface for molecule generation.

    Args:
        n_generate: Number of molecules to generate
        strategy: Generation strategy ("vae", "gan", "both")
        vae_path: Path to VAE model
        gan_path: Path to GAN model
        output_dir: Output directory
        min_qed: Minimum QED score
        min_bbb_prob: Minimum BBB probability
        max_sa: Maximum SA score

    Returns:
        GenerationResult

    Example:
        >>> result = generate_molecules(n_generate=100, strategy="vae")
        >>> print(f"Generated {result.n_filtered} molecules")
        >>> result.to_dataframe().to_csv("output.csv", index=False)
    """
    cfg = GenerationConfig(
        strategy=strategy,
        n_generate=n_generate,
        min_qed=min_qed,
        min_bbb_prob=min_bbb_prob,
        max_sa_score=max_sa,
    )

    if output_dir is not None:
        cfg = GenerationConfig(
            **{**cfg.__dict__, 'output_dir': output_dir}
        )

    pipeline = create_pipeline(
        vae_path=vae_path,
        gan_path=gan_path,
        cfg=cfg,
    )

    return pipeline.generate(n_generate=n_generate, strategy=strategy)


if __name__ == "__main__":
    # Test generation pipeline
    print("Molecule generation module loaded successfully")
    print(f"Default config: {GenerationConfig()}")
