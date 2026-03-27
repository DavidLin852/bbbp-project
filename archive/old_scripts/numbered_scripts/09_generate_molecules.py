"""
Script 09: Generate molecules using trained VAE/GAN.

Generates new BBB-permeable molecules using trained models
and filters based on BBB prediction, QED, and SA scores.

Usage:
    # Generate with both VAE and GAN
    python scripts/09_generate_molecules.py --n_generate 1000 --strategy both

    # Generate with VAE only
    python scripts/09_generate_molecules.py --n_generate 500 --strategy vae

    # Generate with custom filtering
    python scripts/09_generate_molecules.py --n_generate 1000 --min_qed 0.6 --min_bbb 0.8
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import GenerationConfig, Paths
from src.generation import generate_molecules, GenerationPipeline, create_pipeline
from src.multi_model_predictor import MultiModelPredictor, EnsembleStrategy
from src.utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Generate molecules using VAE/GAN")

    # Generation
    parser.add_argument("--n_generate", type=int, default=1000, help="Number of molecules to generate")
    parser.add_argument("--strategy", type=str, default="both", choices=["vae", "gan", "both"],
                        help="Generation strategy")

    # Model paths
    parser.add_argument("--vae_path", type=str, default=None, help="Path to VAE model")
    parser.add_argument("--gan_path", type=str, default=None, help="Path to GAN model")

    # Filtering
    parser.add_argument("--min_qed", type=float, default=0.5, help="Minimum QED score")
    parser.add_argument("--min_bbb", type=float, default=0.7, help="Minimum BBB probability")
    parser.add_argument("--max_sa", type=float, default=4.0, help="Maximum SA score")

    # Novelty
    parser.add_argument("--check_novelty", action="store_true", default=True, help="Check novelty against training set")
    parser.add_argument("--no_check_novelty", action="store_false", dest="check_novelty", help="Skip novelty check")

    # Output
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    seed_everything(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    paths = Paths()

    # Determine model paths
    vae_path = Path(args.vae_path) if args.vae_path else paths.artifacts / "models" / "vae" / "best.pt"
    gan_path = Path(args.gan_path) if args.gan_path else paths.artifacts / "models" / "gan" / "best.pt"

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = paths.artifacts / "outputs" / "generated_molecules"

    # Create generation config
    cfg = GenerationConfig(
        strategy=args.strategy,
        n_generate=args.n_generate,
        min_qed=args.min_qed,
        min_bbb_prob=args.min_bbb,
        max_sa_score=args.max_sa,
        check_novelty=args.check_novelty,
        output_dir=output_dir,
        vae_model_path=vae_path,
        gan_model_path=gan_path,
    )

    print("=" * 60)
    print("Molecule Generation Pipeline")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Generate: {args.n_generate} molecules")
    print(f"Min QED: {args.min_qed}")
    print(f"Min BBB prob: {args.min_bbb}")
    print(f"Max SA: {args.max_sa}")
    print(f"Check novelty: {args.check_novelty}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Check if models exist
    vae_exists = vae_path.exists() if vae_path else False
    gan_exists = gan_path.exists() if gan_path else False

    if not vae_exists and not gan_exists:
        print("WARNING: No trained models found!")
        print("Generating molecules with placeholder models (results will be limited)")

    # Create pipeline
    print("\nCreating generation pipeline...")
    try:
        pipeline = create_pipeline(
            vae_path=vae_path if vae_exists else None,
            gan_path=gan_path if gan_exists else None,
            cfg=cfg,
            device=device,
        )
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        print("Creating simplified pipeline without pre-trained models...")
        # Create basic pipeline
        bbb_predictor = MultiModelPredictor(
            seed=args.seed,
            strategy=EnsembleStrategy.SOFT_VOTING,
        )
        from src.generation import GenerationPipeline
        pipeline = GenerationPipeline(
            vae_model=None,
            gan_model=None,
            bbb_predictor=bbb_predictor,
            cfg=cfg,
            device=device,
        )

    # Generate molecules
    print("\nGenerating molecules...")
    result = pipeline.generate(
        n_generate=args.n_generate,
        strategy=args.strategy,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Generated: {result.n_generated} molecules")
    print(f"Filtered: {result.n_filtered} molecules")
    print(f"Filter rate: {result.filter_rate:.1%}")
    print(f"Time: {result.generation_time:.1f}s")

    if result.metrics:
        print("\nMetrics:")
        if 'diversity' in result.metrics:
            div = result.metrics['diversity']
            print(f"  Mean Tanimoto: {div.get('mean_tanimoto', 0):.3f}")
        if 'scaffolds' in result.metrics:
            scaffolds = result.metrics['scaffolds']
            print(f"  Unique scaffolds: {scaffolds.get('n_unique_scaffolds', 0)}")

    # Save results
    result.save(output_dir)
    print(f"\nResults saved to: {output_dir}")

    # Show sample molecules
    if result.filtered:
        print("\nSample generated molecules:")
        for i, smi in enumerate(result.filtered[:10]):
            print(f"  {i+1}. {smi}")


if __name__ == "__main__":
    main()
