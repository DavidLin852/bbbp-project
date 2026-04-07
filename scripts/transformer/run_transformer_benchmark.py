"""
SMILES Transformer Benchmark Script

Runs Transformer experiments on B3DB classification and regression tasks.
Compatible with existing classical and GNN baseline evaluation protocols.

Usage:
    # Dry run (single seed, quick test)
    python scripts/transformer/run_transformer_benchmark.py --dry_run

    # Classification benchmark
    python scripts/transformer/run_transformer_benchmark.py --tasks classification --seeds 0,1,2,3,4

    # Regression benchmark
    python scripts/transformer/run_transformer_benchmark.py --tasks regression --seeds 0,1,2,3,4

    # Both tasks
    python scripts/transformer/run_transformer_benchmark.py --tasks classification,regression --seeds 0,1,2,3,4
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.transformer.smiles_tokenizer import SMILESTokenizer, create_tokenizer_from_data
from src.transformer.smiles_transformer import TransformerConfig, get_model
from src.transformer.trainer import Trainer, SMILESDataset, collate_fn, evaluate_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SMILES Transformer Benchmark')

    parser.add_argument(
        '--tasks',
        type=str,
        default='classification,regression',
        help='Tasks to run: classification, regression, or both (comma-separated)'
    )

    parser.add_argument(
        '--seeds',
        type=str,
        default='0,1,2,3,4',
        help='Random seeds (comma-separated)'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/splits',
        help='Directory containing data splits'
    )

    parser.add_argument(
        '--split_type',
        type=str,
        default='scaffold',
        choices=['scaffold', 'random'],
        help='Split type to use'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='artifacts/models/transformer',
        help='Directory to save models and results'
    )

    parser.add_argument(
        '--report_dir',
        type=str,
        default='artifacts/reports/transformer',
        help='Directory to save result reports'
    )

    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default=None,
        help='Path to saved tokenizer (will create new if not provided)'
    )

    parser.add_argument(
        '--max_smiles_length',
        type=int,
        default=128,
        help='Maximum SMILES sequence length'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum number of epochs'
    )

    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=15,
        help='Early stopping patience'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Quick test run with 1 seed and reduced epochs'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, auto-detect if not specified)'
    )

    return parser.parse_args()


def load_data_splits(
    data_dir: Path,
    split_type: str,
    seed: int,
    task: str
) -> tuple:
    """Load train/val/test splits for a given seed and task.

    Args:
        data_dir: Base data directory
        split_type: Type of split ('scaffold' or 'random')
        seed: Random seed
        task: 'classification' or 'regression'

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    split_dir = data_dir / f"seed_{seed}" / f"{task}_{split_type}"

    train_df = pd.read_csv(split_dir / 'train.csv')
    val_df = pd.read_csv(split_dir / 'val.csv')
    test_df = pd.read_csv(split_dir / 'test.csv')

    return train_df, val_df, test_df


def get_device(device_arg: str) -> torch.device:
    """Get device for training.

    Args:
        device_arg: Device argument from command line

    Returns:
        torch.device
    """
    if device_arg is not None:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def train_and_evaluate(
    task: str,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: SMILESTokenizer,
    config: TransformerConfig,
    device: torch.device,
    output_dir: Path
) -> dict:
    """Train and evaluate model for one seed.

    Args:
        task: Task type
        seed: Random seed
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        tokenizer: SMILES tokenizer
        config: Model configuration
        device: Training device
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create save directory
    save_dir = output_dir / f"seed_{seed}" / task
    save_dir.mkdir(parents=True, exist_ok=True)

    # Determine label column
    label_col = 'y_cls' if task == 'classification' else 'logBB'

    # Create datasets
    train_dataset = SMILESDataset(
        smiles_list=train_df['SMILES_canon'].tolist(),
        labels=train_df[label_col].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_smiles_length
    )

    val_dataset = SMILESDataset(
        smiles_list=val_df['SMILES_canon'].tolist(),
        labels=val_df[label_col].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_smiles_length
    )

    test_dataset = SMILESDataset(
        smiles_list=test_df['SMILES_canon'].tolist(),
        labels=test_df[label_col].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_smiles_length
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        num_workers=0
    )

    # Create model
    model = get_model(
        task=task,
        vocab_size=len(tokenizer),
        config=config
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        task=task,
        device=device,
        save_dir=str(save_dir)
    )

    # Train
    print(f"\n{'='*60}")
    print(f"Training {task} - Seed {seed}")
    print(f"{'='*60}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = trainer.train(train_loader, val_loader)

    # Evaluate on test set
    test_results = evaluate_model(model, test_loader, device, task)

    # Add metadata
    test_results['seed'] = seed
    test_results['task'] = task
    test_results['best_epoch'] = trainer.best_epoch
    test_results['train_loss_final'] = history['train_loss'][-1]
    test_results['val_loss_final'] = history['val_loss'][-1]

    # Save results
    results_path = save_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\nTest Results:")
    if task == 'classification':
        print(f"  AUC: {test_results['auc']:.4f}")
        print(f"  F1: {test_results['f1']:.4f}")
        print(f"  Accuracy: {test_results['accuracy']:.4f}")
    else:
        print(f"  R²: {test_results['r2']:.4f}")
        print(f"  RMSE: {test_results['rmse']:.4f}")
        print(f"  MAE: {test_results['mae']:.4f}")

    return test_results


def main():
    """Main benchmark function."""
    args = parse_args()

    # Parse arguments
    tasks = args.tasks.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]

    # Dry run adjustments
    if args.dry_run:
        print("DRY RUN MODE: Using 1 seed, reduced epochs")
        seeds = [0]
        args.epochs = 5
        args.early_stopping_patience = 2

    # Create directories
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Get device
    device = get_device(args.device)

    # Load or create tokenizer
    if args.tokenizer_path and Path(args.tokenizer_path).exists():
        print(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = SMILESTokenizer.load(args.tokenizer_path)
    else:
        print("Creating new tokenizer from training data...")
        # Collect all training SMILES across seeds for vocabulary building
        all_smiles = []
        for task in tasks:
            for seed in seeds:
                train_df, _, _ = load_data_splits(
                    Path(args.data_dir),
                    args.split_type,
                    seed,
                    task
                )
                all_smiles.extend(train_df['SMILES_canon'].tolist())

        tokenizer = create_tokenizer_from_data(all_smiles)
        tokenizer_path = output_dir / 'tokenizer.pkl'
        tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")
        print(f"Vocab stats: {tokenizer.get_vocab_stats()}")

    # Create configuration
    config = TransformerConfig(
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        dropout=0.1,
        pooling='mean',
        learning_rate=args.learning_rate,
        weight_decay=1e-5,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        max_smiles_length=args.max_smiles_length
    )

    # Run experiments
    all_results = []

    for task in tasks:
        task_results = []

        for seed in seeds:
            # Load data
            train_df, val_df, test_df = load_data_splits(
                Path(args.data_dir),
                args.split_type,
                seed,
                task
            )

            # Train and evaluate
            result = train_and_evaluate(
                task=task,
                seed=seed,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                tokenizer=tokenizer,
                config=config,
                device=device,
                output_dir=output_dir
            )

            task_results.append(result)
            all_results.append(result)

        # Aggregate results for this task
        task_df = pd.DataFrame(task_results)

        # Calculate mean and std
        if task == 'classification':
            metrics_to_summarize = ['auc', 'f1', 'accuracy', 'precision', 'recall']
        else:
            metrics_to_summarize = ['r2', 'rmse', 'mae']

        summary = {
            'task': task,
            'n_seeds': len(seeds),
            'split_type': args.split_type
        }

        for metric in metrics_to_summarize:
            summary[f'{metric}_mean'] = task_df[metric].mean()
            summary[f'{metric}_std'] = task_df[metric].std()

        print(f"\n{'='*60}")
        print(f"Summary for {task} ({len(seeds)} seeds)")
        print(f"{'='*60}")
        for metric in metrics_to_summarize:
            print(f"  {metric}: {summary[f'{metric}_mean']:.4f} ± {summary[f'{metric}_std']:.4f}")

        # Save task summary
        task_summary_path = report_dir / f'{task}_summary.json'
        with open(task_summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_path = report_dir / f'transformer_benchmark_{args.split_type}_{timestamp}.csv'

    # Flatten results for CSV
    flat_results = []
    for r in all_results:
        flat = {
            'task': r['task'],
            'seed': r['seed']
        }
        if r['task'] == 'classification':
            flat.update({k: v for k, v in r.items() if k in ['auc', 'f1', 'accuracy', 'precision', 'recall']})
        else:
            flat.update({k: v for k, v in r.items() if k in ['r2', 'rmse', 'mae']})
        flat_results.append(flat)

    results_df = pd.DataFrame(flat_results)
    results_df.to_csv(combined_path, index=False)
    print(f"\nSaved results to {combined_path}")

    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)


if __name__ == '__main__':
    main()
