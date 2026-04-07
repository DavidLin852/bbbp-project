#!/bin/bash
#SBATCH --job-name=b3p-smiles-pre
#SBATCH --output=logs/smiles_pretrain_%j.out
#SBATCH --error=logs/smiles_pretrain_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# ============================================================
# SMILES Transformer Pretraining Only (MLM on ZINC22)
# ============================================================
# Usage:
#   sbatch scripts/pretrain/run_smiles_pretrain_cluster.sh
#
# Adjust parameters below
# ============================================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "=================================================="
echo "SMILES Transformer Pretraining - Started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

# --- Configuration ---
PROJECT_DIR="/path/to/your/bbbp-project"  # UPDATE THIS!
DATA_DIR="${PROJECT_DIR}/data/zinc22"

# Training parameters
NUM_SAMPLES=1000000  # 1M samples
EPOCHS=100
BATCH_SIZE=128  # Adjust based on GPU memory
LR=1e-4
MASK_RATIO=0.15

# Model architecture
D_MODEL=512
N_HEADS=8
N_LAYERS=6

SAVE_DIR="${PROJECT_DIR}/artifacts/models/pretrain/transformer"

# --- Setup ---
cd ${PROJECT_DIR}
mkdir -p logs
mkdir -p ${SAVE_DIR}
mkdir -p data/zinc22/cache

# Load environment
echo "Loading conda environment..."
source ~/.bashrc
conda activate bbb

# Check GPU
nvidia-smi

# --- Run Training ---
echo ""
echo "Starting SMILES Transformer pretraining on ZINC22..."
echo "Samples: ${NUM_SAMPLES:,}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Model: ${D_MODEL}d, ${N_HEADS}heads, ${N_LAYERS}layers"
echo ""

python scripts/pretrain/pretrain_smiles.py \
    --data_dir ${DATA_DIR} \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --d_model ${D_MODEL} \
    --n_heads ${N_HEADS} \
    --n_layers ${N_LAYERS} \
    --lr ${LR} \
    --mask_ratio ${MASK_RATIO} \
    --save_dir ${SAVE_DIR} \
    --device auto

echo ""
echo "=================================================="
echo "SMILES Transformer Pretraining Completed!"
echo "Model saved to: ${SAVE_DIR}/transformer_pretrained_encoder.pt"
echo "Tokenizer saved to: ${SAVE_DIR}/tokenizer.pkl"
echo "Finished at: $(date)"
echo "=================================================="
