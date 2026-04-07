#!/bin/bash
#SBATCH --job-name=b3p-graph-pre
#SBATCH --output=logs/graph_pretrain_%j.out
#SBATCH --error=logs/graph_pretrain_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# ============================================================
# Graph Pretraining Only (GIN/GAT on ZINC22)
# ============================================================
# Usage:
#   sbatch scripts/pretrain/run_graph_pretrain_cluster.sh
#
# Adjust parameters below
# ============================================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "=================================================="
echo "Graph Pretraining - Started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

# --- Configuration ---
PROJECT_DIR="/path/to/your/bbbp-project"  # UPDATE THIS!
DATA_DIR="${PROJECT_DIR}/data/zinc22"

# Training parameters
NUM_SAMPLES=1000000  # 1M samples
EPOCHS=100
BATCH_SIZE=128  # Adjust based on GPU memory
LR=1e-3

# Model architecture
MODEL_TYPE="gin"  # or "gat"
HIDDEN_DIM=256
NUM_LAYERS=5

SAVE_DIR="${PROJECT_DIR}/artifacts/models/pretrain/graph"

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
echo "Starting GIN pretraining on ZINC22..."
echo "Samples: ${NUM_SAMPLES:,}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo ""

python scripts/pretrain/pretrain_graph.py \
    --data_dir ${DATA_DIR} \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --model_type ${MODEL_TYPE} \
    --hidden_dim ${HIDDEN_DIM} \
    --num_layers ${NUM_LAYERS} \
    --lr ${LR} \
    --save_dir ${SAVE_DIR} \
    --device auto

echo ""
echo "=================================================="
echo "Graph Pretraining Completed!"
echo "Model saved to: ${SAVE_DIR}/${MODEL_TYPE}_pretrained_backbone.pt"
echo "Finished at: $(date)"
echo "=================================================="
