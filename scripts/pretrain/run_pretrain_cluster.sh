#!/bin/bash
#SBATCH --job-name=b3p-pretrain
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@example.com

# ============================================================
# B3P Pretraining on CFFF Cluster
# ============================================================
# This script runs both graph and SMILES pretraining on ZINC22
#
# Usage:
#   sbatch scripts/pretrain/run_pretrain_cluster.sh
#
# Adjust parameters below before submitting
# ============================================================

set -e  # Exit on error

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "=================================================="
echo "B3P Pretraining - Started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

# --- Configuration ---
PROJECT_DIR="/path/to/your/bbbp-project"  # UPDATE THIS PATH!
DATA_DIR="${PROJECT_DIR}/data/zinc22"
NUM_SAMPLES=1000000  # 1M samples for full pretraining
EPOCHS=100
BATCH_SIZE=128  # Increase for GPU training

# Graph pretraining settings
GRAPH_HIDDEN_DIM=256
GRAPH_NUM_LAYERS=5
GRAPH_LR=1e-3

# Transformer pretraining settings
TRANSFORMER_D_MODEL=512
TRANSFORMER_N_HEADS=8
TRANSFORMER_N_LAYERS=6
TRANSFORMER_LR=1e-4

# --- Setup ---
cd ${PROJECT_DIR}

# Create necessary directories
mkdir -p logs
mkdir -p artifacts/models/pretrain/graph
mkdir -p artifacts/models/pretrain/transformer
mkdir -p data/zinc22/cache

# Load conda environment (adjust if using different env manager)
echo "Loading conda environment..."
source ~/.bashrc  # Or your conda init script
conda activate bbb

# Check GPU
echo "GPU Information:"
nvidia-smi

# ============================================================
# Task 1: Graph Pretraining (GIN)
# ============================================================
echo ""
echo "=================================================="
echo "Task 1: Graph Pretraining (GIN on ZINC22)"
echo "=================================================="
echo "Samples: ${NUM_SAMPLES}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Hidden dim: ${GRAPH_HIDDEN_DIM}"
echo "Num layers: ${GRAPH_NUM_LAYERS}"
echo "=================================================="

python scripts/pretrain/pretrain_graph.py \
    --data_dir ${DATA_DIR} \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --model_type gin \
    --hidden_dim ${GRAPH_HIDDEN_DIM} \
    --num_layers ${GRAPH_NUM_LAYERS} \
    --lr ${GRAPH_LR} \
    --save_dir artifacts/models/pretrain/graph \
    --device auto

echo "Graph pretraining completed at $(date)"

# ============================================================
# Task 2: SMILES Transformer Pretraining
# ============================================================
echo ""
echo "=================================================="
echo "Task 2: SMILES Transformer Pretraining"
echo "=================================================="
echo "Samples: ${NUM_SAMPLES}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Model dim: ${TRANSFORMER_D_MODEL}"
echo "Num layers: ${TRANSFORMER_N_LAYERS}"
echo "=================================================="

python scripts/pretrain/pretrain_smiles.py \
    --data_dir ${DATA_DIR} \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --d_model ${TRANSFORMER_D_MODEL} \
    --n_heads ${TRANSFORMER_N_HEADS} \
    --n_layers ${TRANSFORMER_N_LAYERS} \
    --lr ${TRANSFORMER_LR} \
    --mask_ratio 0.15 \
    --save_dir artifacts/models/pretrain/transformer \
    --device auto

echo "Transformer pretraining completed at $(date)"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=================================================="
echo "All Pretraining Tasks Completed!"
echo "=================================================="
echo "Graph model: artifacts/models/pretrain/graph/gin_pretrained_backbone.pt"
echo "Transformer: artifacts/models/pretrain/transformer/transformer_pretrained_encoder.pt"
echo "Finished at: $(date)"
echo "=================================================="
