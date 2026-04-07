#!/bin/bash
#SBATCH --job-name=b3p-pretrain-test
#SBATCH --output=logs/pretrain_test_%j.out
#SBATCH --error=logs/pretrain_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# ============================================================
# Quick Test - Smoke Test Pretraining
# ============================================================
# Small scale test to verify everything works
# 10K samples, 5 epochs each
# ============================================================

set -e

echo "=================================================="
echo "Pretraining Smoke Test - Started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

PROJECT_DIR="/path/to/your/bbbp-project"  # UPDATE THIS!
DATA_DIR="${PROJECT_DIR}/data/zinc22"

cd ${PROJECT_DIR}
mkdir -p logs

# Load environment
source ~/.bashrc
conda activate bbb

# --- Test 1: Graph Pretraining (10K samples, 5 epochs) ---
echo ""
echo "Test 1: Graph Pretraining (10K samples, 5 epochs)"
echo "=================================================="

python scripts/pretrain/pretrain_graph.py \
    --data_dir ${DATA_DIR} \
    --num_samples 10000 \
    --batch_size 32 \
    --epochs 5 \
    --model_type gin \
    --hidden_dim 128 \
    --num_layers 3 \
    --lr 1e-3 \
    --save_dir artifacts/models/pretrain/graph_test \
    --device auto

echo "Graph test completed!"

# --- Test 2: SMILES Pretraining (10K samples, 5 epochs) ---
echo ""
echo "Test 2: SMILES Transformer Pretraining (10K samples, 5 epochs)"
echo "=================================================="

python scripts/pretrain/pretrain_smiles.py \
    --data_dir ${DATA_DIR} \
    --num_samples 10000 \
    --batch_size 32 \
    --epochs 5 \
    --d_model 256 \
    --n_heads 8 \
    --n_layers 4 \
    --lr 1e-4 \
    --mask_ratio 0.15 \
    --save_dir artifacts/models/pretrain/transformer_test \
    --device auto

echo "Transformer test completed!"

echo ""
echo "=================================================="
echo "Smoke Test Completed Successfully!"
echo "Both models work correctly. Ready for full training."
echo "Finished at: $(date)"
echo "=================================================="
