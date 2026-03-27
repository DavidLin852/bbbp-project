#!/bin/bash
# archive_experiment.sh - Archive experiment results to CFFF outputs
#
# Usage: bash scripts/archive_experiment.sh <experiment_name>
#
# Example:
#   bash scripts/archive_experiment.sh baseline_seed0
#   bash scripts/archive_experiment.sh baseline_full_matrix

set -e  # Exit on error

EXPERIMENT_NAME=${1:-"unnamed_experiment"}
DATE=$(date +%Y-%m-%d)

# CFFF outputs directory
OUTPUT_BASE="/cpfs01/projects-HDD/cfff-98a09c02864d_HDD/lx_24110440019/bbbp_project/outputs"
OUTPUT_DIR="${OUTPUT_BASE}/experiments/${DATE}_${EXPERIMENT_NAME}"

echo "=========================================="
echo "Archiving experiment: ${EXPERIMENT_NAME}"
echo "Date: ${DATE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"/{results,models,logs,scripts}

echo ""
echo "[1/5] Copying benchmark results..."

# Copy benchmark results
if [ -f "artifacts/reports/benchmark_summary.csv" ]; then
    cp artifacts/reports/benchmark_summary.csv "${OUTPUT_DIR}/results/"
    echo "  ✓ benchmark_summary.csv"
fi

if [ -f "artifacts/reports/baseline_results_master.csv" ]; then
    cp artifacts/reports/baseline_results_master.csv "${OUTPUT_DIR}/results/"
    echo "  ✓ baseline_results_master.csv"
fi

if [ -f "artifacts/reports/benchmark_report.txt" ]; then
    cp artifacts/reports/benchmark_report.txt "${OUTPUT_DIR}/results/"
    echo "  ✓ benchmark_report.txt"
fi

echo ""
echo "[2/5] Creating experiment configuration..."

# Create config.json
cat > "${OUTPUT_DIR}/config.json" << EOF
{
  "experiment_name": "${EXPERIMENT_NAME}",
  "date": "${DATE}",
  "timestamp": "$(date -Iseconds)",
  "command": "python scripts/baseline/03_train_baselines.py --seed 0",
  "repository": "$(git remote get-url origin 2>/dev/null || echo 'unknown')",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
}
EOF

echo "  ✓ config.json created"

echo ""
echo "[3/5] Copying analysis scripts..."

# Copy core analysis scripts
for script in scripts/analysis/*.py; do
    if [ -f "$script" ]; then
        cp "$script" "${OUTPUT_DIR}/scripts/"
        echo "  ✓ $(basename $script)"
    fi
done

echo ""
echo "[4/5] Creating README..."

# Create README
cat > "${OUTPUT_DIR}/README.md" << EOF
# Experiment: ${EXPERIMENT_NAME}

**Date:** ${DATE}
**Archive Location:** ${OUTPUT_DIR}

## Description

Archived results from BBB baseline pipeline experiment.

## Files

### Results
- \`benchmark_summary.csv\` - Aggregated benchmark results
- \`baseline_results_master.csv\` - All individual experiment results
- \`benchmark_report.txt\` - Human-readable report

### Configuration
- \`config.json\` - Experiment configuration and git info

### Scripts
- Code versions used for this experiment

## Reproduction

To reproduce these results:

\`\`\`bash
# Activate environment
conda activate bbb-baseline

# Run experiment (see config.json for exact command)
python scripts/analysis/run_baseline_matrix.py
\`\`\`

## Best Baseline

See \`benchmark_report.txt\` for detailed results.
EOF

echo "  ✓ README.md created"

echo ""
echo "[5/5] Finalizing archive..."

# Create experiment summary
echo "Experiment: ${EXPERIMENT_NAME}" > "${OUTPUT_DIR}/summary.txt"
echo "Date: ${DATE}" >> "${OUTPUT_DIR}/summary.txt"
echo "Location: ${OUTPUT_DIR}" >> "${OUTPUT_DIR}/summary.txt"
echo "" >> "${OUTPUT_DIR}/summary.txt"
echo "Files:" >> "${OUTPUT_DIR}/summary.txt"
ls -lh "${OUTPUT_DIR}/results/" >> "${OUTPUT_DIR}/summary.txt"

echo ""
echo "=========================================="
echo "Archiving complete!"
echo ""
echo "Archive location:"
echo "  ${OUTPUT_DIR}"
echo ""
echo "Contents:"
du -sh "${OUTPUT_DIR}"/* | sort -h
echo ""
echo "To view results:"
echo "  cat ${OUTPUT_DIR}/results/benchmark_summary.csv"
echo "  cat ${OUTPUT_DIR}/README.md"
echo "=========================================="
