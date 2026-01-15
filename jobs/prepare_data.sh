#!/bin/bash
# ============================================================================
# Data Preparation Job - Creates 70/5/25 train/val/test splits
# ============================================================================
# BEFORE SUBMITTING: flux resource list
# SUBMIT: flux batch -t 2:00:00 -N 1 -n 4 jobs/prepare_data.sh
# MONITOR: flux jobs -u all
# ============================================================================

set -euo pipefail

echo "=============================================="
echo "Entity Resolution Data Preparation (25% Holdout)"
echo "Started: $(date)"
echo "Hostname: $(hostname)"
echo "=============================================="

# Load ROCm module
module load rocm/6.3.1

# Activate conda environment
source /usr/WS1/smith585/x86_miniconda/etc/profile.d/conda.sh
conda activate llm-rocm63

# Navigate to project directory
cd /usr/WS1/smith585/codebases/OSINT_entity_resolution

# Load environment variables from .env
set -a && source .env && set +a

echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo ""

# Run data preparation with 70/5/25 split
python scripts/finetune/prepare_data.py \
    --input data/raw/pairs-20251209.json.gz \
    --output-dir data/processed \
    --train-ratio 0.70 \
    --val-ratio 0.05 \
    --seed 42

echo ""
echo "=============================================="
echo "Data Preparation Complete"
echo "Finished: $(date)"
echo "=============================================="

# Verify outputs
echo ""
echo "Output files:"
ls -lh data/processed/

echo ""
echo "Line counts (ChatML format):"
wc -l data/processed/*.jsonl

echo ""
echo "Universal test set (V2 format for ALL methods):"
ls -lh data/processed/test_pairs.json

echo ""
echo "Metadata with checksum:"
cat data/processed/metadata.json | python -m json.tool | head -40

echo ""
echo "Verify test set is loadable by all methods:"
python -c "
import json
with open('data/processed/test_pairs.json') as f:
    data = json.load(f)
print(f'✓ Loaded {len(data[\"pairs\"]):,} test pairs')
print(f'✓ Format: V2 with metadata')
print(f'✓ Sample pair keys: {list(data[\"pairs\"][0].keys())}')
"
