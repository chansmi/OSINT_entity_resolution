#!/bin/bash
# ============================================================================
# Evaluation Job for Fine-Tuned Entity Resolution Models
# ============================================================================
# BEFORE SUBMITTING: flux resource list
# USAGE: flux batch -t 2:00:00 -N 1 -n 1 jobs/eval_finetune.sh <model>
#        Where <model> is: llama-8b or deepseek-14b
# MONITOR: flux jobs -u all
# ============================================================================

set -euo pipefail

MODEL=${1:-llama-8b}

echo "=============================================="
echo "Entity Resolution Model Evaluation: $MODEL"
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

# Set up environment
export HF_HOME=/p/vast1/smith585/caches/hf_home

echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo "Model: $MODEL"
echo ""

# Find latest checkpoint for this model
CHECKPOINT_DIR=$(ls -td /p/vast1/smith585/checkpoints/entity_resolution/${MODEL}_*/final 2>/dev/null | head -1)

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "ERROR: No checkpoint found for model $MODEL"
    echo "Looking in: /p/vast1/smith585/checkpoints/entity_resolution/${MODEL}_*/"
    ls -la /p/vast1/smith585/checkpoints/entity_resolution/ 2>/dev/null || echo "Checkpoint directory does not exist"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT_DIR"
echo ""

# Check GPU availability
echo "GPU Status:"
/opt/rocm/bin/rocm-smi --showmeminfo vram 2>/dev/null || echo "rocm-smi not available"
echo ""

# Run evaluation on universal test set
# Note: Using --no-4bit because bitsandbytes is not available on ROCm/AMD GPUs
echo "Starting evaluation on universal test set..."
echo "Test set: data/processed/test_pairs.json"
flux run -n 1 -g 8 python -u scripts/finetune/inference.py \
    --model $MODEL \
    --adapter-path "$CHECKPOINT_DIR" \
    --input data/processed/test_pairs.json \
    --no-4bit

echo ""
echo "=============================================="
echo "Evaluation Complete"
echo "Finished: $(date)"
echo "=============================================="
