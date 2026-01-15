#!/bin/bash
# Quick test for distributed training with accelerate
set -euo pipefail

echo "=============================================="
echo "Testing Distributed Training with Accelerate"
echo "Started: $(date)"
echo "Hostname: $(hostname)"
echo "=============================================="

module load rocm/6.3.1
source /usr/WS1/smith585/x86_miniconda/etc/profile.d/conda.sh
conda activate llm-rocm63

cd /usr/WS1/smith585/codebases/OSINT_entity_resolution
set -a && source .env && set +a

export HF_HOME=/p/vast1/smith585/caches/hf_home
export WANDB_MODE=disabled

echo ""
echo "Python: $(which python)"
echo "Testing accelerate launch with 8 GPUs..."
echo ""

accelerate launch \
    --multi_gpu \
    --num_processes=8 \
    --mixed_precision=bf16 \
    scripts/finetune/train.py --model llama-8b --no-4bit --quick --limit 50 --no-wandb

echo ""
echo "=============================================="
echo "Test Complete: $(date)"
echo "=============================================="
