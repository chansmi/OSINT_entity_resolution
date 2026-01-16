#!/bin/bash
# ============================================================================
# DeepSeek-R1-Distill-Qwen-14B Fine-Tuning with Multi-Node DDP (4 nodes, 32 GPUs)
# ============================================================================
# BEFORE SUBMITTING: flux resource list
# SUBMIT: flux batch -t 720m -N 4 -n 4 -g 32 jobs/train_deepseek14b.sh
# MONITOR: flux jobs -u all
# CANCEL: flux cancel <job_id>
# ============================================================================
#
# Training Time Estimate: ~6 hours (4x speedup from 4 nodes)
# - 697k samples × 3 epochs = 2.1M examples
# - 32 GPUs × batch_size=2 × grad_accum=8 = 512 samples/step
# - ~4,100 steps at ~5s/step = ~6 hours
#
# ============================================================================

set -euo pipefail

echo "=============================================="
echo "DeepSeek-R1-14B Fine-Tuning with Multi-Node DDP"
echo "Configuration: 4 nodes, 32 GPUs (MI250X)"
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

# Load environment variables from .env (contains HF_TOKEN and WANDB_API_KEY)
set -a && source .env && set +a

# Set up WandB
export WANDB_PROJECT=osint-entity-resolution
export WANDB_DIR=/p/vast1/smith585/wandb
mkdir -p $WANDB_DIR

# Set HuggingFace cache directory
export HF_HOME=/p/vast1/smith585/caches/hf_home

# Set up distributed training environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29501  # Different port from Llama to allow concurrent jobs
export NCCL_DEBUG=WARN

# Print environment info
echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "HF_HOME: $HF_HOME"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo ""

# Run multi-node distributed training with accelerate
# 4 nodes × 8 GPUs = 32 processes total
# flux run distributes across all allocated nodes
echo "Starting multi-node distributed training..."
echo "Using 4 nodes × 8 GPUs = 32 processes with data parallelism"
echo ""

flux run -N 4 -n 4 bash -c "
    # Each task runs on a different node with FLUX_TASK_RANK = 0,1,2,3
    module load rocm/6.3.1
    source /usr/WS1/smith585/x86_miniconda/etc/profile.d/conda.sh
    conda activate llm-rocm63
    cd /usr/WS1/smith585/codebases/OSINT_entity_resolution
    set -a && source .env && set +a

    export WANDB_PROJECT=osint-entity-resolution
    export WANDB_DIR=/p/vast1/smith585/wandb
    export HF_HOME=/p/vast1/smith585/caches/hf_home
    export NCCL_DEBUG=WARN

    echo \"Node \${FLUX_TASK_RANK}: \$(hostname) starting accelerate launch...\"

    accelerate launch \
        --multi_gpu \
        --num_machines=4 \
        --num_processes=32 \
        --machine_rank=\${FLUX_TASK_RANK} \
        --main_process_ip=${MASTER_ADDR} \
        --main_process_port=${MASTER_PORT} \
        --mixed_precision=bf16 \
        scripts/finetune/train.py --model deepseek-14b --no-4bit
"

echo ""
echo "=============================================="
echo "Training Complete"
echo "Finished: $(date)"
echo "=============================================="

# Show checkpoint location
echo ""
echo "Checkpoints saved to:"
ls -la /p/vast1/smith585/checkpoints/entity_resolution/ | grep deepseek-14b | tail -5
