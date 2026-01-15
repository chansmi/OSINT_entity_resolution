#!/bin/bash
# ============================================================================
# Local LLM Entity Resolution Evaluation
# ============================================================================
# Runs Llama-3.1-8B and DeepSeek-R1-14B with zero-shot and few-shot prompts
#
# SUBMIT: flux batch -t 8:00:00 -N 1 -n 1 -g 8 jobs/eval_local_models.sh
# MONITOR: flux jobs -u all
# CANCEL: flux cancel <job_id>
#
# Note: Requires GPU access via Flux.
# Estimated time: ~6-8 hours for all 6 configurations
# ============================================================================

set -euo pipefail

echo "=============================================="
echo "Local LLM Entity Resolution Evaluation Suite"
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

# Load environment variables
set -a && source .env && set +a

# Set HuggingFace cache
export HF_HOME=/p/vast1/smith585/caches/hf_home

INPUT_FILE="data/samples/sample_1000.json"

echo "Input file: $INPUT_FILE"
echo "Python: $(which python)"
echo ""

# Check GPU availability
echo "GPU Status:"
/opt/rocm/bin/rocm-smi --showmeminfo vram 2>/dev/null || echo "rocm-smi not available"
echo ""

# ============================================================================
# Llama-3.1-8B Evaluations
# ============================================================================

echo "=============================================="
echo "[1/6] Llama-8B: Zero-Shot"
echo "=============================================="
python -u scripts/baselines/llm_local.py \
    --model llama-8b \
    --input "$INPUT_FILE" \
    --shots 0

echo ""
echo "=============================================="
echo "[2/6] Llama-8B: 4-Shot"
echo "=============================================="
python -u scripts/baselines/llm_local.py \
    --model llama-8b \
    --input "$INPUT_FILE" \
    --shots 4

echo ""
echo "=============================================="
echo "[3/6] Llama-8B: 8-Shot"
echo "=============================================="
python -u scripts/baselines/llm_local.py \
    --model llama-8b \
    --input "$INPUT_FILE" \
    --shots 8

# ============================================================================
# DeepSeek-R1-14B (Qwen) Evaluations
# ============================================================================

echo ""
echo "=============================================="
echo "[4/6] DeepSeek-14B: Zero-Shot"
echo "=============================================="
python -u scripts/baselines/llm_local.py \
    --model deepseek-14b \
    --input "$INPUT_FILE" \
    --shots 0

echo ""
echo "=============================================="
echo "[5/6] DeepSeek-14B: 4-Shot"
echo "=============================================="
python -u scripts/baselines/llm_local.py \
    --model deepseek-14b \
    --input "$INPUT_FILE" \
    --shots 4

echo ""
echo "=============================================="
echo "[6/6] DeepSeek-14B: 8-Shot"
echo "=============================================="
python -u scripts/baselines/llm_local.py \
    --model deepseek-14b \
    --input "$INPUT_FILE" \
    --shots 8

echo ""
echo "=============================================="
echo "Local LLM Evaluation Suite Complete"
echo "Finished: $(date)"
echo "=============================================="

# List output files
echo ""
echo "Output files:"
ls -la data/outputs/llm_local_*.json 2>/dev/null || echo "No output files found yet"
