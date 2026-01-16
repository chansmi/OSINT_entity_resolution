#!/bin/bash
# ============================================================================
# GPT-4.1 Entity Resolution Evaluation
# ============================================================================
# Runs all GPT-4.1 configurations: zero-shot (medium/high) and few-shot (4/8)
#
# SUBMIT: flux batch -t 4:00:00 -N 1 -n 1 jobs/eval_gpt41.sh
# MONITOR: flux jobs -u all
# CANCEL: flux cancel <job_id>
#
# Note: This uses the OpenAI API, so no GPU is required.
# ============================================================================

set -euo pipefail

echo "=============================================="
echo "GPT-4.1 Entity Resolution Evaluation Suite"
echo "Started: $(date)"
echo "Hostname: $(hostname)"
echo "=============================================="

# Activate conda environment
source /usr/WS1/smith585/x86_miniconda/etc/profile.d/conda.sh
conda activate llm-rocm63

# Navigate to project directory
cd /usr/WS1/smith585/codebases/OSINT_entity_resolution

# Load environment variables (OPENAI_API_KEY)
set -a && source .env && set +a

INPUT_FILE="data/samples/sample_1000.json"
MODEL="gpt-4.1-2025-04-14"
PARALLEL=30

echo "Input file: $INPUT_FILE"
echo "Model: $MODEL"
echo "Parallel requests: $PARALLEL"
echo ""

# ============================================================================
# Zero-Shot Evaluations
# ============================================================================

echo "=============================================="
echo "[1/4] Zero-Shot with Medium Reasoning"
echo "=============================================="
python scripts/baselines/llm_zeroshot.py \
    --input "$INPUT_FILE" \
    --model "$MODEL" \
    --reasoning medium \
    --shots 0 \
    --parallel $PARALLEL

echo ""
echo "=============================================="
echo "[2/4] Zero-Shot with High Reasoning"
echo "=============================================="
python scripts/baselines/llm_zeroshot.py \
    --input "$INPUT_FILE" \
    --model "$MODEL" \
    --reasoning high \
    --shots 0 \
    --parallel $PARALLEL

# ============================================================================
# Few-Shot Evaluations (using medium reasoning)
# ============================================================================

echo ""
echo "=============================================="
echo "[3/4] 4-Shot with Medium Reasoning"
echo "=============================================="
python scripts/baselines/llm_zeroshot.py \
    --input "$INPUT_FILE" \
    --model "$MODEL" \
    --reasoning medium \
    --shots 4 \
    --parallel $PARALLEL

echo ""
echo "=============================================="
echo "[4/4] 8-Shot with Medium Reasoning"
echo "=============================================="
python scripts/baselines/llm_zeroshot.py \
    --input "$INPUT_FILE" \
    --model "$MODEL" \
    --reasoning medium \
    --shots 8 \
    --parallel $PARALLEL

echo ""
echo "=============================================="
echo "GPT-4.1 Evaluation Suite Complete"
echo "Finished: $(date)"
echo "=============================================="

# List output files
echo ""
echo "Output files:"
ls -la data/outputs/llm_*gpt*4*1*.json 2>/dev/null || echo "No output files found yet"
