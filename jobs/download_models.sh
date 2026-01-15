#!/bin/bash
# Download models for entity resolution evaluation
# Submit: flux batch -t 120 -N 1 -n 1 jobs/download_models.sh

echo "Starting model download job..."
echo "Hostname: $(hostname)"
echo "Date: $(date)"

# Source environment
source ~/.bashrc
module load rocm/6.3.1
source /usr/WS1/smith585/x86_miniconda/etc/profile.d/conda.sh
conda activate llm-rocm63

# Go to project directory
cd /usr/WS1/smith585/codebases/OSINT_entity_resolution

# Source .env file for tokens
if [ -f .env ]; then
    echo "Loading .env file..."
    set -a  # Export all variables
    source .env
    set +a
fi

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set!"
    echo "Gated models (Llama) will fail."
else
    echo "HF_TOKEN is set ✓"
fi

# Run download script
python scripts/download_models.py

echo ""
echo "Download job complete: $(date)"
