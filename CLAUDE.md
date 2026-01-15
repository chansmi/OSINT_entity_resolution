# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A benchmark for evaluating entity resolution approaches on OSINT sanctions screening data. The task is determining whether two entity records (persons/organizations) refer to the same real-world entity.

**Dataset**: 755,540 entity pairs from OpenSanctions data (76.9% positive matches, 23.1% negative).

## Key Commands

### Data Loading & Sampling
```bash
# Create stratified sample (maintains 76.9% positive ratio)
python scripts/create_proper_sample.py --n 1000

# Cache full dataset for faster loading
python scripts/cache_full_dataset.py
```

### Running Baselines
```bash
# Simple fuzzy matching (no API/GPU required)
python scripts/baselines/simple_fuzzy.py --input data/samples/sample_1000.json

# Nomenklatura RegressionV1
python scripts/baselines/nomenklatura_v1.py --input data/samples/sample_1000.json

# LLM Zero-Shot with GPT-5 (parallel async)
python scripts/baselines/llm_zeroshot.py --input data/samples/sample_1000.json --parallel 30

# LLM Few-Shot
python scripts/baselines/llm_zeroshot.py --input data/samples/sample_1000.json --parallel 30 --shots 4

# Local LLM (requires GPU via Flux)
flux run -N 1 -n 1 -g 8 python scripts/baselines/llm_local.py --model llama-8b --input data/samples/sample_1000.json
```

### Fine-Tuning Pipeline
```bash
# Prepare data with entity-level splits (prevents leakage)
python scripts/finetune/prepare_data.py \
    --input data/raw/pairs-20251209.json.gz \
    --output-dir data/processed \
    --train-ratio 0.70 --val-ratio 0.05

# Quick fine-tuning test
python scripts/finetune/train.py --model llama-8b --quick --limit 100

# Multi-node training (4 nodes, 32 GPUs)
flux batch -t 720m -N 4 -n 4 -g 32 jobs/train_llama8b.sh

# Inference on fine-tuned model
python scripts/finetune/inference.py --model-path /path/to/checkpoint --input data/samples/sample_1000.json
```

### Model Management
```bash
# Download models from HuggingFace
export HF_TOKEN=your_token
flux run -N 1 -n 1 python scripts/download_models.py

# Test GPU setup
flux run -N 1 -n 1 -g 8 python scripts/baselines/test_gpu_setup.py
```

### Experiment Tracking
```bash
python scripts/experiment.py --list                    # List experiments
python scripts/experiment.py --compare --method llm    # Compare LLM experiments
python scripts/generate_summary.py --filter-method llm # Generate summary report
```

## Architecture

```
scripts/
├── load_data.py              # Data loading (gzip support, caching)
├── evaluate.py               # Metrics (accuracy, precision, recall, F1)
├── experiment.py             # Experiment tracking (JSONL registry)
├── baselines/                # Evaluation scripts
│   ├── simple_fuzzy.py       # Weighted heuristic matching
│   ├── nomenklatura_v1.py    # OpenSanctions algorithm
│   ├── llm_zeroshot.py       # GPT-5 async parallel
│   └── llm_local.py          # Local LLM (Llama-8B, DeepSeek-14B)
├── finetune/                 # Fine-tuning pipeline
│   ├── config.py             # Model configs, LoRA settings
│   ├── prepare_data.py       # Entity-level splits, ChatML format
│   ├── train.py              # QLoRA + SFT training
│   └── inference.py          # Checkpoint inference
└── vendor/nomenklatura/      # Vendored OpenSanctions library

data/
├── raw/                      # Large files (gitignored)
│   └── pairs-20251209.json.gz
├── samples/                  # Small samples (tracked)
│   └── sample_1000.json
├── processed/                # Fine-tuning data (gitignored)
└── outputs/                  # Results (gitignored)

jobs/                         # Flux batch scripts
```

## Data Format

Entity pairs with ground truth labels:
```json
{
  "left": {
    "id": "ofac-40604",
    "caption": "Aliasghar Norouzi",
    "schema": "Person",
    "properties": {
      "alias": ["Ali Asghar Norowzi"],
      "birthDate": ["1962-11-11"],
      "nationality": ["ir"]
    }
  },
  "right": { ... },
  "judgement": "positive"  // or "negative"
}
```

## Key Design Decisions

**Conflict-Focused Prompting**: The LLM baseline uses a prompting strategy where the default assumption is POSITIVE (same entity), only classifying as NEGATIVE when explicit conflicts exist. This reduces false negatives by 97% compared to conservative approaches.

**Entity-Level Splits**: Fine-tuning uses entity-level splits (no entity appears in both train and test) to prevent data leakage and ensure generalization.

**Experiment Tracking**: All runs are logged to `data/outputs/experiments.jsonl` with run ID, git commit, parameters, and metrics for reproducibility.

## Environment Setup

Required environment variables in `.env`:
```bash
WANDB_API_KEY=wandb_v1_...           # Experiment tracking
HF_TOKEN=hf_...                       # HuggingFace model access
```

Model paths on Tioga cluster:
```
/p/vast1/smith585/models/pretrained/
├── meta-llama--Llama-3.1-8B-Instruct
└── deepseek-ai--DeepSeek-R1-Distill-Qwen-14B
```

## Current Baseline Results

| Method | F1 | Precision | Recall | Notes |
|--------|-----|-----------|--------|-------|
| Nomenklatura RegressionV1 | 90.61% | 82.84% | 100% | High false positives |
| LLM Zero-Shot (GPT-5-nano) | 93.10% | 87.84% | 99.02% | Best balanced |
| LLM Ternary Mode | 99.53%* | 99.38% | 99.69% | *62.8% coverage |
