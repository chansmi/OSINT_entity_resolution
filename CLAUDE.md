# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A benchmark for evaluating entity resolution approaches on OSINT sanctions screening data. The task is determining whether two entity records (persons/organizations) refer to the same real-world entity.

**Dataset**: 755,540 entity pairs from OpenSanctions data (76.9% positive matches, 23.1% negative).

## Quick Validation

```bash
# Verify environment (no GPU/API required)
python scripts/baselines/simple_fuzzy.py --input data/samples/sample_1000.json

# Verify OpenAI API access
python scripts/baselines/llm_zeroshot.py --input data/samples/sample_1000.json --limit 5

# Verify local LLM setup (requires GPU)
flux run -N1 -g8 -t 10m python scripts/baselines/dspy_er/run_dspy.py --model llama-8b --mode zero-shot --limit 5
```

## Key Commands

### Data Loading & Sampling
```bash
# Create stratified sample (maintains 76.9% positive ratio)
python scripts/create_proper_sample.py --n 1000

# Validate existing sample
python scripts/create_proper_sample.py --validate-only data/samples/sample_1000.json

# Cache full dataset for faster loading
python scripts/cache_full_dataset.py
```

### Quick Testing

Most scripts support `--limit N` and `--dry-run` for fast iteration:
```bash
# Limit API/GPU calls for testing
python scripts/baselines/llm_zeroshot.py --limit 10 --input data/samples/sample_1000.json
python scripts/baselines/dspy_er/run_dspy.py --model llama-8b --mode zero-shot --limit 20

# Dry run to preview prompts (no API/GPU usage)
python scripts/baselines/dspy_er/run_dspy.py --dry-run --mode mipro
python scripts/baselines/llm_fewshot.py --dry-run --examples 8
```

### Running Baselines
```bash
# Simple fuzzy matching (no API/GPU required)
python scripts/baselines/simple_fuzzy.py --input data/samples/sample_1000.json

# Nomenklatura RegressionV1
python scripts/baselines/nomenklatura_v1.py --input data/samples/sample_1000.json

# LLM Zero-Shot with GPT-5 (parallel async)
python scripts/baselines/llm_zeroshot.py --input data/samples/sample_1000.json --parallel 30

# LLM Few-Shot with example strategies
python scripts/baselines/llm_fewshot.py --input data/samples/sample_1000.json --examples 8 --strategy diverse

# Few-shot with predefined config
python scripts/baselines/llm_fewshot.py --config few_shot_8

# Dry run (preview prompts without API calls)
python scripts/baselines/llm_fewshot.py --dry-run --examples 8

# View available predefined configs with cost estimates
python scripts/baselines/config.py

# Reasoning effort sweep
python scripts/baselines/reasoning_sweep.py --dry-run
python scripts/baselines/reasoning_sweep.py --model gpt-5-nano --limit 100

# Local LLM (requires GPU via Flux)
flux run -N 1 -n 1 -g 8 python scripts/baselines/llm_local.py --model llama-8b --input data/samples/sample_1000.json
```

### DSPy Prompt Optimization
```bash
# Install DSPy (required)
pip install dspy-ai

# Zero-shot baseline (validate against existing 94.95% F1)
python scripts/baselines/dspy_er/run_dspy.py --model gpt-5-nano --mode zero-shot

# Parallel evaluation (for large datasets with API models)
python scripts/baselines/dspy_er/run_dspy.py --model gpt-5-nano --mode zero-shot --parallel 10

# Dry run (preview prompts without API calls)
python scripts/baselines/dspy_er/run_dspy.py --dry-run --mode zero-shot

# Bootstrap few-shot optimization (8 demonstrations)
python scripts/baselines/dspy_er/run_dspy.py --model gpt-5-nano --mode bootstrap --demos 8

# Bootstrap with teacher-student distillation (improve Llama-8B from 57% → 75-85%)
flux run -N1 -g8 python scripts/baselines/dspy_er/run_dspy.py \
    --model llama-8b --mode bootstrap --teacher gpt-5-nano --demos 8

# MIPROv2 joint instruction + example optimization
python scripts/baselines/dspy_er/run_dspy.py --model gpt-5-nano --mode mipro --candidates 15

# Save/load compiled modules
python scripts/baselines/dspy_er/run_dspy.py --mode bootstrap --save data/outputs/dspy_compiled/bootstrap_8.json
python scripts/baselines/dspy_er/run_dspy.py --load data/outputs/dspy_compiled/bootstrap_8.json

# Ablation study: evaluate with different demo counts
# Use --eval-demos to limit demos when evaluating a loaded module
python scripts/baselines/dspy_er/run_dspy.py \
    --load data/prompts/llama-8b/mipro_20260116.json \
    --eval-demos 0  # MIPROv2 instruction only (no demos)
python scripts/baselines/dspy_er/run_dspy.py \
    --load data/prompts/llama-8b/mipro_20260116.json \
    --eval-demos 4  # MIPROv2 instruction + 4 demos

# Local LLM with MIPROv2 (uses HuggingFace directly, no vLLM needed)
flux run -N1 -n1 -g8 -t 12h python scripts/baselines/dspy_er/run_dspy.py \
    --model llama-8b --mode mipro --candidates 15 --demos 8 \
    --save data/prompts/dspy_compiled/llama8b_mipro.json --export-prompt

# Use explicit model path
python scripts/baselines/dspy_er/run_dspy.py \
    --model llama-8b \
    --model-path /p/vast1/smith585/models/pretrained/meta-llama--Llama-3.1-8B-Instruct \
    --mode zero-shot
```

### Large-Scale Parallel Evaluation
```bash
# Split 10k dataset across 10 parallel GPU jobs
python scripts/baselines/dspy_er/run_parallel_eval.py --model llama-8b --chunks 10 \
    --input data/samples/sample_10000.json --load data/prompts/llama-8b/mipro_20260116.json

# Aggregate results after all jobs complete
python scripts/baselines/dspy_er/run_parallel_eval.py --aggregate --run-id 20260117_123456
```

### Prompt Optimization Workflow
```bash
# Phase 1: Baseline (default prompt on TEST set)
flux run -N1 -n1 -g8 -t 2h python scripts/baselines/dspy_er/run_dspy.py \
    --model llama-8b --mode zero-shot --input data/samples/sample_1000.json

# Phase 2: MIPROv2 Discovery (optimize on DEV set) - submit as batch job
flux batch -N1 -n1 -g8 -t 12h <<'EOF'
#!/bin/bash
source ~/.bashrc && module load rocm/6.3.1
source /usr/workspace/smith585/x86_miniconda/etc/profile.d/conda.sh
conda activate llm-rocm63
cd /usr/workspace/smith585/codebases/OSINT_entity_resolution
python scripts/baselines/dspy_er/run_dspy.py \
    --model llama-8b --mode mipro --candidates 15 --demos 8 \
    --save data/prompts/dspy_compiled/llama8b_mipro.json --export-prompt
EOF

# Phase 3: Evaluate optimized prompt on TEST set
flux run -N1 -n1 -g8 -t 2h python scripts/baselines/dspy_er/run_dspy.py \
    --model llama-8b --load data/prompts/dspy_compiled/llama8b_mipro.json \
    --input data/samples/sample_1000.json

# Manage prompts
python scripts/baselines/prompt_optimization/prompt_store.py list --model llama-8b
python scripts/baselines/prompt_optimization/prompt_store.py best llama-8b
```

### Fine-Tuning Pipeline
```bash
# Prepare SFT data (OpenAI, Alpaca, ShareGPT formats)
python scripts/finetune/prepare_sft_data.py --input data/samples/sample_10000.json
python scripts/finetune/prepare_sft_data.py --preview 3  # Preview examples

# Prepare data with entity-level splits (prevents leakage)
python scripts/finetune/prepare_data.py \
    --input data/raw/pairs-20251209.json.gz \
    --output-dir data/processed \
    --train-ratio 0.70 --val-ratio 0.05

# Quick fine-tuning test (local)
python scripts/finetune/train.py --model llama-8b --quick --limit 100

# Multi-node training (4 nodes, 32 GPUs) - submit via Flux
flux batch -t 720m -N 4 -n 4 -g 32 jobs/train_llama8b.sh

# Inference on fine-tuned model
python scripts/finetune/inference.py --model-path /path/to/checkpoint --input data/samples/sample_1000.json
```

**AMD ROCm Note**: Multi-GPU training on Tioga requires `HIP_VISIBLE_DEVICES` set per-process to avoid GPU conflicts. This is handled automatically in `train.py` but is important if modifying training scripts or running manual distributed training.

### Experiment Tracking
```bash
python scripts/experiment.py --list                    # List recent experiments
python scripts/experiment.py --compare --method llm    # Compare LLM experiments
python scripts/generate_summary.py --filter-method llm # Generate summary report
```

### Job Management
```bash
# Monitor running/pending jobs
flux jobs -u all

# View job output in real-time
flux job attach <job_id>

# Check GPU utilization (on compute node)
/opt/rocm-5.4.2/bin/rocm-smi --showmeminfo vram

# Cancel stuck/wrong jobs
flux cancel <job_id>

# List available GPU resources before submitting
flux resource list
```

## Architecture

### Data Flow
1. Raw data (`pairs-20251209.json.gz`) → `load_data.py` → entity pairs
2. `create_proper_sample.py` → stratified samples for experiments
3. Baselines classify pairs → results to `data/outputs/`
4. `evaluate.py` computes metrics, `experiment.py` tracks runs

### Data Loading Strategies (`scripts/load_data.py`)

| Strategy | Function | Memory | Speed | Use Case |
|----------|----------|--------|-------|----------|
| Iterator | `load_pairs()` | <100MB | Stream | One-pass processing, memory-constrained |
| Sampling | `load_sample()` | <500MB | <1s | Quick experiments |
| Full Cache | `load_full_dataset()` | ~10GB | 5-10s | Batch processing, repeated access |

### API Architecture

The LLM baselines support two OpenAI APIs:
- **Chat Completions API**: Standard models (gpt-5-nano, gpt-5, etc.)
- **Responses API**: Pro models (gpt-5.2-pro) - uses `developer` role instead of `system`

Models requiring Responses API are defined in `RESPONSES_API_MODELS` in `llm_zeroshot.py`.

### Configuration System

`scripts/baselines/config.py` provides:
- `ExperimentConfig` dataclass with all experiment parameters
- `CONFIGS` dict with predefined configurations (baseline, gpt52pro, few_shot_8, etc.)
- `add_config_args()` for consistent CLI argument parsing
- Cost estimation via `config.estimate_cost(n_pairs)`

### Example Selection

`scripts/baselines/example_selector.py` provides three strategies:
- **random**: Baseline random selection from dev set
- **diverse**: Balanced positive/negative, mixed Person/Organization types
- **curated**: Hand-picked examples from `data/examples/curated_examples.json`

**Data Splits** (from `sample_1000.json`):
- DEV set: pairs 0-199 (200 pairs) - used for few-shot example selection and optimization
- TEST set: pairs 200-999 (800 pairs) - used for final evaluation

**Evaluation Protocol**: Never optimize on the test set. MIPROv2 uses DEV for prompt discovery, then the best prompt is evaluated on TEST. This prevents overfitting to the evaluation data.

### DSPy Architecture

`scripts/baselines/dspy_er/` provides DSPy-based prompt optimization:

| File | Purpose |
|------|---------|
| `signatures.py` | Typed I/O signatures (EntityResolution, ConflictDetection) |
| `modules.py` | DSPy modules (EntityResolver, ChainOfThoughtResolver, TwoStageResolver) |
| `data.py` | Adapters to convert entity pairs to DSPy Examples |
| `metrics.py` | DSPy-compatible metrics (exact_match, f1_metric, BatchEvaluator) |
| `optimize.py` | Optimizer wrappers (BootstrapFewShot, MIPROv2), LM configuration |
| `hf_lm.py` | HuggingFace LM wrapper for local models (no vLLM/SGLang needed) |
| `run_dspy.py` | Main CLI with modes: zero-shot, bootstrap, bootstrap-search, mipro |

`scripts/baselines/prompt_optimization/` provides prompt storage and workflow:

| File | Purpose |
|------|---------|
| `prompt_store.py` | Save/load prompts, registry management, DSPy extraction |
| `README.md` | Workflow documentation and usage examples |

**Key concepts:**
- **Signatures** define typed inputs/outputs; docstrings become instructions
- **Modules** wrap signatures with execution strategies (Predict, ChainOfThought)
- **Optimizers** automatically find effective prompts and demonstrations
- **HuggingFace LM** enables MIPROv2 on local models without server setup
- **Prompt Registry** stores optimized prompts with provenance metadata

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

**Class Imbalance Handling**: Fine-tuning config includes class weights (positive: 0.65, negative: 2.17) to handle the 77%/23% class imbalance.

## Environment Setup

Required environment variables in `.env`:
```bash
OPENAI_API_KEY=sk-...                # For LLM baselines
WANDB_API_KEY=wandb_v1_...           # Experiment tracking (fine-tuning)
HF_TOKEN=hf_...                       # HuggingFace model access (fine-tuning)
```

**Dependencies**: Core packages are in `requirements.txt`. Additional packages for specific features:
```bash
pip install dspy-ai                  # DSPy prompt optimization
pip install transformers torch       # Local LLM inference
pip install peft trl datasets        # Fine-tuning pipeline
```

### Model Aliases

Scripts use short aliases that map to full paths:

| Alias | Full Path |
|-------|-----------|
| `llama-8b` | `/p/vast1/smith585/models/pretrained/meta-llama--Llama-3.1-8B-Instruct` |
| `deepseek-14b` | `/p/vast1/smith585/models/pretrained/deepseek-ai--DeepSeek-R1-Distill-Qwen-14B` |
| `deepseek-32b` | `/p/vast1/smith585/models/pretrained/deepseek-ai--DeepSeek-R1-Distill-Qwen-32B` |
| `claude-opus-4-5` | `anthropic/claude-opus-4-5-20251101` (via DSPy Anthropic adapter) |

Override with `--model-path /custom/path` if needed.

Checkpoints saved to: `/p/vast1/smith585/checkpoints/entity_resolution/`

## Current Baseline Results

### API Models (800-pair test set)

| Method | F1 | Precision | Recall | Notes |
|--------|-----|-----------|--------|-------|
| Nomenklatura RegressionV1 | 90.61% | 82.84% | 100% | High false positives |
| LLM Zero-Shot (GPT-5-nano) | 94.95% | 91.77% | 98.37% | Conflict-focused prompt |
| LLM Zero-Shot (GPT-5.2-pro) | 98.53% | 98.37% | 98.69% | Best overall |
| LLM Few-Shot 4-ex (GPT-5.2-pro) | 98.75% | 98.75% | 98.75% | 724/800 evaluated |
| Claude Opus 4.5 MIPROv2 | 99.02%* | - | - | *Dev set only (200 pairs) |

### Local Models with DSPy MIPROv2 (800-pair test set)

| Model | Config | F1 | Precision | Recall | Notes |
|-------|--------|-----|-----------|--------|-------|
| Llama-8B | Zero-shot (default) | 93.55% | 89.32% | 98.21% | Baseline |
| Llama-8B | **MIPROv2 0-shot** | **97.25%** | 96.47% | 98.04% | **Best local model** |
| Llama-8B | MIPROv2 2-shot | 95.46% | 94.85% | 96.08% | Instruction + 2 demos |
| Llama-8B | MIPROv2 4-shot | 95.64% | 94.72% | 96.57% | Instruction + 4 demos |
| Llama-8B | MIPROv2 8-shot | 95.38% | 98.27% | 92.66% | Full MIPROv2 |
| DeepSeek-14B | Zero-shot (20 ex) | 100% | 100% | 100% | Perfect on small test |

### Key Finding: MIPROv2 Instruction Alone Outperforms Few-Shot

The MIPROv2-optimized instruction (0-shot) achieves **97.25% F1**, beating all few-shot configurations. Adding demonstrations actually hurts performance by 1.6-1.9 percentage points. The optimized instruction encodes a "conflict-focused" decision strategy that's more effective than example-based learning.

**MIPROv2 Optimized Instruction** (from `data/prompts/llama-8b/mipro_20260116.json`):
- Frames task as "high-stakes conflict resolution"
- Default: POSITIVE (same entity) unless explicit conflicts found
- 3-step decision process: Look for contradictions → No contradictions = POSITIVE → Only NEGATIVE if conflicts exist
