# OSINT Entity Resolution Benchmark

A benchmark for evaluating entity resolution approaches on OSINT sanctions screening data. The task: given two entity records (persons or organizations), determine whether they refer to the same real-world entity.

**Dataset**: 755,540 entity pairs from [OpenSanctions](https://opensanctions.org/) data (76.9% positive matches, 23.1% negative).

## Results

<img width="877" height="537" alt="Screenshot 2026-02-23 at 10 34 37 PM" src="https://github.com/user-attachments/assets/36b1aa0e-f887-4204-ab6c-53e9872eae4b" />


## Quick Start

### 1. Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy and fill in API keys
cp .env.example .env
```

### 2. Data

Download the raw data file and place it in `data/raw/`:

```bash
mkdir -p data/raw
wget -P data/raw/ https://data.opensanctions.org/contrib/training/pairs-20251209.json.gz
```

Create a stratified sample for experiments:

```bash
python scripts/create_proper_sample.py --n 1000
```

### 3. Run Baselines

```bash
# Fuzzy matching (no API required)
python scripts/baselines/simple_fuzzy.py --input data/samples/sample_1000.json

# Nomenklatura rule-based baseline
python scripts/baselines/nomenklatura_v1.py --input data/samples/sample_1000.json

# LLM zero-shot (requires OPENAI_API_KEY)
python scripts/baselines/llm_zeroshot.py --input data/samples/sample_1000.json --parallel 30

# LLM few-shot with diverse example selection
python scripts/baselines/llm_fewshot.py --input data/samples/sample_1000.json --examples 8 --strategy diverse
```

Most scripts support `--limit N` (cap API/GPU calls), `--dry-run` (preview prompts), and `--offset 200` (skip dev set, evaluate on test only).

## DSPy Prompt Optimization

Uses [DSPy](https://dspy.ai/) for automatic prompt optimization via MIPROv2.

```bash
pip install dspy-ai

# Zero-shot baseline
python scripts/baselines/dspy_er/run_dspy.py --model gpt-5-nano --mode zero-shot

# MIPROv2 optimization (discovers optimal instructions on dev set)
python scripts/baselines/dspy_er/run_dspy.py --model gpt-5-nano --mode mipro --candidates 15

# Evaluate saved prompt on test set
python scripts/baselines/dspy_er/run_dspy.py --load data/prompts/llama-8b/mipro_20260116.json --offset 200

# Local models via HuggingFace (requires GPU)
flux run -N1 -g8 python scripts/baselines/dspy_er/run_dspy.py --model llama-8b --mode mipro
```

Optimized prompts are stored in `data/prompts/` with a registry at `data/prompts/registry.json`. Manage with:

```bash
python scripts/baselines/prompt_optimization/prompt_store.py list --model llama-8b
python scripts/baselines/prompt_optimization/prompt_store.py best llama-8b
```

## Fine-Tuning

Infrastructure for supervised fine-tuning of local models with entity-level train/test splits (no entity appears in both sets).

```bash
pip install peft trl datasets

# Prepare SFT data
python scripts/finetune/prepare_sft_data.py --input data/samples/sample_10000.json

# Entity-level splits (prevents data leakage)
python scripts/finetune/prepare_data.py --input data/raw/pairs-20251209.json.gz \
    --output-dir data/processed --train-ratio 0.70 --val-ratio 0.05

# Quick test
python scripts/finetune/train.py --model llama-8b --quick --limit 100

# Inference
python scripts/finetune/inference.py --model-path /path/to/checkpoint --input data/samples/sample_1000.json
```

## Experiment Tracking

All runs are logged to `data/outputs/experiments.jsonl` with run ID, git commit, parameters, and metrics.

```bash
python scripts/experiment.py --list                    # List recent experiments
python scripts/experiment.py --compare --method llm    # Compare LLM experiments
python scripts/generate_summary.py --filter-method llm # Generate summary report
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
  "right": { "..." : "..." },
  "judgement": "positive"
}
```

- **positive**: Records refer to the same real-world entity
- **negative**: Records refer to different entities

## Repository Structure

```
├── data/
│   ├── raw/                              # Raw data (gitignored)
│   ├── samples/                          # Stratified samples (tracked)
│   │   └── sample_1000.json
│   ├── prompts/                          # Optimized prompts with registry
│   │   └── registry.json
│   ├── examples/                         # Curated few-shot examples
│   └── outputs/                          # Experiment results (gitignored)
├── scripts/
│   ├── load_data.py                      # Data loading (iterator, sample, full cache)
│   ├── create_proper_sample.py           # Stratified sampling
│   ├── evaluate.py                       # Evaluation metrics
│   ├── experiment.py                     # Experiment tracking
│   ├── generate_summary.py              # Summary reports
│   ├── baselines/
│   │   ├── simple_fuzzy.py               # Deterministic weighted scoring
│   │   ├── nomenklatura_v1.py            # OpenSanctions RegressionV1
│   │   ├── llm_zeroshot.py              # GPT zero-shot (async parallel)
│   │   ├── llm_fewshot.py              # GPT few-shot with example selection
│   │   ├── llm_local.py                # Local LLM inference (Llama, DeepSeek)
│   │   ├── config.py                    # Experiment configs and cost estimation
│   │   ├── example_selector.py          # Few-shot example strategies
│   │   ├── reasoning_sweep.py           # Reasoning effort ablation
│   │   ├── dspy_er/                     # DSPy prompt optimization module
│   │   │   ├── run_dspy.py              # Main CLI (zero-shot, bootstrap, mipro)
│   │   │   ├── signatures.py            # Typed I/O signatures
│   │   │   ├── modules.py              # DSPy modules
│   │   │   ├── data.py                 # Entity pair → DSPy Example adapters
│   │   │   ├── metrics.py             # DSPy-compatible metrics
│   │   │   ├── optimize.py            # Optimizer wrappers
│   │   │   └── hf_lm.py              # HuggingFace LM (local, no vLLM needed)
│   │   └── prompt_optimization/        # Prompt storage and management
│   │       └── prompt_store.py
│   ├── finetune/                        # Fine-tuning infrastructure
│   │   ├── train.py                     # Training loop (multi-GPU, LoRA)
│   │   ├── inference.py                # Checkpoint evaluation
│   │   ├── prepare_data.py            # Entity-level train/test splits
│   │   ├── prepare_sft_data.py        # SFT data formatting
│   │   ├── data_pipeline.py           # Data loading for training
│   │   └── config.py                  # Model paths and training config
│   └── vendor/
│       └── nomenklatura/               # Vendored nomenklatura (git submodule)
├── notebooks/
│   └── explore_data.ipynb              # EDA notebook
├── requirements.txt
└── .env.example
```

## Design Decisions

**Conflict-focused prompting**: The default assumption is POSITIVE (same entity). Only classify as NEGATIVE when explicit contradictions exist (different birth dates, conflicting IDs, etc.). This reduces false negatives by 97% vs. a conservative "are these the same?" framing.

**DEV/TEST split**: From `sample_1000.json`, pairs 0-199 are DEV (for optimization), pairs 200-999 are TEST (for evaluation). Use `--offset 200` to evaluate on test only.

**Entity-level splits**: Fine-tuning ensures no entity appears in both train and test to prevent data leakage.

**HPC notes**: Local model experiments were run on the Tioga cluster (AMD ROCm GPUs) via the Flux scheduler. Model paths like `llama-8b` map to `/p/vast1/smith585/models/pretrained/...` — override with `--model-path` for other environments.

## License

See [LICENSE](LICENSE) for details.
