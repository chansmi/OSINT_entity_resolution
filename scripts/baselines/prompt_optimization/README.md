# Prompt Optimization Workflow

This package implements a scientifically rigorous workflow for optimizing prompts directly on each model using DSPy's MIPROv2 optimizer.

## Key Principle

**Prompts are optimized ON each target model, not transferred from another model.**

This addresses the Peeters & Bizer finding that "no single best prompt exists" across models.

## Workflow Overview

```
Phase 1: BASELINE     → Run all models with DEFAULT prompt on TEST set (800 pairs)
Phase 2: DISCOVERY    → Run DSPy MIPROv2 per model on DEV set (200 pairs)
Phase 3: OPTIMIZED    → Evaluate optimized prompts on TEST set (800 pairs)
Phase 4: SENSITIVITY  → (Optional) Check top-K prompts on DEV for robustness
```

**Data Split:**
- DEV set: Pairs 0-199 (for optimization, example selection)
- TEST set: Pairs 200-999 (for final evaluation - sacred)

## Quick Start

### 1. Run Baseline (Zero-Shot with Default Prompt)

```bash
# For API models (no Flux needed)
python scripts/baselines/dspy_er/run_dspy.py \
    --model gpt-5-nano \
    --mode zero-shot \
    --input data/samples/sample_1000.json

# For local models (requires GPU via Flux)
flux run -N1 -n1 -g8 -t 2h python scripts/baselines/dspy_er/run_dspy.py \
    --model llama-8b \
    --mode zero-shot \
    --input data/samples/sample_1000.json
```

### 2. Run MIPROv2 Optimization (DEV set)

```bash
# For API models (~30-60 minutes)
python scripts/baselines/dspy_er/run_dspy.py \
    --model gpt-5-nano \
    --mode mipro \
    --candidates 15 \
    --demos 8 \
    --save data/prompts/dspy_compiled/gpt5nano_mipro.json \
    --export-prompt

# For local models (6-8 hours, submit as batch job)
flux batch -N1 -n1 -g8 -t 12h <<'EOF'
#!/bin/bash
source ~/.bashrc
module load rocm/6.3.1
source /usr/workspace/smith585/x86_miniconda/etc/profile.d/conda.sh
conda activate llm-rocm63
cd /usr/workspace/smith585/codebases/OSINT_entity_resolution

python scripts/baselines/dspy_er/run_dspy.py \
    --model llama-8b \
    --mode mipro \
    --candidates 15 \
    --demos 8 \
    --save data/prompts/dspy_compiled/llama8b_mipro.json \
    --export-prompt
EOF
```

### 3. Evaluate Optimized Prompt (TEST set)

```bash
# Load compiled module and evaluate
python scripts/baselines/dspy_er/run_dspy.py \
    --model llama-8b \
    --load data/prompts/dspy_compiled/llama8b_mipro.json \
    --input data/samples/sample_1000.json
```

### 4. Compare Results

```bash
python scripts/experiment.py --compare --method dspy
```

## File Structure

```
data/prompts/
├── registry.json              # Index of all prompts with metadata
├── default/
│   └── conflict_focused_v1.json
├── gpt-5-nano/
│   └── mipro_20260116_v1.json
├── llama-8b/
│   └── mipro_20260116_v1.json
└── dspy_compiled/             # Raw DSPy module checkpoints
    └── llama8b_mipro.json

scripts/baselines/prompt_optimization/
├── __init__.py
├── prompt_store.py            # Save/load prompts, registry management
└── README.md                  # This file
```

## Prompt JSON Schema

```json
{
  "prompt_id": "llama-8b_mipro_20260116_v1",
  "model": "llama-8b",
  "description": "MIPROv2 optimized prompt for Llama-8B",
  "system_prompt": "...",
  "user_template": "...",
  "demonstrations": [...],
  "provenance": {
    "optimizer": "mipro",
    "dev_f1": 0.75,
    "optimization_date": "2026-01-16",
    "git_commit": "abc123"
  }
}
```

## API Reference

### prompt_store.py

```python
from scripts.baselines.prompt_optimization import (
    save_prompt,
    load_prompt,
    get_best_prompt,
    list_prompts,
)

# Save an optimized prompt
prompt_id = save_prompt(
    prompt_dict,
    model="llama-8b",
    optimizer="mipro",
    dev_f1=0.75,
)

# Load a specific prompt
prompt = load_prompt("llama-8b_mipro_20260116_v1")

# Get the best prompt for a model (by dev_f1)
best = get_best_prompt("llama-8b")

# List all prompts for a model
prompts = list_prompts(model="llama-8b")
```

## Estimated Runtimes

| Model | MIPROv2 (15 candidates, 200 dev pairs) |
|-------|----------------------------------------|
| GPT-5-nano (API) | ~30-60 minutes |
| Llama-8B (local) | ~6-8 hours |
| DeepSeek-14B (local) | ~8-10 hours |

## Technical Details

### HuggingFace LM Wrapper

For local models, we use a custom `HuggingFaceLanguageModel` class that wraps HuggingFace transformers directly:

```python
from scripts.baselines.dspy_er.hf_lm import create_hf_lm
import dspy

lm = create_hf_lm("llama-8b")
dspy.configure(lm=lm)
```

This approach:
- Avoids the complexity of running a separate vLLM/SGLang server
- Works well for MIPROv2 optimization on GPU clusters
- Completes within the 12-hour Flux job limit

### Alternative: vLLM Server Mode

If you need faster inference (e.g., for larger-scale experiments), you can use vLLM:

```bash
# Start vLLM server (in separate terminal/job)
python -m vllm.entrypoints.openai.api_server \
    --model /p/vast1/smith585/models/pretrained/meta-llama--Llama-3.1-8B-Instruct

# Run with --use-vllm flag
python scripts/baselines/dspy_er/run_dspy.py \
    --model llama-8b \
    --use-vllm \
    --mode mipro
```

## Success Criteria

1. ✅ HuggingFace LM wrapper loads local models and generates text
2. ✅ DSPy zero-shot runs without errors on local models
3. ✅ DSPy MIPROv2 completes optimization (6-8 hour job)
4. ✅ MIPROv2-optimized prompt improves F1 over baseline by 15%+
5. ✅ Compiled modules saved/loaded correctly
6. ✅ Experiment tracking captures optimization metadata

## Expected Results

| Model | Baseline F1 | MIPROv2 F1 | Δ |
|-------|-------------|------------|---|
| GPT-5-nano | 94.95% | ~96-97% | +1-2% |
| Llama-8B | 57.29% | ~75-80% | +18-23% |
| DeepSeek-14B | TBD | TBD | TBD |
