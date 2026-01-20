#!/bin/bash
# submit_overnight.sh - Submit all overnight jobs with dependencies
# Run once, then check results in the morning
#
# Usage: bash submit_overnight.sh
#
# This script submits:
# Phase 1: Quick Results (1000 pairs)
#   - Llama-8B MIPROv2 0-shot, 2-shot, 4-shot evaluations
#   - DeepSeek-14b Zero-shot evaluation
#   - DeepSeek-14b MIPROv2 Optimization (5 candidates)
#
# Phase 2: Full 10k Evaluation (depends on Phase 1)
#   - Llama-8B 10k MIPROv2 0-shot
#   - Llama-8B 10k MIPROv2 8-shot
#   - DeepSeek-14b 10k Zero-shot
#   - DeepSeek-14b 10k MIPROv2 (after optimization)

set -e
cd /usr/workspace/smith585/codebases/OSINT_entity_resolution
export PYTHONPATH=$PWD:$PYTHONPATH

echo "=== Submitting Overnight Jobs ==="
echo "Started: $(date)"
echo ""

# ============================================================
# PHASE 1: Quick Results (1000 pairs)
# ============================================================

echo "--- Phase 1: Quick Results (1000 pairs) ---"
echo ""

# Job 1: Llama-8B MIPROv2 0-shot (instruction only)
JOB_LLAMA_0=$(flux batch -N1 -n1 -g8 -t 3h --job-name=llama8b_mipro0 -o flux-llama8b-mipro0.out <<'SCRIPT'
#!/bin/bash
echo "=== Llama-8B MIPROv2 0-shot ==="
echo "Started: $(date)"

source ~/.bashrc && module load rocm/6.3.1
export CONDAPATH=/usr/workspace/smith585/x86_miniconda
eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
conda activate base
cd /usr/workspace/smith585/codebases/OSINT_entity_resolution
export PYTHONPATH=$PWD:$PYTHONPATH

python -m scripts.baselines.dspy_er.run_dspy \
    --model llama-8b \
    --load data/prompts/llama-8b/mipro_20260116.json \
    --input data/samples/sample_1000.json \
    --eval-demos 0

echo ""
echo "Completed: $(date)"
SCRIPT
)
echo "Submitted llama8b_mipro0: $JOB_LLAMA_0"

# Job 2: Llama-8B MIPROv2 2-shot
JOB_LLAMA_2=$(flux batch -N1 -n1 -g8 -t 3h --job-name=llama8b_mipro2 -o flux-llama8b-mipro2.out <<'SCRIPT'
#!/bin/bash
echo "=== Llama-8B MIPROv2 2-shot ==="
echo "Started: $(date)"

source ~/.bashrc && module load rocm/6.3.1
export CONDAPATH=/usr/workspace/smith585/x86_miniconda
eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
conda activate base
cd /usr/workspace/smith585/codebases/OSINT_entity_resolution
export PYTHONPATH=$PWD:$PYTHONPATH

python -m scripts.baselines.dspy_er.run_dspy \
    --model llama-8b \
    --load data/prompts/llama-8b/mipro_20260116.json \
    --input data/samples/sample_1000.json \
    --eval-demos 2

echo ""
echo "Completed: $(date)"
SCRIPT
)
echo "Submitted llama8b_mipro2: $JOB_LLAMA_2"

# Job 3: Llama-8B MIPROv2 4-shot
JOB_LLAMA_4=$(flux batch -N1 -n1 -g8 -t 3h --job-name=llama8b_mipro4 -o flux-llama8b-mipro4.out <<'SCRIPT'
#!/bin/bash
echo "=== Llama-8B MIPROv2 4-shot ==="
echo "Started: $(date)"

source ~/.bashrc && module load rocm/6.3.1
export CONDAPATH=/usr/workspace/smith585/x86_miniconda
eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
conda activate base
cd /usr/workspace/smith585/codebases/OSINT_entity_resolution
export PYTHONPATH=$PWD:$PYTHONPATH

python -m scripts.baselines.dspy_er.run_dspy \
    --model llama-8b \
    --load data/prompts/llama-8b/mipro_20260116.json \
    --input data/samples/sample_1000.json \
    --eval-demos 4

echo ""
echo "Completed: $(date)"
SCRIPT
)
echo "Submitted llama8b_mipro4: $JOB_LLAMA_4"

# Job 4: DeepSeek-14b Zero-shot (1000 pairs)
JOB_DS_ZERO=$(flux batch -N1 -n1 -g8 -t 6h --job-name=deepseek14b_zero -o flux-deepseek14b-zero.out <<'SCRIPT'
#!/bin/bash
echo "=== DeepSeek-14b Zero-shot ==="
echo "Started: $(date)"

source ~/.bashrc && module load rocm/6.3.1
export CONDAPATH=/usr/workspace/smith585/x86_miniconda
eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
conda activate base
cd /usr/workspace/smith585/codebases/OSINT_entity_resolution
export PYTHONPATH=$PWD:$PYTHONPATH

python -m scripts.baselines.dspy_er.run_dspy \
    --model deepseek-14b \
    --mode zero-shot \
    --input data/samples/sample_1000.json

echo ""
echo "Completed: $(date)"
SCRIPT
)
echo "Submitted deepseek14b_zero: $JOB_DS_ZERO"

# Job 5: DeepSeek-14b MIPROv2 Optimization (5 candidates = faster)
JOB_DS_MIPRO=$(flux batch -N1 -n1 -g8 -t 12h --job-name=deepseek14b_mipro_opt -o flux-deepseek14b-mipro-opt.out <<'SCRIPT'
#!/bin/bash
echo "=== DeepSeek-14b MIPROv2 Optimization (5 candidates) ==="
echo "Started: $(date)"

source ~/.bashrc && module load rocm/6.3.1
export CONDAPATH=/usr/workspace/smith585/x86_miniconda
eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
conda activate base
cd /usr/workspace/smith585/codebases/OSINT_entity_resolution
export PYTHONPATH=$PWD:$PYTHONPATH

python -m scripts.baselines.dspy_er.run_dspy \
    --model deepseek-14b \
    --mode mipro \
    --candidates 5 \
    --demos 4 \
    --input data/samples/sample_1000.json \
    --save data/prompts/deepseek-14b/mipro_20260118.json

echo ""
echo "Completed: $(date)"
SCRIPT
)
echo "Submitted deepseek14b_mipro_opt: $JOB_DS_MIPRO"

echo ""
echo "--- Phase 1 Complete ---"
echo ""

# ============================================================
# PHASE 2: Full 10k Evaluation (depends on Phase 1)
# ============================================================

echo "--- Phase 2: Full 10k Evaluation (with dependencies) ---"
echo ""

# Job 6: Llama-8B 10k MIPROv2 0-shot (depends on JOB_LLAMA_0)
JOB_LLAMA_10K_0=$(flux batch -N1 -n1 -g8 -t 3h \
    --job-name=llama8b_10k_0shot \
    --dependency=afterany:$JOB_LLAMA_0 \
    -o flux-llama8b-10k-mipro0.out <<'SCRIPT'
#!/bin/bash
echo "=== Llama-8B 10k MIPROv2 0-shot ==="
echo "Started: $(date)"

source ~/.bashrc && module load rocm/6.3.1
export CONDAPATH=/usr/workspace/smith585/x86_miniconda
eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
conda activate base
cd /usr/workspace/smith585/codebases/OSINT_entity_resolution
export PYTHONPATH=$PWD:$PYTHONPATH

python scripts/baselines/dspy_er/run_parallel_eval.py \
    --model llama-8b \
    --load data/prompts/llama-8b/mipro_20260116.json \
    --eval-demos 0 \
    --input data/samples/sample_10000.json \
    --chunks 10 \
    --time 2h

echo ""
echo "Completed: $(date)"
SCRIPT
)
echo "Submitted llama8b_10k_0shot: $JOB_LLAMA_10K_0 (depends on $JOB_LLAMA_0)"

# Job 7: Llama-8B 10k MIPROv2 8-shot (depends on JOB_LLAMA_4)
JOB_LLAMA_10K_8=$(flux batch -N1 -n1 -g8 -t 3h \
    --job-name=llama8b_10k_8shot \
    --dependency=afterany:$JOB_LLAMA_4 \
    -o flux-llama8b-10k-mipro8.out <<'SCRIPT'
#!/bin/bash
echo "=== Llama-8B 10k MIPROv2 8-shot ==="
echo "Started: $(date)"

source ~/.bashrc && module load rocm/6.3.1
export CONDAPATH=/usr/workspace/smith585/x86_miniconda
eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
conda activate base
cd /usr/workspace/smith585/codebases/OSINT_entity_resolution
export PYTHONPATH=$PWD:$PYTHONPATH

python scripts/baselines/dspy_er/run_parallel_eval.py \
    --model llama-8b \
    --load data/prompts/llama-8b/mipro_20260116.json \
    --input data/samples/sample_10000.json \
    --chunks 10 \
    --time 2h

echo ""
echo "Completed: $(date)"
SCRIPT
)
echo "Submitted llama8b_10k_8shot: $JOB_LLAMA_10K_8 (depends on $JOB_LLAMA_4)"

# Job 8: DeepSeek-14b 10k Zero-shot (depends on JOB_DS_ZERO)
JOB_DS_10K_ZERO=$(flux batch -N1 -n1 -g8 -t 3h \
    --job-name=deepseek_10k_zero \
    --dependency=afterany:$JOB_DS_ZERO \
    -o flux-deepseek-10k-zero.out <<'SCRIPT'
#!/bin/bash
echo "=== DeepSeek-14b 10k Zero-shot ==="
echo "Started: $(date)"

source ~/.bashrc && module load rocm/6.3.1
export CONDAPATH=/usr/workspace/smith585/x86_miniconda
eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
conda activate base
cd /usr/workspace/smith585/codebases/OSINT_entity_resolution
export PYTHONPATH=$PWD:$PYTHONPATH

python scripts/baselines/dspy_er/run_parallel_eval.py \
    --model deepseek-14b \
    --input data/samples/sample_10000.json \
    --chunks 12 \
    --time 4h

echo ""
echo "Completed: $(date)"
SCRIPT
)
echo "Submitted deepseek_10k_zero: $JOB_DS_10K_ZERO (depends on $JOB_DS_ZERO)"

# Job 9: DeepSeek-14b 10k MIPROv2 (depends on MIPROv2 optimization completing)
JOB_DS_10K_MIPRO=$(flux batch -N1 -n1 -g8 -t 3h \
    --job-name=deepseek_10k_mipro \
    --dependency=afterok:$JOB_DS_MIPRO \
    -o flux-deepseek-10k-mipro.out <<'SCRIPT'
#!/bin/bash
echo "=== DeepSeek-14b 10k MIPROv2 ==="
echo "Started: $(date)"

source ~/.bashrc && module load rocm/6.3.1
export CONDAPATH=/usr/workspace/smith585/x86_miniconda
eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
conda activate base
cd /usr/workspace/smith585/codebases/OSINT_entity_resolution
export PYTHONPATH=$PWD:$PYTHONPATH

# Only run if optimization succeeded and saved the file
if [ -f data/prompts/deepseek-14b/mipro_20260118.json ]; then
    python scripts/baselines/dspy_er/run_parallel_eval.py \
        --model deepseek-14b \
        --load data/prompts/deepseek-14b/mipro_20260118.json \
        --input data/samples/sample_10000.json \
        --chunks 12 \
        --time 4h
else
    echo 'ERROR: MIPROv2 optimization did not produce expected file'
    echo 'Expected: data/prompts/deepseek-14b/mipro_20260118.json'
    exit 1
fi

echo ""
echo "Completed: $(date)"
SCRIPT
)
echo "Submitted deepseek_10k_mipro: $JOB_DS_10K_MIPRO (depends on $JOB_DS_MIPRO)"

# ============================================================
# Summary
# ============================================================

echo ""
echo "============================================================"
echo "=== Submission Complete ==="
echo "============================================================"
echo ""
echo "Phase 1 jobs (5):"
echo "  $JOB_LLAMA_0  - llama8b_mipro0"
echo "  $JOB_LLAMA_2  - llama8b_mipro2"
echo "  $JOB_LLAMA_4  - llama8b_mipro4"
echo "  $JOB_DS_ZERO  - deepseek14b_zero"
echo "  $JOB_DS_MIPRO - deepseek14b_mipro_opt"
echo ""
echo "Phase 2 jobs (4) - will start after Phase 1 dependencies complete:"
echo "  $JOB_LLAMA_10K_0    - llama8b_10k_0shot (after $JOB_LLAMA_0)"
echo "  $JOB_LLAMA_10K_8    - llama8b_10k_8shot (after $JOB_LLAMA_4)"
echo "  $JOB_DS_10K_ZERO    - deepseek_10k_zero (after $JOB_DS_ZERO)"
echo "  $JOB_DS_10K_MIPRO   - deepseek_10k_mipro (after $JOB_DS_MIPRO)"
echo ""
echo "============================================================"
echo "Monitoring Commands:"
echo "============================================================"
echo "  flux jobs -a | head -20          # Check job status"
echo "  flux job info <jobid>            # Get job details"
echo "  flux job attach <jobid>          # Attach to running job"
echo ""
echo "Results will be in: flux-*.out"
echo ""
echo "Check results after completion:"
echo "  for f in flux-llama8b-mipro*.out flux-deepseek*.out; do"
echo "      echo \"=== \$f ===\""
echo "      grep -A5 'FINAL RESULTS' \$f 2>/dev/null || echo 'Not found'"
echo "  done"
echo ""
