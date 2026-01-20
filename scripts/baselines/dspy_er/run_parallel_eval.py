#!/usr/bin/env python3
"""
Parallel DSPy Evaluation - Split dataset across multiple nodes.

Splits a large dataset into chunks and evaluates them in parallel across
multiple Flux jobs, then aggregates results.

Usage:
    # Submit 10 parallel jobs for 10k dataset
    python run_parallel_eval.py --model llama-8b --chunks 10 --input data/samples/sample_10000.json

    # With MIPROv2 module
    python run_parallel_eval.py --model llama-8b --chunks 10 \
        --load data/prompts/llama-8b/mipro_20260116.json --eval-demos 0

    # Aggregate results after jobs complete
    python run_parallel_eval.py --aggregate --run-id 20260117_123456
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add parent for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel DSPy Evaluation")

    # Mode
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate results from completed parallel jobs")
    parser.add_argument("--run-id", type=str,
                        help="Run ID for aggregation (format: YYYYMMDD_HHMMSS)")

    # Parallel config
    parser.add_argument("--chunks", type=int, default=10,
                        help="Number of parallel chunks/jobs")
    parser.add_argument("--time", type=str, default="2h",
                        help="Time limit per job")

    # Model config
    parser.add_argument("--model", type=str, default="llama-8b",
                        help="Model name")
    parser.add_argument("--load", type=str, default=None,
                        help="Load compiled MIPROv2 module")
    parser.add_argument("--eval-demos", type=int, default=None,
                        help="Limit demos for ablation")

    # Data
    parser.add_argument("--input", type=str, default="data/samples/sample_10000.json",
                        help="Input dataset")
    parser.add_argument("--output-dir", type=str, default="data/outputs/parallel_eval",
                        help="Output directory for results")

    # Debug
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without submitting")

    return parser.parse_args()


def create_chunk_script(args, chunk_id: int, start_idx: int, end_idx: int, run_id: str) -> str:
    """Generate bash script for a single chunk evaluation."""

    load_arg = f"--load {args.load}" if args.load else ""
    eval_demos_arg = f"--eval-demos {args.eval_demos}" if args.eval_demos is not None else ""

    script = f'''#!/bin/bash
echo "=== Parallel Eval Chunk {chunk_id}/{args.chunks} ==="
echo "Run ID: {run_id}"
echo "Range: {start_idx} to {end_idx}"
echo "Started: $(date)"
echo ""

source ~/.bashrc && module load rocm/6.3.1
export CONDAPATH=/usr/workspace/smith585/x86_miniconda
eval "$($CONDAPATH/bin/conda shell.bash hook)"
conda activate base
cd /usr/workspace/smith585/codebases/OSINT_entity_resolution
export PYTHONPATH=$PWD:$PYTHONPATH

# Run evaluation on chunk
python -c "
import json
import sys
sys.path.append('.')

from scripts.baselines.dspy_er.run_dspy import setup_lm, run_zero_shot
from scripts.baselines.dspy_er.data import load_dspy_examples
from scripts.baselines.dspy_er.modules import get_module, MODULES
from scripts.baselines.dspy_er.optimize import load_compiled_module
from scripts.baselines.dspy_er.metrics import BatchEvaluator, print_metrics
import dspy
from types import SimpleNamespace

# Args
args = SimpleNamespace(
    model='{args.model}',
    input='{args.input}',
    load={repr(args.load) if args.load else None},
    eval_demos={args.eval_demos if args.eval_demos is not None else 'None'},
    module='cot',
    api_base=None,
    temperature=0.0,
    model_path=None,
    max_new_tokens=512,
    use_vllm=False,
)

# Load all examples
dev_examples, all_test = load_dspy_examples(args.input, dev_size=200)

# Get chunk
chunk_test = all_test[{start_idx}:{end_idx}]
print(f'Evaluating chunk: {start_idx} to {end_idx} ({{len(chunk_test)}} examples)')

# Setup LM
setup_lm(args)

# Load or create module
if args.load:
    module_class = MODULES[args.module]
    module = load_compiled_module(module_class, args.load)
    if args.eval_demos is not None:
        for name, predictor in module.named_predictors():
            if hasattr(predictor, 'demos'):
                predictor.demos = predictor.demos[:args.eval_demos]
else:
    module = get_module(args.module)

# Evaluate
evaluator = BatchEvaluator(module)
results = evaluator.evaluate(chunk_test)

# Save chunk results
output = {{
    'chunk_id': {chunk_id},
    'start_idx': {start_idx},
    'end_idx': {end_idx},
    'n_examples': len(chunk_test),
    'metrics': results['metrics'],
    'predictions': [
        {{'classification': p.classification, 'reasoning': getattr(p, 'reasoning', '')}}
        for p in results.get('predictions', [])
    ],
}}

output_path = '{args.output_dir}/{run_id}/chunk_{chunk_id:02d}.json'
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f'Saved results to {{output_path}}')
print_metrics(results['metrics'], f'Chunk {chunk_id} Results')
"

echo ""
echo "Completed: $(date)"
'''
    return script


def submit_parallel_jobs(args):
    """Submit parallel evaluation jobs."""

    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load dataset to get size
    with open(args.input) as f:
        data = json.load(f)

    # Handle both formats: list or dict with 'pairs' key
    if isinstance(data, dict):
        pairs = data.get('pairs', data.get('data', []))
    else:
        pairs = data

    # Test set is after dev set (200 examples)
    total_test = len(pairs) - 200
    chunk_size = total_test // args.chunks

    print(f"=== Parallel Evaluation Setup ===")
    print(f"Run ID: {run_id}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.input} ({len(pairs)} total, {total_test} test)")
    print(f"Chunks: {args.chunks} x ~{chunk_size} examples")
    print(f"Load module: {args.load}")
    print(f"Eval demos: {args.eval_demos}")
    print(f"Time per job: {args.time}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = {
        "run_id": run_id,
        "model": args.model,
        "input": args.input,
        "chunks": args.chunks,
        "load": args.load,
        "eval_demos": args.eval_demos,
        "total_test": total_test,
        "chunk_size": chunk_size,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Submit jobs
    job_ids = []
    for i in range(args.chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < args.chunks - 1 else total_test

        script = create_chunk_script(args, i, start_idx, end_idx, run_id)

        if args.dry_run:
            print(f"[DRY RUN] Would submit chunk {i}: examples {start_idx}-{end_idx}")
            continue

        # Submit via flux batch
        cmd = [
            "flux", "batch",
            "-N1", "-n1", "-g8",
            "-t", args.time,
            f"--job-name=chunk_{i:02d}_{args.model}",
            "-o", str(output_dir / f"chunk_{i:02d}.out"),
        ]

        result = subprocess.run(
            cmd,
            input=script,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            job_id = result.stdout.strip()
            job_ids.append(job_id)
            print(f"Submitted chunk {i}: {job_id} (examples {start_idx}-{end_idx})")
        else:
            print(f"ERROR submitting chunk {i}: {result.stderr}")

    if not args.dry_run:
        # Save job IDs
        with open(output_dir / "job_ids.txt", "w") as f:
            f.write("\n".join(job_ids))

        print()
        print(f"=== {len(job_ids)} jobs submitted ===")
        print(f"Output directory: {output_dir}")
        print(f"Monitor: flux jobs -a | grep chunk")
        print(f"Aggregate: python {__file__} --aggregate --run-id {run_id}")

    return run_id, job_ids


def aggregate_results(args):
    """Aggregate results from completed parallel jobs."""

    output_dir = Path(args.output_dir) / args.run_id

    if not output_dir.exists():
        print(f"ERROR: Run directory not found: {output_dir}")
        return

    # Load config
    with open(output_dir / "config.json") as f:
        config = json.load(f)

    print(f"=== Aggregating Results for {args.run_id} ===")
    print(f"Model: {config['model']}")
    print(f"Chunks: {config['chunks']}")
    print()

    # Collect chunk results
    all_predictions = []
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    chunks_found = 0

    for i in range(config['chunks']):
        chunk_file = output_dir / f"chunk_{i:02d}.json"
        if chunk_file.exists():
            with open(chunk_file) as f:
                chunk = json.load(f)

            cm = chunk['metrics'].get('confusion_matrix', {})
            # Support both short keys (tp) and full keys (true_positive)
            total_tp += cm.get('tp', cm.get('true_positive', 0))
            total_fp += cm.get('fp', cm.get('false_positive', 0))
            total_tn += cm.get('tn', cm.get('true_negative', 0))
            total_fn += cm.get('fn', cm.get('false_negative', 0))

            all_predictions.extend(chunk.get('predictions', []))
            chunks_found += 1
            print(f"  Chunk {i}: {chunk['n_examples']} examples, F1={chunk['metrics']['f1']:.4f}")
        else:
            print(f"  Chunk {i}: MISSING")

    if chunks_found == 0:
        print("ERROR: No chunk results found")
        return

    # Calculate aggregate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn)

    print()
    print(f"=== AGGREGATE RESULTS ({chunks_found}/{config['chunks']} chunks) ===")
    print(f"Total examples: {total_tp + total_fp + total_tn + total_fn}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Confusion: TP={total_tp}, FP={total_fp}, TN={total_tn}, FN={total_fn}")

    # Save aggregate results
    aggregate = {
        "run_id": args.run_id,
        "config": config,
        "chunks_completed": chunks_found,
        "total_examples": total_tp + total_fp + total_tn + total_fn,
        "metrics": {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "confusion_matrix": {
                "tp": total_tp,
                "fp": total_fp,
                "tn": total_tn,
                "fn": total_fn,
            }
        }
    }

    with open(output_dir / "aggregate_results.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\nSaved to: {output_dir / 'aggregate_results.json'}")


def main():
    args = parse_args()

    if args.aggregate:
        if not args.run_id:
            print("ERROR: --run-id required for aggregation")
            return
        aggregate_results(args)
    else:
        submit_parallel_jobs(args)


if __name__ == "__main__":
    main()
