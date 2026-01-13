#!/usr/bin/env python3
"""
Reasoning Effort Sweep - Test impact of reasoning tokens on accuracy.

Sweeps through reasoning effort levels (none, low, medium, high, xhigh)
to measure the accuracy/cost tradeoff.

Usage:
    # Dry run (preview what will be run)
    python scripts/baselines/reasoning_sweep.py --dry-run

    # Full sweep
    python scripts/baselines/reasoning_sweep.py

    # Custom model and limit
    python scripts/baselines/reasoning_sweep.py --model gpt-5-nano --limit 100

    # Run from notebook
    from scripts.baselines.reasoning_sweep import run_reasoning_sweep
    results = run_reasoning_sweep(model="gpt-5-nano", dry_run=False)
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.baselines.config import ExperimentConfig, CONFIGS


REASONING_LEVELS = ["none", "low", "medium", "high", "xhigh"]


def run_single_experiment(
    reasoning: str,
    model: str = "gpt-5-nano",
    input_file: str = "data/samples/sample_1000.json",
    offset: int = 200,
    limit: Optional[int] = None,
    parallel: int = 20,
    dry_run: bool = False
) -> dict:
    """Run a single experiment with given reasoning level.

    Args:
        reasoning: Reasoning effort level
        model: Model to use
        input_file: Path to input data
        offset: Number of pairs to skip
        limit: Limit number of pairs (None = all)
        parallel: Number of parallel requests
        dry_run: If True, just print command without running

    Returns:
        Dict with experiment results or command info
    """
    cmd = [
        "python", "scripts/baselines/llm_zeroshot.py",
        "--input", input_file,
        "--offset", str(offset),
        "--model", model,
        "--reasoning", reasoning,
        "--parallel", str(parallel),
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    if dry_run:
        return {
            "reasoning": reasoning,
            "command": " ".join(cmd),
            "status": "dry_run"
        }

    print(f"\n{'='*60}")
    print(f"Running with reasoning={reasoning}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)

    return {
        "reasoning": reasoning,
        "command": " ".join(cmd),
        "return_code": result.returncode,
        "status": "completed" if result.returncode == 0 else "failed"
    }


def run_reasoning_sweep(
    model: str = "gpt-5-nano",
    input_file: str = "data/samples/sample_1000.json",
    offset: int = 200,
    limit: Optional[int] = None,
    parallel: int = 20,
    dry_run: bool = False,
    levels: List[str] = None
) -> List[dict]:
    """Run sweep across all reasoning levels.

    Args:
        model: Model to use for all experiments
        input_file: Path to input data
        offset: Number of pairs to skip
        limit: Limit number of pairs (None = all)
        parallel: Number of parallel requests
        dry_run: If True, just print commands without running
        levels: List of reasoning levels to test (default: all)

    Returns:
        List of result dicts, one per reasoning level
    """
    if levels is None:
        levels = REASONING_LEVELS

    results = []

    print(f"\nReasoning Effort Sweep")
    print(f"Model: {model}")
    print(f"Input: {input_file}")
    print(f"Offset: {offset}, Limit: {limit or 'all'}")
    print(f"Levels: {', '.join(levels)}")
    print(f"Dry run: {dry_run}")

    if dry_run:
        print("\n" + "="*60)
        print("DRY RUN - Commands that would be executed:")
        print("="*60)

    for level in levels:
        result = run_single_experiment(
            reasoning=level,
            model=model,
            input_file=input_file,
            offset=offset,
            limit=limit,
            parallel=parallel,
            dry_run=dry_run
        )
        results.append(result)

        if dry_run:
            print(f"\n[{level}]")
            print(f"  {result['command']}")

    if dry_run:
        print("\n" + "="*60)
        print("To run the sweep, remove --dry-run flag")
        print("="*60)

        # Estimate costs
        print("\nCost Estimates:")
        for level in levels:
            config = ExperimentConfig(
                model=model,
                reasoning_effort=level
            )
            n_pairs = (1000 - offset) if limit is None else min(limit, 1000 - offset)
            estimate = config.estimate_cost(n_pairs)
            print(f"  {level}: ${estimate['estimated_cost_usd']} ({estimate['total_tokens']:,} tokens)")

    else:
        print("\n" + "="*60)
        print("Sweep Complete!")
        print("="*60)
        print("\nResults:")
        for r in results:
            status = "OK" if r["status"] == "completed" else "FAILED"
            print(f"  {r['reasoning']}: {status}")

        print("\nCompare results with:")
        print("  python scripts/experiment.py --compare --method llm_zeroshot")

    return results


def load_experiments_for_sweep(output_dir: str = "data/outputs") -> List[dict]:
    """Load recent experiment results for analysis.

    Returns list of experiments that look like reasoning sweep results.
    """
    experiments_file = Path(output_dir) / "experiments.jsonl"
    if not experiments_file.exists():
        return []

    experiments = []
    with open(experiments_file) as f:
        for line in f:
            exp = json.loads(line)
            if exp.get("method") == "llm_zeroshot":
                experiments.append(exp)

    return experiments


def analyze_sweep_results(output_dir: str = "data/outputs") -> None:
    """Analyze and print summary of reasoning sweep results."""
    experiments = load_experiments_for_sweep(output_dir)

    if not experiments:
        print("No experiments found. Run the sweep first.")
        return

    # Group by reasoning level
    by_reasoning = {}
    for exp in experiments:
        reasoning = exp.get("params", {}).get("reasoning_effort", "unknown")
        if reasoning not in by_reasoning:
            by_reasoning[reasoning] = []
        by_reasoning[reasoning].append(exp)

    print("\nReasoning Sweep Analysis")
    print("="*60)
    print(f"{'Reasoning':<12} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Tokens':>10} {'Cost':>8}")
    print("-"*60)

    for level in REASONING_LEVELS:
        if level in by_reasoning:
            # Use most recent experiment for this level
            exp = sorted(by_reasoning[level], key=lambda x: x.get("timestamp", ""), reverse=True)[0]
            metrics = exp.get("metrics", {})
            params = exp.get("params", {})

            f1 = metrics.get("f1", 0) * 100
            precision = metrics.get("precision", 0) * 100
            recall = metrics.get("recall", 0) * 100
            tokens = params.get("total_tokens", 0)
            cost = tokens * 0.0000005  # Rough estimate

            print(f"{level:<12} {f1:>7.1f}% {precision:>9.1f}% {recall:>7.1f}% {tokens:>10,} ${cost:>7.4f}")
        else:
            print(f"{level:<12} {'N/A':>8} {'N/A':>10} {'N/A':>8} {'N/A':>10} {'N/A':>8}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep reasoning effort levels to measure accuracy/cost tradeoff"
    )
    parser.add_argument("--model", default="gpt-5-nano",
                        help="Model to use (default: gpt-5-nano)")
    parser.add_argument("--input", default="data/samples/sample_1000.json",
                        help="Input sample file")
    parser.add_argument("--offset", type=int, default=200,
                        help="Skip first N pairs (default: 200)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of pairs")
    parser.add_argument("--parallel", type=int, default=20,
                        help="Number of parallel requests")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--levels", nargs="+", default=None,
                        choices=REASONING_LEVELS,
                        help="Specific reasoning levels to test")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing sweep results")

    args = parser.parse_args()

    if args.analyze:
        analyze_sweep_results()
        return

    results = run_reasoning_sweep(
        model=args.model,
        input_file=args.input,
        offset=args.offset,
        limit=args.limit,
        parallel=args.parallel,
        dry_run=args.dry_run,
        levels=args.levels
    )


if __name__ == "__main__":
    main()
