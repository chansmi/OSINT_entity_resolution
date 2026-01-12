#!/usr/bin/env python3
"""
Create properly stratified, shuffled samples for entity resolution experiments.

This script creates samples that:
1. Use stratified sampling to match full dataset class ratios
2. Shuffle with a fixed seed for reproducibility
3. Include metadata for traceability

Usage:
    python scripts/create_proper_sample.py --n 1000
    python scripts/create_proper_sample.py --n 10000 --seed 42
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from scripts.load_data import load_full_dataset


MASTER_SEED = 42

# Full dataset ratios (from actual data)
FULL_DATASET_POSITIVE_RATIO = 0.769  # 76.9% positive


def create_stratified_sample(
    data: list,
    n: int,
    seed: int = MASTER_SEED,
    target_positive_ratio: float = FULL_DATASET_POSITIVE_RATIO
) -> dict:
    """
    Create a stratified sample maintaining class ratios.

    Args:
        data: Full dataset
        n: Number of samples to create
        seed: Random seed for reproducibility
        target_positive_ratio: Target ratio of positive class

    Returns:
        dict with 'metadata' and 'pairs' keys
    """
    random.seed(seed)

    # Separate by class
    positives = [d for d in data if d['judgement'] == 'positive']
    negatives = [d for d in data if d['judgement'] == 'negative']

    print(f"Full dataset: {len(positives):,} positive, {len(negatives):,} negative")

    # Calculate stratified sample sizes
    n_pos = int(n * target_positive_ratio)
    n_neg = n - n_pos

    # Verify we have enough samples
    if n_pos > len(positives):
        raise ValueError(f"Requested {n_pos} positives but only {len(positives)} available")
    if n_neg > len(negatives):
        raise ValueError(f"Requested {n_neg} negatives but only {len(negatives)} available")

    print(f"Sampling: {n_pos} positive ({100*n_pos/n:.1f}%), {n_neg} negative ({100*n_neg/n:.1f}%)")

    # Random sample from each class
    sampled_pos = random.sample(positives, n_pos)
    sampled_neg = random.sample(negatives, n_neg)

    # Combine and shuffle
    sample = sampled_pos + sampled_neg
    random.shuffle(sample)

    # Create output with metadata
    output = {
        'metadata': {
            'seed': seed,
            'n_total': n,
            'n_positive': n_pos,
            'n_negative': n_neg,
            'positive_ratio': n_pos / n,
            'target_ratio': target_positive_ratio,
            'source': 'pairs-20251209.json.gz',
            'created': datetime.now().isoformat(),
            'stratified': True,
            'shuffled': True
        },
        'pairs': sample
    }

    return output


def count_runs(items: list) -> int:
    """Count the number of runs (consecutive same values) in a list."""
    if not items:
        return 0
    runs = 1
    for i in range(1, len(items)):
        if items[i] != items[i-1]:
            runs += 1
    return runs


def validate_sample(sample_data: dict) -> dict:
    """
    Validate that sample is properly shuffled and balanced.

    Returns dict with validation results.
    """
    pairs = sample_data['pairs']
    metadata = sample_data['metadata']
    n_pairs = len(pairs)

    # Check 1: Class ratio matches target
    actual_pos = sum(1 for p in pairs if p['judgement'] == 'positive')
    actual_ratio = actual_pos / n_pairs
    ratio_diff = abs(actual_ratio - metadata['target_ratio'])

    # Check 2: Data is shuffled (count runs)
    judgements = [p['judgement'] for p in pairs]
    runs = count_runs(judgements)
    # For random data, expected runs = 2 * n_pos * n_neg / n + 1
    expected_runs = 2 * metadata['n_positive'] * metadata['n_negative'] / n_pairs + 1

    # Check 3: Both classes appear in every 100-pair window
    window_size = 100
    total_windows = n_pairs // window_size
    windows_with_both = sum(
        1 for i in range(0, n_pairs - window_size + 1, window_size)
        if (any(p['judgement'] == 'positive' for p in pairs[i:i+window_size])
            and any(p['judgement'] == 'negative' for p in pairs[i:i+window_size]))
    )

    checks = {
        'class_ratio': {
            'actual': actual_ratio,
            'target': metadata['target_ratio'],
            'diff': ratio_diff,
            'pass': ratio_diff < 0.02
        },
        'shuffled': {
            'runs': runs,
            'expected': expected_runs,
            'pass': runs > expected_runs * 0.7
        },
        'distribution': {
            'windows_with_both_classes': windows_with_both,
            'total_windows': total_windows,
            'pass': windows_with_both == total_windows
        }
    }

    return {
        'valid': all(check['pass'] for check in checks.values()),
        'checks': checks
    }


def print_validation_results(validation: dict) -> None:
    """Print validation results in a consistent format."""
    status = 'PASS' if validation['valid'] else 'FAIL'
    print(f"  Overall: {status}")
    for name, check in validation['checks'].items():
        check_status = 'PASS' if check['pass'] else 'FAIL'
        print(f"  {name}: {check_status}")


def main():
    parser = argparse.ArgumentParser(description='Create stratified sample')
    parser.add_argument('--n', type=int, default=1000, help='Number of samples')
    parser.add_argument('--seed', type=int, default=MASTER_SEED, help='Random seed')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    parser.add_argument('--validate-only', type=str, default=None,
                       help='Path to existing sample to validate')

    args = parser.parse_args()

    # Validate existing sample
    if args.validate_only:
        print(f"Validating existing sample: {args.validate_only}")
        with open(args.validate_only) as f:
            sample_data = json.load(f)

        # Handle both formats (with/without metadata wrapper)
        if 'pairs' not in sample_data:
            sample_data = {'pairs': sample_data, 'metadata': {'target_ratio': 0.769}}

        validation = validate_sample(sample_data)
        print("\nValidation Results:")
        print_validation_results(validation)
        return

    # Create new sample
    print("Loading full dataset...")
    data = load_full_dataset()

    print(f"\nCreating stratified sample (n={args.n}, seed={args.seed})...")
    sample_data = create_stratified_sample(data, args.n, args.seed)

    # Validate
    print("\nValidating sample...")
    validation = validate_sample(sample_data)
    print_validation_results(validation)

    if not validation['valid']:
        print("\nWARNING: Sample failed validation!")

    # Save
    output_path = args.output or f"data/samples/sample_{args.n}.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_path}")
    print(f"  Total pairs: {len(sample_data['pairs'])}")
    print(f"  Positive: {sample_data['metadata']['n_positive']} ({100*sample_data['metadata']['positive_ratio']:.1f}%)")
    print(f"  Negative: {sample_data['metadata']['n_negative']} ({100*(1-sample_data['metadata']['positive_ratio']):.1f}%)")


if __name__ == "__main__":
    main()
