#!/usr/bin/env python3
"""
Nomenklatura Baseline (RegressionV1)

This script uses the OpenSanctions Nomenklatura library to perform entity resolution.
It uses the default 'RegressionV1' algorithm.

Usage:
    python scripts/baselines/nomenklatura_v1.py --input data/samples/sample_1000.json
"""

import argparse
import json
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add vendor directory to path to find nomenklatura
sys.path.append(str(Path(__file__).parent.parent / "vendor/nomenklatura"))

# Add parent directory to path to import evaluation metrics
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.evaluate import evaluate
from scripts.experiment import Experiment

try:
    from nomenklatura.matching import get_algorithm, DefaultAlgorithm
    from nomenklatura.matching.types import ScoringConfig
    from followthemoney import model
    from followthemoney.proxy import EntityProxy
except ImportError as e:
    print(f"Error importing nomenklatura: {e}")
    print("Please ensure dependencies are installed.")
    sys.exit(1)

def load_data_as_proxies(filepath: str) -> List[Dict[str, Any]]:
    """
    Load data and convert to EntityProxy objects.
    Returns list of dicts: {'left': Proxy, 'right': Proxy, 'judgement': str}

    Supports:
    - v1 format: JSON array of pairs
    - v2 format: JSON object with 'metadata' and 'pairs' keys
    - JSONL format: One pair per line
    - Gzipped files
    """
    path = Path(filepath)
    raw_data = []

    # Handle both JSON array and JSONL
    if path.name.endswith('.gz'):
        import gzip
        with gzip.open(path, 'rt') as f:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line))
    else:
        with open(path, 'r') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                raw_data = json.load(f)
            elif first_char == '{':
                # Could be v2 format with metadata wrapper or JSONL
                content = json.load(f)
                if 'pairs' in content:
                    # v2 format
                    raw_data = content['pairs']
                else:
                    # Single object - probably JSONL, re-read
                    f.seek(0)
                    for line in f:
                        if line.strip():
                            raw_data.append(json.loads(line))
            else:
                for line in f:
                    if line.strip():
                        raw_data.append(json.loads(line))

    processed_data = []
    for item in raw_data:
        try:
            left = EntityProxy.from_dict(item['left'])
            right = EntityProxy.from_dict(item['right'])
            processed_data.append({
                'left': left,
                'right': right,
                'judgement': item['judgement']
            })
        except Exception as e:
            # Skip invalid entries
            if len(processed_data) == 0:
                print(f"Error loading pair: {e}")
            continue
            
    return processed_data

def optimize_threshold(scores: List[float], labels: List[str]) -> float:
    """Find the threshold that maximizes F1 score on the development set."""
    best_f1 = 0.0
    best_thresh = 0.5
    
    # Test thresholds from 0.05 to 0.95
    thresholds = [i/100.0 for i in range(5, 100, 5)]
    
    y_true = [1 if l == "positive" else 0 for l in labels]
    
    for thresh in thresholds:
        y_pred = [1 if s >= thresh else 0 for s in scores]
        
        tp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh

def main():
    parser = argparse.ArgumentParser(description="Run Nomenklatura Baseline")
    parser.add_argument("--input", required=True, help="Path to input JSON/JSONL file")
    parser.add_argument("--output", default="data/outputs", help="Output directory")
    parser.add_argument("--matcher", default="regression-v1", help="Algorithm name (default: regression-v1)")
    parser.add_argument("--dev-ratio", type=float, default=0.2,
                       help="Ratio of data to use for dev/tuning (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-shuffle", action="store_true",
                       help="Don't shuffle data (use if input is already shuffled)")
    args = parser.parse_args()

    # Load Data
    print(f"Loading data from {args.input}...")
    data = load_data_as_proxies(args.input)
    print(f"Loaded {len(data)} pairs.")

    # Initialize Matcher
    Algorithm = get_algorithm(args.matcher)
    if not Algorithm:
        # Try to match the internal name string if 'regression-v1' passed
        # The code usually uses class attributes. Let's try to find it.
        if args.matcher == 'regression-v1':
             # It might be named differently in the ALGORITHMS list
             # Based on __init__.py it is RegressionV1.NAME
             from nomenklatura.matching import RegressionV1
             Algorithm = RegressionV1
        else:
            print(f"Unknown matcher: {args.matcher}")
            return

    print(f"Using Algorithm: {Algorithm.NAME}")
    config = ScoringConfig.defaults()

    # Optionally shuffle (skip if input is already properly shuffled)
    random.seed(args.seed)
    if not args.no_shuffle:
        random.shuffle(data)
        print("Data shuffled with seed:", args.seed)
    else:
        print("Using data order as-is (--no-shuffle)")

    # Split into dev (for threshold tuning) and test
    split_idx = int(len(data) * args.dev_ratio)
    dev_set = data[:split_idx]
    test_set = data[split_idx:]

    print(f"Split: {len(dev_set)} dev ({args.dev_ratio:.0%}), {len(test_set)} test ({1-args.dev_ratio:.0%})")
    
    # 1. Tune on Dev Set
    print("\nTuning threshold on Dev Set...")
    dev_scores = []
    dev_labels = []
    
    for item in dev_set:
        result = Algorithm.compare(item['left'], item['right'], config)
        dev_scores.append(result.score)
        dev_labels.append(item['judgement'])
        
    best_threshold = optimize_threshold(dev_scores, dev_labels)
    print(f"Optimal Threshold found: {best_threshold}")
    
    # 2. Evaluate on Test Set
    print("\nEvaluating on Test Set...")
    test_preds = []
    test_ground_truth = []
    
    for item in test_set:
        result = Algorithm.compare(item['left'], item['right'], config)
        pred = "positive" if result.score >= best_threshold else "negative"
        test_preds.append(pred)
        test_ground_truth.append(item['judgement'])
        
    results = evaluate(test_ground_truth, test_preds)

    # Track experiment
    exp = Experiment(method="nomenklatura", input_file=args.input)
    exp.set_params(
        algorithm=Algorithm.NAME,
        threshold=best_threshold,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        shuffled=not args.no_shuffle,
        dev_pairs=len(dev_set),
        test_pairs=len(test_set)
    )
    exp.set_metrics(
        accuracy=results['accuracy'],
        precision=results['precision'],
        recall=results['recall'],
        f1=results['f1'],
        confusion_matrix=results['confusion_matrix']
    )

    result_file = exp.save(args.output)
    exp.print_summary()
    print(f"Results saved to: {result_file}")

if __name__ == "__main__":
    main()
