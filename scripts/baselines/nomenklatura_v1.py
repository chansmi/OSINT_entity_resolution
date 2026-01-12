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

# Add parent directory to path to import evaluation metrics
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.evaluate import evaluate

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

def run_single_split(data: List[Dict], Algorithm, config, split_ratio: float, seed: int):
    """Run evaluation with a single dev/test split."""
    random.seed(seed)
    random.shuffle(data)
    
    split_idx = int(len(data) * split_ratio)
    dev_set = data[:split_idx]
    test_set = data[split_idx:]
    
    print(f"Split: {len(dev_set)} dev, {len(test_set)} test.")
    
    # Tune on Dev Set
    print("\nTuning threshold on Dev Set...")
    dev_scores = []
    dev_labels = []
    
    for item in dev_set:
        result = Algorithm.compare(item['left'], item['right'], config)
        dev_scores.append(result.score)
        dev_labels.append(item['judgement'])
        
    best_threshold = optimize_threshold(dev_scores, dev_labels)
    print(f"Optimal Threshold found: {best_threshold}")
    
    # Evaluate on Test Set
    print("\nEvaluating on Test Set...")
    test_preds = []
    test_ground_truth = []
    
    for item in test_set:
        result = Algorithm.compare(item['left'], item['right'], config)
        pred = "positive" if result.score >= best_threshold else "negative"
        test_preds.append(pred)
        test_ground_truth.append(item['judgement'])
        
    results = evaluate(test_ground_truth, test_preds)
    results['threshold'] = best_threshold
    return results


def run_cross_validation(data: List[Dict], Algorithm, config, n_folds: int, seed: int):
    """Run k-fold cross-validation for more robust evaluation."""
    random.seed(seed)
    random.shuffle(data)
    
    fold_size = len(data) // n_folds
    all_results = []
    all_thresholds = []
    
    print(f"\nRunning {n_folds}-fold cross-validation...")
    print(f"Fold size: ~{fold_size} samples\n")
    
    for fold in range(n_folds):
        # Create fold splits
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < n_folds - 1 else len(data)
        
        test_set = data[start_idx:end_idx]
        dev_set = data[:start_idx] + data[end_idx:]
        
        # Tune threshold on dev set (everything except this fold)
        dev_scores = []
        dev_labels = []
        for item in dev_set:
            result = Algorithm.compare(item['left'], item['right'], config)
            dev_scores.append(result.score)
            dev_labels.append(item['judgement'])
        
        threshold = optimize_threshold(dev_scores, dev_labels)
        all_thresholds.append(threshold)
        
        # Evaluate on this fold
        test_preds = []
        test_ground_truth = []
        for item in test_set:
            result = Algorithm.compare(item['left'], item['right'], config)
            pred = "positive" if result.score >= threshold else "negative"
            test_preds.append(pred)
            test_ground_truth.append(item['judgement'])
        
        fold_results = evaluate(test_ground_truth, test_preds)
        all_results.append(fold_results)
        
        print(f"Fold {fold+1}/{n_folds}: Threshold={threshold:.2f}, "
              f"F1={fold_results['f1']:.4f}, Acc={fold_results['accuracy']:.4f}")
    
    # Aggregate results
    avg_results = {
        'accuracy': sum(r['accuracy'] for r in all_results) / n_folds,
        'precision': sum(r['precision'] for r in all_results) / n_folds,
        'recall': sum(r['recall'] for r in all_results) / n_folds,
        'f1': sum(r['f1'] for r in all_results) / n_folds,
        'threshold_mean': sum(all_thresholds) / n_folds,
        'threshold_std': (sum((t - sum(all_thresholds)/n_folds)**2 for t in all_thresholds) / n_folds) ** 0.5,
        'f1_std': (sum((r['f1'] - sum(r2['f1'] for r2 in all_results)/n_folds)**2 for r in all_results) / n_folds) ** 0.5,
    }
    
    return avg_results, all_results


def main():
    parser = argparse.ArgumentParser(description="Run Nomenklatura Baseline")
    parser.add_argument("--input", required=True, help="Path to input JSON/JSONL file")
    parser.add_argument("--matcher", default="regression-v1", help="Algorithm name (default: regression-v1)")
    parser.add_argument("--split-ratio", type=float, default=0.5, help="Ratio of data to use for dev/tuning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cv", type=int, default=0, 
                        help="Number of cross-validation folds (0 = single split, default)")
    args = parser.parse_args()

    # Load Data
    print(f"Loading data from {args.input}...")
    data = load_data_as_proxies(args.input)
    print(f"Loaded {len(data)} pairs.")
    
    # Initialize Matcher
    Algorithm = get_algorithm(args.matcher)
    if not Algorithm:
        if args.matcher == 'regression-v1':
             from nomenklatura.matching import RegressionV1
             Algorithm = RegressionV1
        else:
            print(f"Unknown matcher: {args.matcher}")
            return

    print(f"Using Algorithm: {Algorithm.NAME}")
    config = ScoringConfig.defaults()

    if args.cv > 0:
        # Cross-validation mode
        avg_results, _ = run_cross_validation(data, Algorithm, config, args.cv, args.seed)
        
        print("\n" + "="*50)
        print(f"NOMENKLATURA RESULTS ({args.cv}-Fold Cross-Validation)")
        print("="*50)
        print(f"Accuracy:  {avg_results['accuracy']:.4f}")
        print(f"Precision: {avg_results['precision']:.4f}")
        print(f"Recall:    {avg_results['recall']:.4f}")
        print(f"F1 Score:  {avg_results['f1']:.4f} (±{avg_results['f1_std']:.4f})")
        print(f"Threshold: {avg_results['threshold_mean']:.2f} (±{avg_results['threshold_std']:.2f})")
        print("="*50)
    else:
        # Single split mode (original behavior)
        results = run_single_split(data, Algorithm, config, args.split_ratio, args.seed)
        
        print("\n" + "="*50)
        print(f"NOMENKLATURA RESULTS (Threshold: {results['threshold']})")
        print("="*50)
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1 Score:  {results['f1']:.4f}")
        print("="*50)

if __name__ == "__main__":
    main()
