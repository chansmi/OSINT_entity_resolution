#!/usr/bin/env python3
"""
Simple Fuzzy Matching Baseline

This script implements a deterministic rule-based baseline for entity resolution.
It uses a weighted scoring system based on:
1. Name similarity (Token Sort Ratio equivalent)
2. Birth date matching
3. Country/Nationality matching

Usage:
    python scripts/baselines/simple_fuzzy.py --input data/samples/sample_1000.json
"""

import argparse
import json
import difflib
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
import sys

# Add parent directory to path to import evaluation metrics
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.evaluate import evaluate

def normalize_text(text: str) -> str:
    """Lowercase and simple normalization."""
    if not text:
        return ""
    return text.lower().strip()

def get_best_name_similarity(names1: List[str], names2: List[str]) -> float:
    """
    Find the best similarity score between any pair of names.
    Uses SequenceMatcher (0-1 range).
    """
    if not names1 or not names2:
        return 0.0
    
    max_score = 0.0
    for n1 in names1:
        for n2 in names2:
            # Simple token sort ratio approximation
            # Split into tokens, sort, and join
            t1 = " ".join(sorted(normalize_text(n1).split()))
            t2 = " ".join(sorted(normalize_text(n2).split()))
            
            score = difflib.SequenceMatcher(None, t1, t2).ratio()
            if score > max_score:
                max_score = score
    return max_score

def check_intersection(list1: List[str], list2: List[str]) -> bool:
    """Check if two lists have any common elements."""
    if not list1 or not list2:
        return False
    s1 = set(normalize_text(x) for x in list1)
    s2 = set(normalize_text(x) for x in list2)
    return len(s1.intersection(s2)) > 0

def extract_years(dates: List[str]) -> List[str]:
    """Extract years from date strings (YYYY-MM-DD or YYYY)."""
    years = []
    for d in dates:
        if not d:
            continue
        # simple heuristic: take first 4 chars if they look like digits
        clean = d.strip()
        if len(clean) >= 4 and clean[:4].isdigit():
            years.append(clean[:4])
    return years

def compute_similarity_score(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    """
    Compute a weighted similarity score (0.0 to 1.0).
    """
    props_l = left.get("properties", {})
    props_r = right.get("properties", {})

    # 1. Name Similarity (Primary Signal)
    names_l = props_l.get("name", [])
    names_r = props_r.get("name", [])
    name_score = get_best_name_similarity(names_l, names_r)

    # 2. Country/Nationality Match
    countries_l = props_l.get("country", []) + props_l.get("nationality", [])
    countries_r = props_r.get("country", []) + props_r.get("nationality", [])
    country_match = check_intersection(countries_l, countries_r)

    # 3. Birth Date/Year Match
    dob_l = props_l.get("birthDate", [])
    dob_r = props_r.get("birthDate", [])
    # Check exact date match first
    date_match = check_intersection(dob_l, dob_r)
    # If no exact match, check year
    if not date_match:
        years_l = extract_years(dob_l)
        years_r = extract_years(dob_r)
        year_match = check_intersection(years_l, years_r)
    else:
        year_match = True # Implied

    # Weighted Scoring Formula
    # Base score comes from names. Boosts for other matches.
    
    score = name_score * 0.7  # Max 0.7 from names
    
    if country_match:
        score += 0.1
        
    if date_match:
        score += 0.2
    elif year_match:
        score += 0.1
        
    return min(1.0, score)

def optimize_threshold(scores: List[float], labels: List[str]) -> float:
    """Find the threshold that maximizes F1 score on the development set."""
    best_f1 = 0.0
    best_thresh = 0.5
    
    # Test thresholds from 0.1 to 0.95
    thresholds = [i/100.0 for i in range(10, 100, 5)]
    
    y_true = [1 if l == "positive" else 0 for l in labels]
    
    for thresh in thresholds:
        y_pred = [1 if s >= thresh else 0 for s in scores]
        # Calculate F1 manually or use sklearn if available
        # (Using simple implementation here to avoid circular imports if evaluate is complex)
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

def run_single_split(data: List[Dict], split_ratio: float, seed: int):
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
    for pair in dev_set:
        s = compute_similarity_score(pair['left'], pair['right'])
        dev_scores.append(s)
        dev_labels.append(pair['judgement'])
        
    best_threshold = optimize_threshold(dev_scores, dev_labels)
    print(f"Optimal Threshold found: {best_threshold}")
    
    # Evaluate on Test Set
    print("\nEvaluating on Test Set...")
    test_preds = []
    test_ground_truth = []
    
    for pair in test_set:
        s = compute_similarity_score(pair['left'], pair['right'])
        pred = "positive" if s >= best_threshold else "negative"
        test_preds.append(pred)
        test_ground_truth.append(pair['judgement'])
        
    results = evaluate(test_ground_truth, test_preds)
    results['threshold'] = best_threshold
    return results


def run_cross_validation(data: List[Dict], n_folds: int, seed: int):
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
        for pair in dev_set:
            s = compute_similarity_score(pair['left'], pair['right'])
            dev_scores.append(s)
            dev_labels.append(pair['judgement'])
        
        threshold = optimize_threshold(dev_scores, dev_labels)
        all_thresholds.append(threshold)
        
        # Evaluate on this fold
        test_preds = []
        test_ground_truth = []
        for pair in test_set:
            s = compute_similarity_score(pair['left'], pair['right'])
            pred = "positive" if s >= threshold else "negative"
            test_preds.append(pred)
            test_ground_truth.append(pair['judgement'])
        
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
    parser = argparse.ArgumentParser(description="Run Fuzzy Matching Baseline")
    parser.add_argument("--input", required=True, help="Path to input JSON/JSONL file")
    parser.add_argument("--split-ratio", type=float, default=0.5, help="Ratio of data to use for dev/tuning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cv", type=int, default=0, 
                        help="Number of cross-validation folds (0 = single split, default)")
    args = parser.parse_args()

    # Load Data
    data = []
    path = Path(args.input)
    if path.name.endswith('.gz'):
        print("Please uncompress the file or use a .json/.jsonl sample file.")
        return
    
    with open(path, 'r') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
        else:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    print(f"Loaded {len(data)} pairs.")
    
    if args.cv > 0:
        # Cross-validation mode
        avg_results, _ = run_cross_validation(data, args.cv, args.seed)
        
        print("\n" + "="*50)
        print(f"BASELINE RESULTS ({args.cv}-Fold Cross-Validation)")
        print("="*50)
        print(f"Accuracy:  {avg_results['accuracy']:.4f}")
        print(f"Precision: {avg_results['precision']:.4f}")
        print(f"Recall:    {avg_results['recall']:.4f}")
        print(f"F1 Score:  {avg_results['f1']:.4f} (±{avg_results['f1_std']:.4f})")
        print(f"Threshold: {avg_results['threshold_mean']:.2f} (±{avg_results['threshold_std']:.2f})")
        print("="*50)
    else:
        # Single split mode (original behavior)
        results = run_single_split(data, args.split_ratio, args.seed)
        
        print("\n" + "="*50)
        print(f"BASELINE RESULTS (Threshold: {results['threshold']})")
        print("="*50)
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1 Score:  {results['f1']:.4f}")
        print("="*50)

if __name__ == "__main__":
    main()
