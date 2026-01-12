#!/usr/bin/env python3
"""
Evaluation utilities for entity resolution benchmark.

Provides simple functions to compute standard classification metrics.
"""

from typing import List, Dict, Any, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import numpy as np


def evaluate(
    ground_truth: List[str],
    predictions: List[str],
    positive_label: str = "positive",
    negative_label: str = "negative"
) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth labels.
    
    Args:
        ground_truth: List of true labels
        predictions: List of predicted labels
        positive_label: Label for positive class (default: "positive")
        negative_label: Label for negative class (default: "negative")
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, F1, and confusion matrix
        
    Example:
        >>> results = evaluate(
        ...     ground_truth=["positive", "negative", "positive"],
        ...     predictions=["positive", "positive", "positive"]
        ... )
        >>> print(results['accuracy'])
        0.6666666666666666
    """
    if len(ground_truth) != len(predictions):
        raise ValueError(
            f"Length mismatch: ground_truth={len(ground_truth)}, "
            f"predictions={len(predictions)}"
        )
    
    # Convert to binary labels for metrics calculation
    y_true = [1 if label == positive_label else 0 for label in ground_truth]
    y_pred = [1 if label == positive_label else 0 for label in predictions]
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Confusion matrix: [[TN, FP], [FN, TP]]
    # Use labels=[0, 1] to ensure 2x2 matrix even if one class is missing
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    results['confusion_matrix'] = {
        'true_negative': int(cm[0, 0]),
        'false_positive': int(cm[0, 1]),
        'false_negative': int(cm[1, 0]),
        'true_positive': int(cm[1, 1]),
    }
    
    # Per-class metrics
    results['per_class'] = {
        positive_label: {
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
        },
        negative_label: {
            'precision': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
            'recall': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            'f1': f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        }
    }
    
    return results


def print_evaluation_report(results: Dict[str, Any]):
    """
    Print a formatted evaluation report.
    
    Args:
        results: Results dictionary from evaluate()
    """
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"  True Negatives:  {cm['true_negative']:>6}")
    print(f"  False Positives: {cm['false_positive']:>6}")
    print(f"  False Negatives: {cm['false_negative']:>6}")
    print(f"  True Positives:  {cm['true_positive']:>6}")
    
    print(f"\nPer-Class Metrics:")
    for label, metrics in results['per_class'].items():
        print(f"  {label.capitalize()}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1 Score:  {metrics['f1']:.4f}")
    
    print("="*50 + "\n")


def evaluate_from_files(
    ground_truth_file: str,
    predictions_file: str,
    format: str = "jsonl"
) -> Dict[str, Any]:
    """
    Evaluate predictions from files.
    
    Args:
        ground_truth_file: Path to ground truth file
        predictions_file: Path to predictions file
        format: File format ('jsonl' or 'txt')
        
    Returns:
        dict: Evaluation results
    """
    import json
    from pathlib import Path
    
    gt_path = Path(ground_truth_file)
    pred_path = Path(predictions_file)
    
    if format == "jsonl":
        # Load JSONL files
        with open(gt_path) as f:
            ground_truth = [json.loads(line)['judgement'] for line in f]
        with open(pred_path) as f:
            predictions = [json.loads(line)['judgement'] for line in f]
    elif format == "txt":
        # Load text files (one label per line)
        with open(gt_path) as f:
            ground_truth = [line.strip() for line in f]
        with open(pred_path) as f:
            predictions = [line.strip() for line in f]
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return evaluate(ground_truth, predictions)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate entity resolution predictions')
    parser.add_argument('--ground-truth', required=True, help='Ground truth file')
    parser.add_argument('--predictions', required=True, help='Predictions file')
    parser.add_argument('--format', default='jsonl', choices=['jsonl', 'txt'],
                       help='File format')
    
    args = parser.parse_args()
    
    results = evaluate_from_files(
        args.ground_truth,
        args.predictions,
        format=args.format
    )
    
    print_evaluation_report(results)
