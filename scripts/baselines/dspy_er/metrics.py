#!/usr/bin/env python3
"""
DSPy-compatible metrics for entity resolution.

Provides metric functions that work with DSPy optimizers and evaluation.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import dspy

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.evaluate import evaluate as sklearn_evaluate


def exact_match(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """Simple exact match metric for DSPy.

    Returns True if predicted classification matches ground truth.

    Args:
        example: DSPy Example with ground truth classification
        prediction: DSPy Prediction with predicted classification
        trace: Optional trace for debugging (not used)

    Returns:
        bool: True if classification matches
    """
    expected = getattr(example, "classification", None)
    predicted = getattr(prediction, "classification", None)

    if expected is None or predicted is None:
        return False

    return expected.lower().strip() == predicted.lower().strip()


def f1_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """F1-weighted metric for DSPy optimization.

    Returns 1.0 for correct predictions, with higher weight for
    correctly identifying negatives (minority class).

    Args:
        example: DSPy Example with ground truth classification
        prediction: DSPy Prediction with predicted classification
        trace: Optional trace for debugging

    Returns:
        float: 1.0 for correct, 0.0 for incorrect (or 1.5 for correct negative)
    """
    expected = getattr(example, "classification", None)
    predicted = getattr(prediction, "classification", None)

    if expected is None or predicted is None:
        return 0.0

    expected = expected.lower().strip()
    predicted = predicted.lower().strip()

    if expected == predicted:
        # Weight negative class higher (it's the minority class at ~23%)
        return 1.5 if expected == "negative" else 1.0
    return 0.0


def classification_accuracy(
    examples: List[dspy.Example],
    predictions: List[dspy.Prediction],
) -> Dict[str, Any]:
    """Calculate classification metrics for a batch of predictions.

    Wraps the existing sklearn-based evaluate function.

    Args:
        examples: List of DSPy Examples with ground truth
        predictions: List of DSPy Predictions

    Returns:
        Dict with accuracy, precision, recall, f1, confusion_matrix
    """
    ground_truth = []
    preds = []

    for ex, pred in zip(examples, predictions):
        gt = getattr(ex, "classification", None)
        pr = getattr(pred, "classification", None)

        if gt is not None and pr is not None:
            ground_truth.append(gt.lower().strip())
            preds.append(pr.lower().strip())

    if not ground_truth:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": {},
            "n_evaluated": 0,
        }

    metrics = sklearn_evaluate(ground_truth, preds)
    metrics["n_evaluated"] = len(ground_truth)
    return metrics


class BatchEvaluator:
    """Evaluator for running DSPy modules on batches and computing metrics.

    Usage:
        evaluator = BatchEvaluator(module)
        metrics = evaluator.evaluate(test_examples)
    """

    def __init__(self, module: dspy.Module):
        """Initialize with a DSPy module.

        Args:
            module: DSPy module with forward() method
        """
        self.module = module

    def evaluate(
        self,
        examples: List[dspy.Example],
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate module on examples and compute metrics.

        Args:
            examples: List of DSPy Examples to evaluate
            show_progress: Show tqdm progress bar

        Returns:
            Dict with metrics and predictions
        """
        from tqdm import tqdm

        predictions = []
        errors = []

        iterator = tqdm(examples, desc="Evaluating") if show_progress else examples

        for ex in iterator:
            try:
                pred = self.module(entity_a=ex.entity_a, entity_b=ex.entity_b)
                predictions.append(pred)
            except Exception as e:
                errors.append({"index": len(predictions), "error": str(e)})
                # Create a dummy prediction for error cases
                predictions.append(dspy.Prediction(classification="error", reasoning=str(e)))

        # Calculate metrics
        metrics = classification_accuracy(examples, predictions)
        metrics["errors"] = errors
        metrics["n_errors"] = len(errors)

        return {
            "metrics": metrics,
            "predictions": predictions,
        }


def print_metrics(metrics: Dict[str, Any], title: str = "Evaluation Results") -> None:
    """Print formatted metrics summary.

    Args:
        metrics: Dict from classification_accuracy or BatchEvaluator
        title: Title for the output
    """
    m = metrics.get("metrics", metrics)

    print(f"\n{'=' * 50}")
    print(title)
    print('=' * 50)

    print(f"  Accuracy:  {m.get('accuracy', 0):.4f}")
    print(f"  Precision: {m.get('precision', 0):.4f}")
    print(f"  Recall:    {m.get('recall', 0):.4f}")
    print(f"  F1 Score:  {m.get('f1', 0):.4f}")

    if "n_evaluated" in m:
        print(f"  Evaluated: {m['n_evaluated']}")
    if "n_errors" in m:
        print(f"  Errors:    {m['n_errors']}")

    cm = m.get("confusion_matrix", {})
    if cm:
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {cm.get('true_negative', 0)}")
        print(f"  False Positives: {cm.get('false_positive', 0)}")
        print(f"  False Negatives: {cm.get('false_negative', 0)}")
        print(f"  True Positives:  {cm.get('true_positive', 0)}")

    print('=' * 50)
