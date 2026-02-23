#!/usr/bin/env python3
"""
Data loading utilities for DSPy entity resolution.

Adapts the existing data loading functions to produce DSPy Example objects.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import dspy

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.load_data import load_sample
from scripts.baselines.llm_zeroshot import format_entity


# --- DSPy Example Conversion ---

def pair_to_example(pair: dict, include_label: bool = True) -> dspy.Example:
    """Convert an entity pair dict to a DSPy Example.

    Args:
        pair: Entity pair dict with 'left', 'right', and optionally 'judgement'
        include_label: If True, include classification label for training

    Returns:
        dspy.Example with entity_a, entity_b, and optionally classification
    """
    example_dict = {
        "entity_a": format_entity(pair["left"]),
        "entity_b": format_entity(pair["right"]),
    }

    if include_label and "judgement" in pair:
        example_dict["classification"] = pair["judgement"]

    example = dspy.Example(**example_dict)

    # Mark input fields
    if include_label:
        example = example.with_inputs("entity_a", "entity_b")
    else:
        example = example.with_inputs("entity_a", "entity_b")

    return example


def load_dspy_examples(
    path: str = "data/samples/sample_1000.json",
    dev_size: int = 200,
    include_labels: bool = True,
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """Load sample data and split into dev/test DSPy Examples.

    Args:
        path: Path to sample JSON file
        dev_size: Number of pairs for dev set (default: 200)
        include_labels: Include classification labels in examples

    Returns:
        Tuple of (dev_examples, test_examples)
    """
    pairs = load_sample(path)

    dev_pairs = pairs[:dev_size]
    test_pairs = pairs[dev_size:]

    dev_examples = [pair_to_example(p, include_labels) for p in dev_pairs]
    test_examples = [pair_to_example(p, include_labels) for p in test_pairs]

    return dev_examples, test_examples


def load_all_examples(
    path: str = "data/samples/sample_1000.json",
    include_labels: bool = True,
) -> List[dspy.Example]:
    """Load all pairs as DSPy Examples without splitting.

    Args:
        path: Path to sample JSON file
        include_labels: Include classification labels in examples

    Returns:
        List of all examples
    """
    pairs = load_sample(path)
    return [pair_to_example(p, include_labels) for p in pairs]


def get_raw_pairs(
    path: str = "data/samples/sample_1000.json",
    dev_size: int = 200,
) -> Tuple[List[dict], List[dict]]:
    """Load raw pairs split into dev/test for custom processing.

    Args:
        path: Path to sample JSON file
        dev_size: Number of pairs for dev set

    Returns:
        Tuple of (dev_pairs, test_pairs) as raw dicts
    """
    pairs = load_sample(path)
    return pairs[:dev_size], pairs[dev_size:]


if __name__ == "__main__":
    # Test data loading
    import argparse

    parser = argparse.ArgumentParser(description="Test DSPy data loading")
    parser.add_argument("--input", default="data/samples/sample_1000.json")
    parser.add_argument("--dev-size", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    dev, test = load_dspy_examples(args.input, args.dev_size)

    print(f"\nDev set: {len(dev)} examples")
    print(f"Test set: {len(test)} examples")

    # Show sample
    if dev:
        print("\n--- Sample Dev Example ---")
        ex = dev[0]
        print(f"Entity A:\n{ex.entity_a[:200]}...")
        print(f"\nEntity B:\n{ex.entity_b[:200]}...")
        print(f"\nClassification: {ex.classification}")
