#!/usr/bin/env python3
"""
Example selection module for in-context learning.

Provides multiple strategies for selecting high-quality examples
for few-shot entity matching prompts.

Strategies:
1. Random: Baseline random selection
2. Diverse: Cover different entity types and patterns
3. Curated: Hand-picked representative examples

Usage:
    # As library
    from scripts.baselines.example_selector import get_examples
    examples = get_examples(strategy="diverse", n=8)

    # CLI test mode
    python scripts/baselines/example_selector.py --test
    python scripts/baselines/example_selector.py --strategy diverse --n 8
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.load_data import load_sample


MASTER_SEED = 42
DEV_SET_SIZE = 200  # First 200 pairs are dev set
CURATED_EXAMPLES_PATH = "data/examples/curated_examples.json"


def load_dev_pairs(
    path: str = "data/samples/sample_1000.json",
    dev_size: int = DEV_SET_SIZE
) -> List[dict]:
    """Load dev set pairs (first N pairs from sample).

    Args:
        path: Path to sample JSON file
        dev_size: Number of pairs to use as dev set

    Returns:
        List of entity pair dicts
    """
    pairs = load_sample(path)
    return pairs[:dev_size]


def select_random(
    pairs: List[dict],
    n: int,
    seed: int = MASTER_SEED
) -> List[dict]:
    """Random selection from dev set.

    Args:
        pairs: Available pairs to select from
        n: Number of examples to select
        seed: Random seed for reproducibility

    Returns:
        List of n randomly selected pairs
    """
    random.seed(seed)
    return random.sample(pairs, min(n, len(pairs)))


def select_diverse(
    pairs: List[dict],
    n: int,
    seed: int = MASTER_SEED
) -> List[dict]:
    """Select diverse examples covering different patterns.

    Ensures:
    - Mix of positive and negative examples (~60/40 split)
    - Different entity types (Person, Organization)
    - Different attribute patterns (with/without dates, etc.)

    Args:
        pairs: Available pairs to select from
        n: Number of examples to select
        seed: Random seed for reproducibility

    Returns:
        List of diverse example pairs
    """
    random.seed(seed)

    def get_schema(pair: dict) -> str:
        return pair.get("left", {}).get("schema", "Unknown")

    def sample_from_pool(pool: List[dict], count: int) -> List[dict]:
        """Sample up to count items from pool."""
        return random.sample(pool, min(count, len(pool)))

    def select_by_class(
        pairs_list: List[dict],
        target_count: int,
        person_ratio: float = 0.6
    ) -> List[dict]:
        """Select examples with Person/Organization balance."""
        persons = [p for p in pairs_list if get_schema(p) == "Person"]
        orgs = [p for p in pairs_list if get_schema(p) != "Person"]

        n_persons = max(1, int(target_count * person_ratio))
        n_orgs = target_count - n_persons

        result = sample_from_pool(persons, n_persons)
        result.extend(sample_from_pool(orgs, n_orgs))

        # Fill remaining slots from any available
        if len(result) < target_count:
            remaining = [p for p in pairs_list if p not in result]
            result.extend(sample_from_pool(remaining, target_count - len(result)))

        return result

    # Separate by class
    positives = [p for p in pairs if p["judgement"] == "positive"]
    negatives = [p for p in pairs if p["judgement"] == "negative"]

    # Target: ~62.5% positive, ~37.5% negative
    n_pos = max(1, int(n * 0.625))
    n_neg = n - n_pos

    # Select from each class with entity type diversity
    selected = select_by_class(positives, n_pos)
    selected.extend(select_by_class(negatives, n_neg))

    # Shuffle to mix positives and negatives
    random.shuffle(selected)

    return selected[:n]


def select_curated(
    path: str = CURATED_EXAMPLES_PATH,
    n: Optional[int] = None
) -> List[dict]:
    """Load hand-curated representative examples.

    Args:
        path: Path to curated examples JSON file
        n: Optional limit on number of examples (None = all)

    Returns:
        List of curated example pairs
    """
    curated_path = Path(path)
    if not curated_path.exists():
        print(f"Warning: Curated examples file not found at {path}")
        print("Falling back to diverse selection.")
        return []

    with open(curated_path, encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("examples", data) if isinstance(data, dict) else data

    if n is not None:
        examples = examples[:n]

    return examples


def get_examples(
    strategy: str = "diverse",
    n: int = 8,
    seed: int = MASTER_SEED,
    input_file: str = "data/samples/sample_1000.json"
) -> List[dict]:
    """Get examples for in-context learning using specified strategy.

    Args:
        strategy: Selection strategy ("random", "diverse", "curated")
        n: Number of examples to select
        seed: Random seed for reproducibility
        input_file: Path to sample file (for random/diverse strategies)

    Returns:
        List of example pairs

    Raises:
        ValueError: If strategy is unknown
    """
    if strategy == "curated":
        examples = select_curated(n=n)
        if examples:
            return examples
        # Fall back to diverse if curated not available
        strategy = "diverse"
        print("Falling back to diverse selection.")

    pairs = load_dev_pairs(input_file)

    if strategy == "random":
        return select_random(pairs, n, seed)
    elif strategy == "diverse":
        return select_diverse(pairs, n, seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'random', 'diverse', or 'curated'.")


def analyze_examples(examples: List[dict]) -> dict:
    """Analyze selected examples for balance and coverage.

    Args:
        examples: List of example pairs

    Returns:
        Dict with analysis statistics
    """
    n_pos = sum(1 for e in examples if e["judgement"] == "positive")
    n_neg = len(examples) - n_pos

    schemas = {}
    for e in examples:
        schema = e.get("left", {}).get("schema", "Unknown")
        schemas[schema] = schemas.get(schema, 0) + 1

    # Check attribute coverage
    has_date = sum(1 for e in examples
                   if "birthDate" in e.get("left", {}).get("properties", {}))
    has_country = sum(1 for e in examples
                      if "country" in e.get("left", {}).get("properties", {}))
    has_id = sum(1 for e in examples
                 if "idNumber" in e.get("left", {}).get("properties", {}))

    return {
        "total": len(examples),
        "positive": n_pos,
        "negative": n_neg,
        "positive_ratio": n_pos / len(examples) if examples else 0,
        "schemas": schemas,
        "has_birthDate": has_date,
        "has_country": has_country,
        "has_idNumber": has_id,
    }


def print_example_summary(examples: List[dict]) -> None:
    """Print summary of selected examples."""
    analysis = analyze_examples(examples)

    print(f"\nExample Selection Summary:")
    print(f"  Total: {analysis['total']}")
    print(f"  Positive: {analysis['positive']} ({analysis['positive_ratio']:.1%})")
    print(f"  Negative: {analysis['negative']}")
    print(f"  Schemas: {analysis['schemas']}")
    print(f"  With birthDate: {analysis['has_birthDate']}")
    print(f"  With country: {analysis['has_country']}")
    print(f"  With idNumber: {analysis['has_idNumber']}")


def main():
    parser = argparse.ArgumentParser(
        description="Select examples for in-context learning"
    )
    parser.add_argument("--strategy", default="diverse",
                        choices=["random", "diverse", "curated"],
                        help="Selection strategy")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of examples to select")
    parser.add_argument("--seed", type=int, default=MASTER_SEED,
                        help="Random seed")
    parser.add_argument("--input", default="data/samples/sample_1000.json",
                        help="Input sample file")
    parser.add_argument("--test", action="store_true",
                        help="Run test mode: select and analyze examples")
    parser.add_argument("--output", default=None,
                        help="Save selected examples to JSON file")

    args = parser.parse_args()

    if args.test:
        print("Testing example selection strategies:\n")
        for strategy in ["random", "diverse"]:
            print(f"\n=== {strategy.upper()} ===")
            examples = get_examples(
                strategy=strategy,
                n=args.n,
                seed=args.seed,
                input_file=args.input
            )
            print_example_summary(examples)

        print("\n=== CURATED ===")
        curated = select_curated()
        if curated:
            print_example_summary(curated)
        else:
            print("  (No curated examples file found)")

    else:
        examples = get_examples(
            strategy=args.strategy,
            n=args.n,
            seed=args.seed,
            input_file=args.input
        )

        print_example_summary(examples)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"examples": examples}, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
