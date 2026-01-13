#!/usr/bin/env python3
"""
Prepare Supervised Fine-Tuning (SFT) dataset for entity matching.

Creates training data in multiple formats:
1. OpenAI format (messages array) - for OpenAI fine-tuning API
2. Alpaca format (instruction/input/output) - for Llama/Mistral with Axolotl
3. ShareGPT format (conversations array) - alternative open-source format

Usage:
    # Create SFT dataset from 10K sample
    python scripts/finetune/prepare_sft_data.py --input data/samples/sample_10000.json

    # Validate only (no file creation)
    python scripts/finetune/prepare_sft_data.py --validate-only

    # Custom sizes
    python scripts/finetune/prepare_sft_data.py --n-train 1000 --n-val 100

    # Preview examples
    python scripts/finetune/prepare_sft_data.py --preview 3

    # From notebook
    from scripts.finetune.prepare_sft_data import prepare_dataset
    prepare_dataset("data/samples/sample_10000.json", "data/finetune")
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.baselines.llm_zeroshot import SYSTEM_PROMPT, format_entity, USER_TEMPLATE


MASTER_SEED = 42


def generate_reasoning(pair: dict) -> str:
    """Generate detailed reasoning for training example.

    Analyzes entity properties to create human-readable reasoning
    that explains why the pair is positive or negative.

    Args:
        pair: Entity pair dict with left, right, judgement

    Returns:
        Reasoning string
    """
    left = pair["left"].get("properties", {})
    right = pair["right"].get("properties", {})

    evidence = []
    conflicts = []

    # Check names
    if "name" in left and "name" in right:
        left_names = set(n.lower() for n in left["name"])
        right_names = set(n.lower() for n in right["name"])
        if left_names & right_names:
            evidence.append("names match")
        else:
            # Check for partial overlap (shared tokens)
            left_tokens = set(w for n in left_names for w in n.split())
            right_tokens = set(w for n in right_names for w in n.split())
            if left_tokens & right_tokens:
                evidence.append("name tokens overlap")

    # Check birth dates
    if "birthDate" in left and "birthDate" in right:
        left_dates = set(left["birthDate"])
        right_dates = set(right["birthDate"])
        if left_dates & right_dates:
            evidence.append("birth dates match")
        else:
            conflicts.append(f"different birth dates ({list(left_dates)[0]} vs {list(right_dates)[0]})")

    # Check countries
    if "country" in left and "country" in right:
        if set(left["country"]) & set(right["country"]):
            evidence.append("countries match")

    # Check nationality
    if "nationality" in left and "nationality" in right:
        if set(left["nationality"]) & set(right["nationality"]):
            evidence.append("nationalities match")

    # Check identifiers
    for id_field in ["idNumber", "passportNumber"]:
        if id_field in left and id_field in right:
            if set(left[id_field]) & set(right[id_field]):
                evidence.append(f"{id_field} matches")
            else:
                conflicts.append(f"different {id_field}")

    # Check gender
    if "gender" in left and "gender" in right:
        if set(left["gender"]) & set(right["gender"]):
            evidence.append("gender matches")
        else:
            conflicts.append("different genders")

    # Build reasoning string
    if pair["judgement"] == "positive":
        if evidence:
            return f"Same entity. Evidence: {', '.join(evidence)}. No contradictions found."
        else:
            return "Same entity. Consistent attributes, no contradictions found."
    else:
        if conflicts:
            return f"Different entities. Conflicts: {', '.join(conflicts)}."
        else:
            return "Different entities. Insufficient matching evidence to confirm same person."


def create_openai_format(pair: dict) -> dict:
    """Create OpenAI fine-tuning format.

    Format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "{json response}"}
        ]
    }
    """
    user_prompt = USER_TEMPLATE.format(
        entity_a=format_entity(pair["left"]),
        entity_b=format_entity(pair["right"])
    )

    response = {
        "classification": pair["judgement"],
        "confidence": "high",
        "reasoning": generate_reasoning(pair)
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(response)}
        ]
    }


def create_alpaca_format(pair: dict) -> dict:
    """Create Alpaca/Llama instruction format.

    Format:
    {
        "instruction": "system prompt",
        "input": "user prompt",
        "output": "response"
    }
    """
    user_prompt = USER_TEMPLATE.format(
        entity_a=format_entity(pair["left"]),
        entity_b=format_entity(pair["right"])
    )

    response = {
        "classification": pair["judgement"],
        "confidence": "high",
        "reasoning": generate_reasoning(pair)
    }

    return {
        "instruction": SYSTEM_PROMPT,
        "input": user_prompt,
        "output": json.dumps(response)
    }


def create_sharegpt_format(pair: dict) -> dict:
    """Create ShareGPT conversation format.

    Format:
    {
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }
    """
    user_prompt = USER_TEMPLATE.format(
        entity_a=format_entity(pair["left"]),
        entity_b=format_entity(pair["right"])
    )

    response = {
        "classification": pair["judgement"],
        "confidence": "high",
        "reasoning": generate_reasoning(pair)
    }

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": json.dumps(response)}
        ]
    }


def load_data(input_path: str) -> List[dict]:
    """Load entity pairs from JSON file."""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle both formats
    if isinstance(data, dict) and "pairs" in data:
        return data["pairs"]
    return data


def stratified_sample(
    pairs: List[dict],
    n: int,
    positive_ratio: float = 0.77,
    seed: int = MASTER_SEED
) -> List[dict]:
    """Create stratified sample maintaining class ratio.

    Args:
        pairs: All available pairs
        n: Number to sample
        positive_ratio: Target ratio of positive examples
        seed: Random seed

    Returns:
        Stratified sample of pairs
    """
    random.seed(seed)

    positives = [p for p in pairs if p["judgement"] == "positive"]
    negatives = [p for p in pairs if p["judgement"] == "negative"]

    n_pos = int(n * positive_ratio)
    n_neg = n - n_pos

    # Sample (or take all if not enough)
    sampled_pos = random.sample(positives, min(n_pos, len(positives)))
    sampled_neg = random.sample(negatives, min(n_neg, len(negatives)))

    result = sampled_pos + sampled_neg
    random.shuffle(result)

    return result


def prepare_dataset(
    input_path: str,
    output_dir: str,
    n_train: int = 2000,
    n_val: int = 200,
    seed: int = MASTER_SEED,
    train_pool_ratio: float = 0.2
) -> dict:
    """Prepare SFT datasets in multiple formats.

    Args:
        input_path: Path to input sample JSON
        output_dir: Output directory for generated files
        n_train: Number of training examples
        n_val: Number of validation examples
        seed: Random seed
        train_pool_ratio: Fraction of data to use as training pool

    Returns:
        Dict with metadata about created datasets
    """
    random.seed(seed)

    # Load data
    pairs = load_data(input_path)
    print(f"Loaded {len(pairs)} pairs from {input_path}")

    # Use first portion as training pool (don't overlap with test set)
    train_pool_size = int(len(pairs) * train_pool_ratio)
    train_pool = pairs[:train_pool_size]

    print(f"Training pool: {len(train_pool)} pairs (first {train_pool_ratio:.0%})")

    # Stratified sampling for train set
    train_pairs = stratified_sample(train_pool, n_train, seed=seed)

    # Validation set from remaining pool
    remaining = [p for p in train_pool if p not in train_pairs]
    val_pairs = stratified_sample(remaining, min(n_val, len(remaining)), seed=seed + 1)

    print(f"Training: {len(train_pairs)} pairs")
    print(f"Validation: {len(val_pairs)} pairs")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Format converters
    formats = {
        "openai": create_openai_format,
        "alpaca": create_alpaca_format,
        "sharegpt": create_sharegpt_format
    }

    # Write all formats
    for fmt_name, fmt_fn in formats.items():
        # Training data
        train_file = output_path / f"train_{fmt_name}.jsonl"
        with open(train_file, "w", encoding="utf-8") as f:
            for pair in train_pairs:
                f.write(json.dumps(fmt_fn(pair), ensure_ascii=False) + "\n")
        print(f"  Created: {train_file}")

        # Validation data
        val_file = output_path / f"val_{fmt_name}.jsonl"
        with open(val_file, "w", encoding="utf-8") as f:
            for pair in val_pairs:
                f.write(json.dumps(fmt_fn(pair), ensure_ascii=False) + "\n")
        print(f"  Created: {val_file}")

    # Calculate statistics
    train_pos = sum(1 for p in train_pairs if p["judgement"] == "positive")
    val_pos = sum(1 for p in val_pairs if p["judgement"] == "positive")

    # Write metadata
    metadata = {
        "source": input_path,
        "seed": seed,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "train_positive_ratio": train_pos / len(train_pairs) if train_pairs else 0,
        "val_positive_ratio": val_pos / len(val_pairs) if val_pairs else 0,
        "formats": list(formats.keys()),
        "train_pool_size": train_pool_size,
        "train_pool_ratio": train_pool_ratio
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Created: {metadata_file}")

    print(f"\nSFT dataset created:")
    print(f"  Training: {metadata['n_train']} examples ({metadata['train_positive_ratio']:.1%} positive)")
    print(f"  Validation: {metadata['n_val']} examples ({metadata['val_positive_ratio']:.1%} positive)")
    print(f"  Formats: {', '.join(metadata['formats'])}")
    print(f"  Output: {output_path}")

    return metadata


def validate_dataset(output_dir: str) -> dict:
    """Validate existing SFT dataset files.

    Args:
        output_dir: Directory containing SFT data files

    Returns:
        Dict with validation results
    """
    output_path = Path(output_dir)
    results = {"valid": True, "errors": [], "stats": {}}

    # Check metadata
    metadata_file = output_path / "metadata.json"
    if not metadata_file.exists():
        results["errors"].append("metadata.json not found")
        results["valid"] = False
    else:
        with open(metadata_file) as f:
            results["metadata"] = json.load(f)

    # Check format files
    for fmt in ["openai", "alpaca", "sharegpt"]:
        for split in ["train", "val"]:
            filepath = output_path / f"{split}_{fmt}.jsonl"
            if not filepath.exists():
                results["errors"].append(f"{filepath.name} not found")
                results["valid"] = False
                continue

            # Validate JSONL format
            count = 0
            try:
                with open(filepath) as f:
                    for i, line in enumerate(f):
                        example = json.loads(line)
                        count += 1

                        # Check required keys
                        if fmt == "openai" and "messages" not in example:
                            results["errors"].append(f"{filepath.name} line {i+1}: missing 'messages'")
                        elif fmt == "alpaca" and "instruction" not in example:
                            results["errors"].append(f"{filepath.name} line {i+1}: missing 'instruction'")
                        elif fmt == "sharegpt" and "conversations" not in example:
                            results["errors"].append(f"{filepath.name} line {i+1}: missing 'conversations'")

                results["stats"][f"{split}_{fmt}"] = count
            except json.JSONDecodeError as e:
                results["errors"].append(f"{filepath.name}: JSON parse error - {e}")
                results["valid"] = False

    return results


def preview_examples(input_path: str, n: int = 3) -> None:
    """Preview formatted training examples.

    Args:
        input_path: Path to input sample JSON
        n: Number of examples to preview
    """
    pairs = load_data(input_path)[:n]

    print(f"\nPreviewing {n} training examples:\n")

    for i, pair in enumerate(pairs):
        print(f"{'='*60}")
        print(f"Example {i+1}: {pair['judgement'].upper()}")
        print(f"{'='*60}")

        print(f"\nLeft entity: {pair['left'].get('caption', 'Unknown')}")
        print(f"Right entity: {pair['right'].get('caption', 'Unknown')}")

        print(f"\nGenerated reasoning:")
        print(f"  {generate_reasoning(pair)}")

        # Show OpenAI format
        openai_fmt = create_openai_format(pair)
        print(f"\nOpenAI format (assistant response):")
        print(f"  {openai_fmt['messages'][2]['content']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT dataset for entity matching"
    )
    parser.add_argument("--input", default="data/samples/sample_10000.json",
                        help="Input sample file")
    parser.add_argument("--output", default="data/finetune",
                        help="Output directory")
    parser.add_argument("--n-train", type=int, default=2000,
                        help="Number of training examples")
    parser.add_argument("--n-val", type=int, default=200,
                        help="Number of validation examples")
    parser.add_argument("--seed", type=int, default=MASTER_SEED,
                        help="Random seed")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate existing dataset without creating new one")
    parser.add_argument("--preview", type=int, default=0,
                        help="Preview N examples (0 = no preview)")

    args = parser.parse_args()

    if args.preview > 0:
        preview_examples(args.input, args.preview)
        return

    if args.validate_only:
        print(f"Validating existing SFT dataset at {args.output}...")
        results = validate_dataset(args.output)

        if results["valid"]:
            print("\nValidation PASSED")
            print(f"\nStatistics:")
            for key, count in results.get("stats", {}).items():
                print(f"  {key}: {count} examples")
        else:
            print("\nValidation FAILED")
            print("Errors:")
            for error in results["errors"]:
                print(f"  - {error}")
        return

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print("\nTo create the 10K sample, run:")
        print("  python scripts/create_proper_sample.py --n 10000 --output data/samples/sample_10000.json")
        sys.exit(1)

    prepare_dataset(
        input_path=args.input,
        output_dir=args.output,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
