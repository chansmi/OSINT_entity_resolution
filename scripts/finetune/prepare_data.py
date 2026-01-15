#!/usr/bin/env python3
"""
Prepare training data for entity resolution fine-tuning.

This script:
1. Loads the full dataset from pairs-20251209.json.gz
2. Creates entity-level train/val/test splits (prevents data leakage)
3. Converts pairs to instruction-following format (ChatML)
4. Saves universal test set in V2 format for ALL methods
5. Computes SHA256 checksum for reproducibility verification

Default splits: 70% train, 5% val, 25% test (entity-level)

Usage:
    python scripts/finetune/prepare_data.py \
        --input data/raw/pairs-20251209.json.gz \
        --output-dir data/processed \
        --train-ratio 0.70 --val-ratio 0.05 --seed 42
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tqdm import tqdm
from scripts.finetune.data_pipeline import (
    load_pairs,
    create_entity_level_splits,
    compute_class_distribution,
    save_processed_data,
    compute_file_checksum,
)
from scripts.finetune.config import SPLIT_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for entity resolution fine-tuning"
    )
    parser.add_argument(
        "--input",
        default="data/raw/pairs-20251209.json.gz",
        help="Input file (JSON.gz or JSON)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=SPLIT_CONFIG["train_ratio"],  # 0.70
        help="Fraction of data for training (default: 0.70)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=SPLIT_CONFIG["val_ratio"],    # 0.05
        help="Fraction of data for validation (default: 0.05)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of pairs to load (for testing)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Entity Resolution Data Preparation")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Split ratios: {args.train_ratio:.0%} train, {args.val_ratio:.0%} val, {1-args.train_ratio-args.val_ratio:.0%} test")
    print(f"Seed: {args.seed}")
    print()

    # Load data
    print("Loading data...")
    pairs = []
    for pair in tqdm(load_pairs(str(input_path)), desc="Loading pairs"):
        pairs.append(pair)
        if args.limit and len(pairs) >= args.limit:
            break

    print(f"Loaded {len(pairs):,} pairs")

    # Show class distribution
    dist = compute_class_distribution(pairs)
    print(f"Class distribution: {dist['positive_ratio']:.1%} positive, {dist['negative_ratio']:.1%} negative")
    print()

    # Create entity-level splits (with verification)
    print("Creating entity-level splits...")
    train_pairs, val_pairs, test_pairs, entity_sets = create_entity_level_splits(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_pairs):,} pairs")
    print(f"  Val:   {len(val_pairs):,} pairs")
    print(f"  Test:  {len(test_pairs):,} pairs")
    print()

    # Show class distribution per split
    for name, split_pairs in [("Train", train_pairs), ("Val", val_pairs), ("Test", test_pairs)]:
        dist = compute_class_distribution(split_pairs)
        print(f"{name}: {dist['positive_ratio']:.1%} positive, {dist['negative_ratio']:.1%} negative")
    print()

    # Save processed data (ChatML format for fine-tuning)
    print("Converting to training format and saving...")
    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"
    test_file = output_dir / "test.jsonl"

    n_train = save_processed_data(train_pairs, train_file)
    n_val = save_processed_data(val_pairs, val_file)
    n_test = save_processed_data(test_pairs, test_file)

    # Save universal test set (V2 raw format for ALL methods)
    print("\nSaving universal test set (V2 format)...")
    test_pairs_file = output_dir / "test_pairs.json"
    test_dist = compute_class_distribution(test_pairs)

    test_set_data = {
        "metadata": {
            "description": "Universal held-out test set (25% of data)",
            "format": "v2",
            "created": datetime.now().isoformat(),
            "seed": args.seed,
            "n_pairs": len(test_pairs),
            "n_positive": test_dist["positive"],
            "n_negative": test_dist["negative"],
            "positive_ratio": test_dist["positive_ratio"],
            "compatible_with": [
                "scripts/baselines/simple_fuzzy.py",
                "scripts/baselines/nomenklatura_v1.py",
                "scripts/baselines/llm_zeroshot.py",
                "scripts/finetune/inference.py"
            ]
        },
        "pairs": test_pairs  # Raw format with left/right/judgement
    }

    with open(test_pairs_file, 'w', encoding='utf-8') as f:
        json.dump(test_set_data, f, ensure_ascii=False)

    # Compute checksum for reproducibility verification
    test_set_checksum = compute_file_checksum(str(test_pairs_file))
    print(f"✓ Test set saved: {test_pairs_file}")
    print(f"✓ SHA256 checksum: {test_set_checksum}")

    print()
    print("=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"Train:      {train_file} ({n_train:,} examples)")
    print(f"Val:        {val_file} ({n_val:,} examples)")
    print(f"Test:       {test_file} ({n_test:,} examples)")
    print(f"Test (V2):  {test_pairs_file} ({len(test_pairs):,} pairs)")

    # Save metadata with checksums and entity counts
    metadata = {
        "created": datetime.now().isoformat(),
        "input_file": str(input_path),
        "seed": args.seed,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1 - args.train_ratio - args.val_ratio,
        },
        "splits": {
            "train": {"file": str(train_file), "n_examples": n_train},
            "val": {"file": str(val_file), "n_examples": n_val},
            "test": {"file": str(test_file), "n_examples": n_test},
            "test_pairs": {"file": str(test_pairs_file), "n_pairs": len(test_pairs)},
        },
        "total_pairs": len(pairs),
        "test_set_checksum": test_set_checksum,
        "entity_counts": {
            "train_entities": len(entity_sets["train"]),
            "val_entities": len(entity_sets["val"]),
            "test_entities": len(entity_sets["test"]),
            "overlap_train_val": 0,   # Verified by verify_no_entity_overlap()
            "overlap_train_test": 0,
            "overlap_val_test": 0,
        },
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata:   {metadata_file}")


if __name__ == "__main__":
    main()
