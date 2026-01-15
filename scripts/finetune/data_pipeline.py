#!/usr/bin/env python3
"""
Data pipeline for entity resolution fine-tuning.

Provides functions for:
- Loading and formatting entity pairs
- Creating entity-level train/val/test splits (prevents data leakage)
- Converting pairs to instruction-following format
"""

import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterator, Set
from collections import defaultdict
from tqdm import tqdm

from .config import SYSTEM_PROMPT, FIELD_SPECS, CLASS_WEIGHTS, SPLIT_CONFIG


def compute_file_checksum(filepath: str) -> str:
    """
    Compute SHA256 checksum of a file for reproducibility verification.

    Used to ensure the exact same test set is used across all evaluations.
    """
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_no_entity_overlap(
    train_entities: Set[str],
    val_entities: Set[str],
    test_entities: Set[str]
) -> Dict[str, int]:
    """
    Verify zero overlap between entity splits.

    Raises AssertionError if data leakage is detected.

    Returns:
        Dict with overlap counts (all should be 0)
    """
    train_val_overlap = train_entities & val_entities
    train_test_overlap = train_entities & test_entities
    val_test_overlap = val_entities & test_entities

    overlap_counts = {
        "train_val": len(train_val_overlap),
        "train_test": len(train_test_overlap),
        "val_test": len(val_test_overlap),
    }

    assert len(train_val_overlap) == 0, \
        f"DATA LEAKAGE: Train-Val overlap: {len(train_val_overlap)} entities"
    assert len(train_test_overlap) == 0, \
        f"DATA LEAKAGE: Train-Test overlap: {len(train_test_overlap)} entities"
    assert len(val_test_overlap) == 0, \
        f"DATA LEAKAGE: Val-Test overlap: {len(val_test_overlap)} entities"

    print("✓ Verified: Zero entity overlap between splits")
    return overlap_counts


def format_entity_for_prompt(entity: Dict[str, Any], max_fields: int = 10) -> str:
    """
    Convert entity properties to readable text for the prompt.

    Args:
        entity: Entity dict with 'schema' and 'properties' keys
        max_fields: Maximum number of fields to include

    Returns:
        Formatted string representation of the entity
    """
    props = entity.get("properties", {})
    lines = [f"Type: {entity.get('schema', 'Unknown')}"]

    fields_added = 0
    for key, label, limit in FIELD_SPECS:
        if key in props and props[key]:
            values = props[key]
            if limit:
                values = values[:limit]
            # Join values, handling potential non-strings
            value_str = ", ".join(str(v) for v in values if v)
            if value_str:
                lines.append(f"{label}: {value_str}")
                fields_added += 1
                if fields_added >= max_fields:
                    break

    return "\n".join(lines)


def create_training_example(pair: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an entity pair to instruction-following training format.

    Args:
        pair: Dict with 'left', 'right', and 'judgement' keys

    Returns:
        Dict with 'messages' key in chat format
    """
    user_content = f"""Compare these two entities:

=== ENTITY A ===
{format_entity_for_prompt(pair['left'])}

=== ENTITY B ===
{format_entity_for_prompt(pair['right'])}

Are these the same entity?"""

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": pair['judgement']}
        ]
    }


def load_pairs(filepath: str) -> Iterator[Dict[str, Any]]:
    """
    Load entity pairs from JSON.gz or JSON file.

    Memory-efficient iterator that doesn't load everything at once.
    """
    filepath = Path(filepath)

    if filepath.suffix == '.gz':
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == '[':
                # JSON array format
                data = json.load(f)
                for item in data:
                    yield item
            elif first_char == '{':
                # Could be v2 format with metadata or JSONL
                content = json.load(f)
                if 'pairs' in content:
                    # v2 format with metadata
                    for item in content['pairs']:
                        yield item
                else:
                    # Single object - probably first line of JSONL
                    f.seek(0)
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            else:
                # JSONL format
                for line in f:
                    if line.strip():
                        yield json.loads(line)


def extract_entity_ids(pairs: List[Dict[str, Any]]) -> Tuple[Set[str], Dict[str, List[int]]]:
    """
    Extract all unique entity IDs and map them to pair indices.

    Returns:
        - Set of all unique entity IDs
        - Dict mapping entity ID to list of pair indices containing that entity
    """
    all_entity_ids = set()
    entity_to_pairs = defaultdict(list)

    for idx, pair in enumerate(pairs):
        left_id = pair['left'].get('id', f"left_{idx}")
        right_id = pair['right'].get('id', f"right_{idx}")

        all_entity_ids.add(left_id)
        all_entity_ids.add(right_id)

        entity_to_pairs[left_id].append(idx)
        entity_to_pairs[right_id].append(idx)

    return all_entity_ids, dict(entity_to_pairs)


def create_entity_level_splits(
    pairs: List[Dict[str, Any]],
    train_ratio: float = SPLIT_CONFIG["train_ratio"],  # 0.70
    val_ratio: float = SPLIT_CONFIG["val_ratio"],      # 0.05
    seed: int = SPLIT_CONFIG["seed"]                   # 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Set[str]]]:
    """
    Split pairs into train/val/test sets by entity ID to prevent data leakage.

    An entity appearing in training cannot appear in validation or test.
    Pairs where entities span splits are assigned to training.

    Args:
        pairs: List of entity pair dicts
        train_ratio: Fraction of entities for training (default: 0.70)
        val_ratio: Fraction of entities for validation (default: 0.05)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_pairs, val_pairs, test_pairs, entity_sets)
        where entity_sets is a dict with 'train', 'val', 'test' keys mapping to entity ID sets
    """
    random.seed(seed)

    # Extract all entity IDs
    all_entity_ids, entity_to_pairs = extract_entity_ids(pairs)
    entity_list = list(all_entity_ids)
    random.shuffle(entity_list)

    # Split entity IDs
    n_entities = len(entity_list)
    train_end = int(n_entities * train_ratio)
    val_end = int(n_entities * (train_ratio + val_ratio))

    train_entities = set(entity_list[:train_end])
    val_entities = set(entity_list[train_end:val_end])
    test_entities = set(entity_list[val_end:])

    print(f"Entity split: {len(train_entities)} train, {len(val_entities)} val, {len(test_entities)} test")

    # Assign pairs to splits
    train_pairs = []
    val_pairs = []
    test_pairs = []

    for pair in pairs:
        left_id = pair['left'].get('id', 'unknown')
        right_id = pair['right'].get('id', 'unknown')

        left_in_train = left_id in train_entities
        left_in_val = left_id in val_entities
        left_in_test = left_id in test_entities

        right_in_train = right_id in train_entities
        right_in_val = right_id in val_entities
        right_in_test = right_id in test_entities

        # Both entities must be in the same split, otherwise assign to train
        if left_in_train or right_in_train:
            train_pairs.append(pair)
        elif left_in_val and right_in_val:
            val_pairs.append(pair)
        elif left_in_test and right_in_test:
            test_pairs.append(pair)
        else:
            # Cross-split pair - assign to training
            train_pairs.append(pair)

    # Shuffle each split
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    random.shuffle(test_pairs)

    # Verify no entity overlap (raises AssertionError if leakage detected)
    verify_no_entity_overlap(train_entities, val_entities, test_entities)

    # Return entity sets for metadata
    entity_sets = {
        "train": train_entities,
        "val": val_entities,
        "test": test_entities,
    }

    return train_pairs, val_pairs, test_pairs, entity_sets


def compute_class_distribution(pairs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute the class distribution of a dataset."""
    total = len(pairs)
    positive = sum(1 for p in pairs if p['judgement'] == 'positive')
    negative = total - positive

    return {
        'total': total,
        'positive': positive,
        'negative': negative,
        'positive_ratio': positive / total if total > 0 else 0,
        'negative_ratio': negative / total if total > 0 else 0,
    }


def save_processed_data(
    pairs: List[Dict[str, Any]],
    output_path: str,
    format_fn=create_training_example
) -> int:
    """
    Convert pairs to training format and save as JSONL.

    Args:
        pairs: List of entity pairs
        output_path: Path to save JSONL file
        format_fn: Function to convert pair to training format

    Returns:
        Number of examples saved
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in tqdm(pairs, desc=f"Saving to {output_path.name}"):
            example = format_fn(pair)
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    return count


if __name__ == "__main__":
    # Quick test
    test_pair = {
        "left": {
            "id": "test-1",
            "schema": "Person",
            "properties": {
                "name": ["John Smith"],
                "birthDate": ["1980-01-15"],
                "nationality": ["us"]
            }
        },
        "right": {
            "id": "test-2",
            "schema": "Person",
            "properties": {
                "name": ["J. Smith", "John A. Smith"],
                "birthDate": ["1980-01-15"],
                "country": ["us"]
            }
        },
        "judgement": "positive"
    }

    print("Test entity formatting:")
    print("=" * 50)
    example = create_training_example(test_pair)
    print(json.dumps(example, indent=2))
