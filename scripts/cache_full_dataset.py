#!/usr/bin/env python3
"""
Cache Full Dataset

This script loads the full ~755k dataset and creates a JSON cache for faster subsequent loads.
Run this once to create data/raw/pairs_full.json (2.1 GB, gitignored).
"""

import sys
from pathlib import Path

# Import from same directory
from load_data import load_full_dataset, load_sample

# Example 1: Load with caching (recommended)
print("=" * 80)
print("EXAMPLE 1: Load full dataset with caching")
print("=" * 80)
data = load_full_dataset(
    input_path="data/raw/pairs-20251209.json.gz",
    cache_path="data/raw/pairs_full.json",
    use_cache=True  # Default: uses cache if available
)

print(f"\nDataset loaded: {len(data):,} entries")
print(f"First entry keys: {list(data[0].keys())}")
print(f"Judgement distribution:")
from collections import Counter
judgements = Counter(pair['judgement'] for pair in data)
for label, count in judgements.items():
    print(f"  {label}: {count:,} ({count/len(data)*100:.1f}%)")

# Example 2: Force reload from .gz (slower, but ensures fresh data)
# data = load_full_dataset(use_cache=False)

# Example 3: Use sample for quick testing
# sample_data = load_sample("data/samples/sample_1000.json")
