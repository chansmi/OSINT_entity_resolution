#!/usr/bin/env python3
"""
Data loading utilities for OSINT entity resolution benchmark.

This module provides simple functions to load and sample the entity pairs dataset.
"""

import gzip
import json
from pathlib import Path
from typing import Iterator, Dict, Any
from tqdm import tqdm


def load_pairs(filepath: str) -> Iterator[Dict[str, Any]]:
    """
    Load entity pairs from JSON.gz file.
    
    Memory-efficient iterator that doesn't load the entire file into memory.
    
    Args:
        filepath: Path to the JSON.gz file
        
    Yields:
        dict: Entity pair with keys 'judgement', 'left', 'right'
        
    Example:
        >>> for pair in load_pairs("pairs-20251209.json.gz"):
        ...     print(pair["judgement"])
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    print(f"Loading data from {filepath}...")
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def create_sample(input_path: str, output_path: str, n: int = 100):
    """
    Create a small sample from the full dataset.
    
    Useful for quick testing and exploration without loading the full dataset.
    
    Args:
        input_path: Path to the full JSON.gz file
        output_path: Path where the sample will be saved (JSON format)
        n: Number of samples to extract
        
    Example:
        >>> create_sample("pairs-20251209.json.gz", "data/sample_100.json", n=100)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    samples = []
    print(f"Creating sample of {n} pairs...")
    
    for i, pair in enumerate(load_pairs(input_path)):
        if i >= n:
            break
        samples.append(pair)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(samples)} samples to {output_path}")


def load_sample(filepath: str) -> list:
    """
    Load a sample JSON file.
    
    Args:
        filepath: Path to the sample JSON file
        
    Returns:
        list: List of entity pairs
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_full_dataset(
    input_path: str = "data/raw/pairs-20251209.json.gz",
    cache_path: str = "data/raw/pairs_full.json",
    use_cache: bool = True
) -> list:
    """
    Load the entire dataset into memory as a list.
    
    This loads all ~755k entries into memory (~10 GB). Optionally caches
    the uncompressed JSON for faster subsequent loads.
    
    Args:
        input_path: Path to the compressed JSON.gz file
        cache_path: Path to save/load uncompressed JSON cache (in data/raw/, gitignored)
        use_cache: If True, use cached JSON if available; if False, always load from .gz
        
    Returns:
        list: All entity pairs loaded into memory
        
    Example:
        >>> data = load_full_dataset()  # First run: ~30-60s
        >>> len(data)
        755540
        >>> data = load_full_dataset()  # Subsequent runs with cache: ~5-10s
    """
    import time
    
    cache_path = Path(cache_path)
    input_path = Path(input_path)
    
    # Try to load from cache first
    if use_cache and cache_path.exists():
        print(f"Loading from cache: {cache_path}")
        start = time.time()
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        elapsed = time.time() - start
        print(f"✓ Loaded {len(data):,} pairs from cache in {elapsed:.2f}s")
        return data
    
    # Load from compressed file
    print(f"Loading full dataset from {input_path}")
    print("⚠️  This will take 30-60 seconds and use ~10 GB of memory...")
    start = time.time()
    
    data = []
    for pair in tqdm(load_pairs(str(input_path)), desc="Loading pairs"):
        data.append(pair)
    
    elapsed = time.time() - start
    print(f"✓ Loaded {len(data):,} pairs in {elapsed:.2f}s")
    
    # Save cache if requested
    if use_cache:
        print(f"Saving cache to {cache_path} for faster future loads...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"✓ Cache saved (next load will be ~5-10x faster)")
    
    return data


def get_data_stats(filepath: str) -> Dict[str, Any]:
    """
    Calculate basic statistics about the dataset.
    
    Args:
        filepath: Path to the JSON.gz file
        
    Returns:
        dict: Statistics including total count, class distribution, etc.
    """
    stats = {
        'total': 0,
        'positive': 0,
        'negative': 0,
    }
    
    print("Calculating dataset statistics...")
    for pair in tqdm(load_pairs(filepath)):
        stats['total'] += 1
        judgement = pair.get('judgement', '').lower()
        if judgement == 'positive':
            stats['positive'] += 1
        elif judgement == 'negative':
            stats['negative'] += 1
    
    stats['positive_ratio'] = stats['positive'] / stats['total'] if stats['total'] > 0 else 0
    stats['negative_ratio'] = stats['negative'] / stats['total'] if stats['total'] > 0 else 0
    
    return stats


if __name__ == "__main__":
    # When run as a script, create a sample file
    import argparse
    
    parser = argparse.ArgumentParser(description='Create sample data from full dataset')
    parser.add_argument(
        '--input', 
        default='data/raw/pairs-20251209.json.gz',
        help='Input JSON.gz file path'
    )
    parser.add_argument(
        '--output',
        default='data/samples/sample_100.json',
        help='Output sample file path'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=100,
        help='Number of samples to extract'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Calculate and display dataset statistics'
    )
    
    args = parser.parse_args()
    
    if args.stats:
        stats = get_data_stats(args.input)
        print("\nDataset Statistics:")
        print(f"  Total pairs: {stats['total']:,}")
        print(f"  Positive: {stats['positive']:,} ({stats['positive_ratio']:.1%})")
        print(f"  Negative: {stats['negative']:,} ({stats['negative_ratio']:.1%})")
    else:
        create_sample(args.input, args.output, args.n)
