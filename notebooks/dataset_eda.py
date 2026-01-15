#!/usr/bin/env python3
"""
Comprehensive EDA for OpenSanctions Entity Resolution Dataset.

This script analyzes the full dataset and saves key statistics to JSON
for use in a paper describing the dataset and benchmark.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.load_data import load_full_dataset


def extract_entity_stats(entity: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key statistics from a single entity."""
    props = entity.get('properties', {})
    
    # Count aliases (name + alias fields)
    names = props.get('name', [])
    aliases = props.get('alias', [])
    total_aliases = len(names) + len(aliases)
    
    # Count other fields
    nationalities = props.get('nationality', [])
    countries = props.get('country', [])
    addresses = props.get('address', [])
    datasets = entity.get('datasets', [])
    
    return {
        'id': entity.get('id'),
        'schema': entity.get('schema'),
        'num_datasets': len(datasets),
        'num_aliases': total_aliases,
        'num_nationalities': len(nationalities),
        'num_countries': len(countries),
        'num_addresses': len(addresses),
        'datasets': datasets,
        'properties': set(props.keys()),
    }


def compute_descriptive_stats(values: List[int]) -> Dict[str, float]:
    """Compute mean, median, max for a list of values."""
    if not values:
        return {'mean': 0, 'median': 0, 'max': 0, 'min': 0}
    return {
        'mean': round(mean(values), 2),
        'median': round(median(values), 2),
        'max': max(values),
        'min': min(values),
    }


def infer_country_from_dataset(dataset_name: str) -> str:
    """Infer country from dataset name prefix."""
    # Common prefixes
    country_prefixes = {
        'us_': 'United States',
        'gb_': 'United Kingdom', 
        'uk_': 'United Kingdom',
        'eu_': 'European Union',
        'un_': 'United Nations',
        'au_': 'Australia',
        'ca_': 'Canada',
        'ch_': 'Switzerland',
        'de_': 'Germany',
        'fr_': 'France',
        'nl_': 'Netherlands',
        'be_': 'Belgium',
        'pl_': 'Poland',
        'ua_': 'Ukraine',
        'ru_': 'Russia',
        'jp_': 'Japan',
        'kr_': 'South Korea',
        'cn_': 'China',
        'in_': 'India',
        'za_': 'South Africa',
        'br_': 'Brazil',
        'ar_': 'Argentina',
        'mx_': 'Mexico',
        'nz_': 'New Zealand',
        'il_': 'Israel',
        'tr_': 'Turkey',
        'sg_': 'Singapore',
        'hk_': 'Hong Kong',
        'ae_': 'UAE',
        'qa_': 'Qatar',
        'kz_': 'Kazakhstan',
        'kp_': 'North Korea',
    }
    
    for prefix, country in country_prefixes.items():
        if dataset_name.lower().startswith(prefix):
            return country
    return 'Other/International'


def analyze_field_lengths(data: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Analyze text field lengths across all entities."""
    field_lengths = defaultdict(list)
    
    for pair in data:
        for side in ['left', 'right']:
            entity = pair.get(side, {})
            props = entity.get('properties', {})
            
            for field, values in props.items():
                if isinstance(values, list):
                    for val in values:
                        if isinstance(val, str):
                            field_lengths[field].append(len(val))
    
    return {
        field: compute_descriptive_stats(lengths)
        for field, lengths in field_lengths.items()
    }


def analyze_missing_values(data: List[Dict], all_fields: set) -> Dict[str, float]:
    """Calculate percentage of entities missing each field."""
    total_entities = len(data) * 2  # left + right
    field_present = Counter()
    
    for pair in data:
        for side in ['left', 'right']:
            props = pair.get(side, {}).get('properties', {})
            for field in props.keys():
                if props[field]:  # Non-empty
                    field_present[field] += 1
    
    missing_pct = {}
    for field in sorted(all_fields):
        present = field_present.get(field, 0)
        missing_pct[field] = round((1 - present / total_entities) * 100, 2)
    
    return missing_pct


def run_eda(data: List[Dict]) -> Dict[str, Any]:
    """Run comprehensive EDA on the dataset."""
    print(f"Analyzing {len(data):,} pairs...")
    
    # Class balance
    positive_count = sum(1 for p in data if p.get('judgement') == 'positive')
    negative_count = len(data) - positive_count
    
    # Collect all unique entities
    entities = {}
    all_property_fields = set()
    dataset_counter = Counter()
    schema_counter = Counter()
    
    entity_stats_list = []
    
    for pair in data:
        for side in ['left', 'right']:
            entity = pair.get(side, {})
            entity_id = entity.get('id')
            
            if entity_id and entity_id not in entities:
                entities[entity_id] = entity
                stats = extract_entity_stats(entity)
                entity_stats_list.append(stats)
                
                # Aggregate
                all_property_fields.update(stats['properties'])
                schema_counter[stats['schema']] += 1
                for ds in stats['datasets']:
                    dataset_counter[ds] += 1
    
    print(f"Found {len(entities):,} unique entities")
    
    # Entity descriptive statistics
    datasets_per_entity = [s['num_datasets'] for s in entity_stats_list]
    aliases_per_entity = [s['num_aliases'] for s in entity_stats_list]
    nationalities_per_entity = [s['num_nationalities'] for s in entity_stats_list]
    addresses_per_entity = [s['num_addresses'] for s in entity_stats_list]
    
    # Datasets by country
    datasets_by_country = defaultdict(list)
    for ds in dataset_counter.keys():
        country = infer_country_from_dataset(ds)
        datasets_by_country[country].append(ds)
    
    # Field analysis
    field_lengths = analyze_field_lengths(data)
    missing_values = analyze_missing_values(data, all_property_fields)
    
    # Build results
    results = {
        'main_stats': {
            'total_pairs': len(data),
            'unique_entities': len(entities),
            'class_balance': {
                'positive': positive_count,
                'negative': negative_count,
                'positive_ratio': round(positive_count / len(data), 4),
                'negative_ratio': round(negative_count / len(data), 4),
            },
            'entity_descriptives': {
                'datasets_per_entity': compute_descriptive_stats(datasets_per_entity),
                'aliases_per_entity': compute_descriptive_stats(aliases_per_entity),
                'nationalities_per_entity': compute_descriptive_stats(nationalities_per_entity),
                'addresses_per_entity': compute_descriptive_stats(addresses_per_entity),
            },
            'dataset_coverage': {
                'num_datasets': len(dataset_counter),
                'top_20_datasets': dataset_counter.most_common(20),
                'datasets_by_country': {
                    country: len(datasets) 
                    for country, datasets in sorted(datasets_by_country.items(), key=lambda x: -len(x[1]))
                },
            },
            'entity_fields': sorted(all_property_fields),
            'field_lengths': {
                k: v for k, v in sorted(field_lengths.items(), key=lambda x: -x[1].get('mean', 0))[:15]
            },
        },
        'appendix_stats': {
            'missing_values_by_field': missing_values,
            'full_dataset_list': [
                {'dataset': ds, 'count': count} 
                for ds, count in dataset_counter.most_common()
            ],
            'schema_distribution': dict(schema_counter.most_common()),
        },
    }
    
    return results


def main():
    """Main entry point."""
    print("=" * 60)
    print("OpenSanctions Dataset EDA")
    print("=" * 60)
    
    # Load data
    data = load_full_dataset(
        input_path='../data/raw/pairs-20251209.json.gz',
        cache_path='../data/raw/pairs_full.json',
        use_cache=True
    )
    
    # Run analysis
    results = run_eda(data)
    
    # Save results
    output_path = Path(__file__).parent / 'eda_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Print summary
    main_stats = results['main_stats']
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total pairs: {main_stats['total_pairs']:,}")
    print(f"Unique entities: {main_stats['unique_entities']:,}")
    print(f"Positive pairs: {main_stats['class_balance']['positive']:,} ({main_stats['class_balance']['positive_ratio']:.1%})")
    print(f"Negative pairs: {main_stats['class_balance']['negative']:,} ({main_stats['class_balance']['negative_ratio']:.1%})")
    print(f"Number of datasets: {main_stats['dataset_coverage']['num_datasets']}")
    print(f"\nEntity descriptives:")
    for key, stats in main_stats['entity_descriptives'].items():
        print(f"  {key}: mean={stats['mean']}, median={stats['median']}, max={stats['max']}")


if __name__ == '__main__':
    main()
