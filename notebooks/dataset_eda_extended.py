#!/usr/bin/env python3
"""
Extended EDA for OpenSanctions Pairs (NeurIPS E&D reviewer response).

Computes all statistics required by paper/REVIEWER_RESPONSE_PLAN.md
in a single streaming pass over data/raw/pairs-20251209.json.gz.

Outputs notebooks/eda_extended_results.json.
"""

from __future__ import annotations

import gzip
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "raw" / "pairs-20251209.json.gz"
OUTPUT = ROOT / "notebooks" / "eda_extended_results.json"

CUTOFF_DATE = "2025-07-01"

SCRIPT_BLOCKS: List[Tuple[str, Tuple[Tuple[int, int], ...]]] = [
    ("Latin", ((0x0041, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF))),
    ("Cyrillic", ((0x0400, 0x04FF), (0x0500, 0x052F))),
    ("Arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("Hebrew", ((0x0590, 0x05FF),)),
    ("Greek", ((0x0370, 0x03FF),)),
    ("Devanagari", ((0x0900, 0x097F),)),
    ("CJK", (
        (0x4E00, 0x9FFF),
        (0x3400, 0x4DBF),
        (0x20000, 0x2A6DF),
        (0x3040, 0x309F),
        (0x30A0, 0x30FF),
        (0xAC00, 0xD7AF),
    )),
    ("Thai", ((0x0E00, 0x0E7F),)),
]


def codepoint_script(cp: int) -> str | None:
    for name, ranges in SCRIPT_BLOCKS:
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    return None


def detect_scripts(text: str) -> set[str]:
    out: set[str] = set()
    for ch in text:
        if ch.isspace():
            continue
        if not ch.isalpha():
            continue
        s = codepoint_script(ord(ch))
        if s:
            out.add(s)
    return out


def gini(values: List[float]) -> float:
    """Gini coefficient on a non-negative distribution (counts)."""
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    n = len(xs)
    s = sum(xs)
    if s == 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, start=1):
        cum += i * x
    return (2 * cum) / (n * s) - (n + 1) / n


def naive_constant_metrics(positive: int, negative: int) -> Dict[str, float]:
    total = positive + negative
    if total == 0 or positive == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    precision = positive / total
    recall = 1.0
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = positive / total
    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
    }


def desc(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p25": 0, "p75": 0, "max": 0, "min": 0, "n": 0}
    s = sorted(values)
    n = len(s)
    return {
        "mean": round(mean(values), 3),
        "median": round(median(values), 3),
        "p25": s[max(0, n // 4)],
        "p75": s[min(n - 1, (3 * n) // 4)],
        "max": max(values),
        "min": min(values),
        "n": n,
    }


COUNTRY_PREFIXES = {
    "us": "United States", "gb": "United Kingdom", "uk": "United Kingdom",
    "eu": "European Union", "un": "United Nations", "au": "Australia",
    "ca": "Canada", "ch": "Switzerland", "de": "Germany", "fr": "France",
    "nl": "Netherlands", "be": "Belgium", "pl": "Poland", "ua": "Ukraine",
    "ru": "Russia", "jp": "Japan", "kr": "South Korea", "cn": "China",
    "in": "India", "za": "South Africa", "br": "Brazil", "ar": "Argentina",
    "mx": "Mexico", "nz": "New Zealand", "il": "Israel", "tr": "Turkey",
    "sg": "Singapore", "hk": "Hong Kong", "ae": "UAE", "qa": "Qatar",
    "kz": "Kazakhstan", "kp": "North Korea", "es": "Spain", "it": "Italy",
    "se": "Sweden", "no": "Norway", "fi": "Finland", "dk": "Denmark",
    "ie": "Ireland", "at": "Austria", "cz": "Czech Republic", "ro": "Romania",
    "bg": "Bulgaria", "by": "Belarus", "lt": "Lithuania", "lv": "Latvia",
    "ee": "Estonia", "md": "Moldova", "rs": "Serbia", "ge": "Georgia",
    "am": "Armenia", "az": "Azerbaijan",
}


def country_from_dataset(name: str) -> str:
    pref = name.split("_", 1)[0].lower() if "_" in name else name[:2].lower()
    return COUNTRY_PREFIXES.get(pref, "Other/International")


def iter_pairs(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def names_of(entity: Dict[str, Any]) -> List[str]:
    props = entity.get("properties", {})
    out: List[str] = []
    for key in ("name", "alias", "firstName", "lastName", "middleName", "weakAlias"):
        for v in props.get(key, []) or []:
            if isinstance(v, str) and v.strip():
                out.append(v)
    return out


def shared_exact_fields(left_props: Dict[str, Any], right_props: Dict[str, Any]) -> int:
    """Count fields where the two entities share at least one identical value."""
    shared = 0
    for k, lv in left_props.items():
        rv = right_props.get(k)
        if not lv or not rv:
            continue
        try:
            ls = {x for x in lv if isinstance(x, (str, int, float))}
            rs = {x for x in rv if isinstance(x, (str, int, float))}
        except TypeError:
            continue
        if ls & rs:
            shared += 1
    return shared


def main() -> None:
    print(f"Streaming {INPUT}...")
    if not INPUT.exists():
        sys.exit(f"input file missing: {INPUT}")

    pair_count = 0
    pos_count = 0
    neg_count = 0

    schema_counter: Counter = Counter()
    dataset_counter: Counter = Counter()
    seen_entities: set[str] = set()
    entity_field_count: List[int] = []
    entity_filled_field_count: List[int] = []
    entity_alias_count: List[int] = []
    entity_dataset_count: List[int] = []
    entity_country_count: List[int] = []
    entity_has_alias = 0

    # Script tracking
    pair_has_nonlatin = 0
    pair_cross_script = 0  # left script != right script (any nonlatin involved)
    script_pair_counter: Counter = Counter()
    entity_script_counter: Counter = Counter()  # primary script per entity
    pair_script_combo: Counter = Counter()  # combo of {left,right} primary scripts

    # Cross-source
    cross_source_total = 0
    cross_source_pos = 0
    cross_source_neg = 0
    same_source_pos = 0
    same_source_neg = 0
    cross_country_pos = 0

    # Pair-difficulty
    shared_fields_pos: List[int] = []
    shared_fields_neg: List[int] = []

    # Pair-level schema combinations
    pair_schema_combo: Counter = Counter()
    pair_schema_combo_pos: Counter = Counter()
    pair_schema_combo_neg: Counter = Counter()

    # Temporal hold-out
    cutoff = CUTOFF_DATE
    post_cutoff_pairs = 0
    post_cutoff_pos = 0
    post_cutoff_neg = 0

    # Caption-level script combinations (use caption only — fewer aliases, more representative)
    pair_caption_scripts_diff = 0

    # Entity first_seen / last_seen tracking
    first_seen_year_counter: Counter = Counter()
    last_seen_year_counter: Counter = Counter()
    min_first_seen: str | None = None
    max_first_seen: str | None = None
    min_last_seen: str | None = None
    max_last_seen: str | None = None

    for i, pair in enumerate(iter_pairs(INPUT)):
        pair_count += 1
        judgement = pair.get("judgement", "")
        if judgement == "positive":
            pos_count += 1
        elif judgement == "negative":
            neg_count += 1

        left = pair.get("left", {})
        right = pair.get("right", {})

        l_datasets = left.get("datasets") or []
        r_datasets = right.get("datasets") or []
        l_set = set(l_datasets)
        r_set = set(r_datasets)
        cross_src = bool(l_set and r_set and not (l_set & r_set))
        if cross_src:
            cross_source_total += 1
            if judgement == "positive":
                cross_source_pos += 1
            elif judgement == "negative":
                cross_source_neg += 1
        else:
            if judgement == "positive":
                same_source_pos += 1
            elif judgement == "negative":
                same_source_neg += 1

        # Country crossing on positive pairs
        l_countries = set((left.get("properties") or {}).get("country") or [])
        r_countries = set((right.get("properties") or {}).get("country") or [])
        if judgement == "positive" and l_countries and r_countries and not (l_countries & r_countries):
            cross_country_pos += 1

        # Pair-level schema combo
        l_schema = left.get("schema", "Unknown")
        r_schema = right.get("schema", "Unknown")
        combo_schema = "|".join(sorted([l_schema, r_schema]))
        pair_schema_combo[combo_schema] += 1
        if judgement == "positive":
            pair_schema_combo_pos[combo_schema] += 1
        elif judgement == "negative":
            pair_schema_combo_neg[combo_schema] += 1

        # Pair-difficulty: number of fields with at least one shared value
        sf = shared_exact_fields(
            left.get("properties") or {},
            right.get("properties") or {},
        )
        if judgement == "positive":
            shared_fields_pos.append(sf)
        elif judgement == "negative":
            shared_fields_neg.append(sf)

        # Caption-script comparison (cheap, robust)
        l_cap = left.get("caption", "") or ""
        r_cap = right.get("caption", "") or ""
        l_scripts = detect_scripts(l_cap)
        r_scripts = detect_scripts(r_cap)
        if l_scripts != r_scripts and l_scripts and r_scripts:
            pair_caption_scripts_diff += 1
        all_scripts = l_scripts | r_scripts
        if all_scripts and all_scripts != {"Latin"}:
            pair_has_nonlatin += 1
        for sc in all_scripts:
            script_pair_counter[sc] += 1
        # primary script per pair (deterministic combo string)
        l_primary = next(iter(sorted(l_scripts)), "Unknown")
        r_primary = next(iter(sorted(r_scripts)), "Unknown")
        combo = "|".join(sorted({l_primary, r_primary}))
        pair_script_combo[combo] += 1

        # Entity-level (count each unique entity once)
        for ent in (left, right):
            eid = ent.get("id")
            if not eid or eid in seen_entities:
                continue
            seen_entities.add(eid)

            schema_counter[ent.get("schema", "Unknown")] += 1
            for ds in ent.get("datasets") or []:
                dataset_counter[ds] += 1

            props = ent.get("properties") or {}
            field_keys = list(props.keys())
            entity_field_count.append(len(field_keys))
            filled = sum(1 for k, v in props.items() if v)
            entity_filled_field_count.append(filled)
            aliases = (props.get("name") or []) + (props.get("alias") or []) + (props.get("weakAlias") or [])
            entity_alias_count.append(len(aliases))
            if (props.get("alias") or []) or (props.get("weakAlias") or []):
                entity_has_alias += 1
            entity_dataset_count.append(len(ent.get("datasets") or []))
            entity_country_count.append(len(props.get("country") or []))

            # primary script of entity caption
            cap = ent.get("caption", "") or ""
            scripts = detect_scripts(cap)
            primary = next(iter(sorted(scripts)), "Unknown")
            entity_script_counter[primary] += 1

            # Entity-level first_seen / last_seen
            ent_fs = (ent.get("first_seen") or "")[:10]
            ent_ls = (ent.get("last_seen") or "")[:10]
            if ent_fs:
                first_seen_year_counter[ent_fs[:4]] += 1
                if min_first_seen is None or ent_fs < min_first_seen:
                    min_first_seen = ent_fs
                if max_first_seen is None or ent_fs > max_first_seen:
                    max_first_seen = ent_fs
            if ent_ls:
                last_seen_year_counter[ent_ls[:4]] += 1
                if min_last_seen is None or ent_ls < min_last_seen:
                    min_last_seen = ent_ls
                if max_last_seen is None or ent_ls > max_last_seen:
                    max_last_seen = ent_ls

        # Temporal hold-out: pairs whose BOTH entities have first_seen >= cutoff
        l_fs = (left.get("first_seen") or "")[:10]
        r_fs = (right.get("first_seen") or "")[:10]
        if l_fs and r_fs and l_fs >= cutoff and r_fs >= cutoff:
            post_cutoff_pairs += 1
            if judgement == "positive":
                post_cutoff_pos += 1
            elif judgement == "negative":
                post_cutoff_neg += 1

        if (i + 1) % 100000 == 0:
            print(f"  ...processed {i + 1:,} pairs (entities so far: {len(seen_entities):,})")

    print(f"Done: {pair_count:,} pairs, {len(seen_entities):,} unique entities")

    # ---- Build report ----
    n_unique_entities = len(seen_entities)
    n_datasets = len(dataset_counter)
    counts = list(dataset_counter.values())
    g_dataset = gini(counts)

    schema_pct = {
        s: {"count": c, "pct": round(c / n_unique_entities, 4)}
        for s, c in schema_counter.most_common()
    }
    top_n = 10
    top_sources = dataset_counter.most_common(top_n)
    head_pct = round(sum(c for _, c in top_sources) / sum(counts), 4)

    # Country tally for sources
    sources_by_country: Dict[str, int] = defaultdict(int)
    for ds in dataset_counter:
        sources_by_country[country_from_dataset(ds)] += 1
    sources_by_country_sorted = dict(
        sorted(sources_by_country.items(), key=lambda x: -x[1])
    )

    # Per-script tallies on entities (primary script of caption)
    primary_script_pct = {
        s: {"count": c, "pct": round(c / n_unique_entities, 4)}
        for s, c in entity_script_counter.most_common()
    }

    naive = naive_constant_metrics(pos_count, neg_count)

    results = {
        "input_file": str(INPUT),
        "cutoff_date": cutoff,
        "main": {
            "total_pairs": pair_count,
            "positive": pos_count,
            "negative": neg_count,
            "positive_ratio": round(pos_count / pair_count, 4) if pair_count else 0,
            "unique_entities": n_unique_entities,
            "num_source_datasets": n_datasets,
            "naive_constant_positive_baseline": naive,
        },
        "source_distribution": {
            "num_sources": n_datasets,
            "gini_coefficient_over_entity_counts": round(g_dataset, 4),
            "top10_share_of_entity_appearances": head_pct,
            "top20_sources": dataset_counter.most_common(20),
            "num_countries_represented": len([c for c in sources_by_country if c != "Other/International"]),
            "sources_by_country": sources_by_country_sorted,
        },
        "schema_distribution": schema_pct,
        "entity_descriptives": {
            "fields_per_entity": desc(entity_field_count),
            "filled_fields_per_entity": desc(entity_filled_field_count),
            "aliases_per_entity": desc(entity_alias_count),
            "datasets_per_entity": desc(entity_dataset_count),
            "countries_per_entity": desc(entity_country_count),
            "fraction_with_alias_or_weakAlias": round(entity_has_alias / n_unique_entities, 4)
            if n_unique_entities
            else 0,
        },
        "scripts": {
            "pair_caption_scripts_differ": pair_caption_scripts_diff,
            "pair_caption_scripts_differ_pct": round(pair_caption_scripts_diff / pair_count, 4)
            if pair_count
            else 0,
            "pairs_with_any_nonlatin": pair_has_nonlatin,
            "pairs_with_any_nonlatin_pct": round(pair_has_nonlatin / pair_count, 4)
            if pair_count
            else 0,
            "script_appearances_in_pairs": dict(script_pair_counter.most_common()),
            "primary_script_by_entity": primary_script_pct,
            "top_pair_script_combos": dict(pair_script_combo.most_common(15)),
        },
        "cross_source": {
            "cross_source_pairs": cross_source_total,
            "cross_source_pct": round(cross_source_total / pair_count, 4) if pair_count else 0,
            "cross_source_positive": cross_source_pos,
            "cross_source_negative": cross_source_neg,
            "same_source_positive": same_source_pos,
            "same_source_negative": same_source_neg,
            "cross_source_positive_ratio_within_positives": round(cross_source_pos / pos_count, 4)
            if pos_count
            else 0,
            "cross_country_positive_pairs": cross_country_pos,
        },
        "pair_difficulty": {
            "shared_exact_fields_positive": desc(shared_fields_pos),
            "shared_exact_fields_negative": desc(shared_fields_neg),
        },
        "pair_schema_combinations": {
            "all": dict(pair_schema_combo.most_common()),
            "positive": dict(pair_schema_combo_pos.most_common()),
            "negative": dict(pair_schema_combo_neg.most_common()),
        },
        "temporal_coverage": {
            "first_seen_min": min_first_seen,
            "first_seen_max": max_first_seen,
            "last_seen_min": min_last_seen,
            "last_seen_max": max_last_seen,
            "first_seen_by_year": dict(sorted(first_seen_year_counter.items())),
            "last_seen_by_year": dict(sorted(last_seen_year_counter.items())),
        },
        "temporal_holdout": {
            "cutoff_date": cutoff,
            "pairs_both_first_seen_after_cutoff": post_cutoff_pairs,
            "post_cutoff_positive": post_cutoff_pos,
            "post_cutoff_negative": post_cutoff_neg,
            "post_cutoff_pct_of_total": round(post_cutoff_pairs / pair_count, 4)
            if pair_count
            else 0,
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {OUTPUT}")

    # Console summary
    m = results["main"]
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Pairs: {m['total_pairs']:,}  (pos {m['positive']:,} / neg {m['negative']:,})")
    print(f"Unique entities: {m['unique_entities']:,}")
    print(f"Sources: {m['num_source_datasets']}  Gini={results['source_distribution']['gini_coefficient_over_entity_counts']:.3f}  top10 share={results['source_distribution']['top10_share_of_entity_appearances']:.1%}")
    print(f"Naive 'always positive' baseline: F1={naive['f1']:.4f}  acc={naive['accuracy']:.4f}")
    print(f"Cross-source pairs: {results['cross_source']['cross_source_pairs']:,} ({results['cross_source']['cross_source_pct']:.1%})")
    print(f"  Cross-source positives within all positives: {results['cross_source']['cross_source_positive_ratio_within_positives']:.1%}")
    print(f"Pair captions in different scripts: {results['scripts']['pair_caption_scripts_differ']:,} ({results['scripts']['pair_caption_scripts_differ_pct']:.1%})")
    print(f"Pairs with any non-Latin script: {results['scripts']['pairs_with_any_nonlatin']:,} ({results['scripts']['pairs_with_any_nonlatin_pct']:.1%})")
    print(f"Primary script per entity (top 5): {list(results['scripts']['primary_script_by_entity'].items())[:5]}")
    print(f"Shared-fields(pos) median={results['pair_difficulty']['shared_exact_fields_positive']['median']}  (neg) median={results['pair_difficulty']['shared_exact_fields_negative']['median']}")
    h = results["temporal_holdout"]
    print(f"Post-{h['cutoff_date']} pairs (both first_seen >=): {h['pairs_both_first_seen_after_cutoff']:,} ({h['post_cutoff_pct_of_total']:.2%}) — pos {h['post_cutoff_positive']:,} / neg {h['post_cutoff_negative']:,}")


if __name__ == "__main__":
    main()
