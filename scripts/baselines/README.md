# Baselines

This directory contains baseline implementations for the OSINT Entity Resolution Benchmark.

## 1. Simple Fuzzy Match (`simple_fuzzy.py`)

A deterministic, rule-based matcher that uses weighted heuristics.

### Features Used
1.  **Name Similarity:** Uses Python's `difflib` to calculate a token-sorted ratio (handling "Last, First" vs "First Last").
    *   *Weight:* 0.7 (Max)
2.  **Country Match:** Checks for intersection in `country` or `nationality` fields.
    *   *Weight:* +0.1
3.  **Date Match:** Checks for exact `birthDate` match or `birthYear` match.
    *   *Weight:* +0.2 (Exact) or +0.1 (Year only)

### Usage
```bash
python scripts/baselines/simple_fuzzy.py --input ../../data/samples/sample_1000.json
```

### Initial Results (Sample N=1000)
*   **Threshold:** 0.60
*   **F1 Score:** ~0.9565
*   **Accuracy:** ~0.9240

## 2. Nomenklatura Matcher (`nomenklatura_v1.py`)

Uses the `RegressionV1` algorithm from the [OpenSanctions Nomenklatura](https://github.com/opensanctions/nomenklatura) library. This is the current production standard for the project.

### Methodology
*   Uses `followthemoney` entities and `nomenklatura`'s regression model.
*   The model weights various features (names, dates, identifiers, etc.) to produce a score.
*   This script tunes the decision threshold on a dev set.

### Usage
```bash
# Requires dependencies installed (see requirements.txt + extra)
python scripts/baselines/nomenklatura_v1.py --input ../../data/samples/sample_1000.json
```

### Initial Results (Sample N=1000)
*   **Threshold:** 0.60
*   **F1 Score:** ~0.9696
*   **Accuracy:** ~0.9460