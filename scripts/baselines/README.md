# Baselines

This directory contains baseline implementations for the OSINT Entity Resolution Benchmark.

All baselines are evaluated on a stratified, shuffled sample (76.9% positive, matching the full dataset distribution) using an 80/20 dev/test split with fixed seed (42) for reproducibility.

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
python scripts/baselines/simple_fuzzy.py --input data/samples/sample_1000.json
```

## 2. Nomenklatura Matcher (`nomenklatura_v1.py`)

Uses the `RegressionV1` algorithm from the [OpenSanctions Nomenklatura](https://github.com/opensanctions/nomenklatura) library. This is the current production standard for the project.

### Methodology
*   Uses `followthemoney` entities and `nomenklatura`'s regression model.
*   The model weights various features (names, dates, identifiers, etc.) to produce a score.
*   This script tunes the decision threshold on a dev set.

### Usage
```bash
python scripts/baselines/nomenklatura_v1.py --input data/samples/sample_1000.json
```

## 3. LLM Zero-Shot (`llm_zeroshot.py`)

Uses GPT-5 models to classify entity pairs with a conflict-focused prompt. Supports parallel execution for ~30x speedup.

### Methodology
*   Frames entity resolution as **contradiction detection** rather than similarity matching
*   Default assumption: entities are the same unless explicit conflicts are found
*   Supports binary mode (positive/negative) and ternary mode (positive/negative/uncertain)
*   Uses structured JSON output for reliable parsing

### Key Insight: Conflict-Focused Prompting
The prompt asks "Is there contradictory evidence?" rather than "Are these the same?". This simple reframing reduced false negatives by 97% (195 → 6) by:
- Treating name variations and missing fields as expected, not suspicious
- Only flagging negative when explicit conflicts exist (different dates, conflicting IDs)

### Usage
```bash
# Binary mode (recommended)
python scripts/baselines/llm_zeroshot.py --input data/samples/sample_1000.json --parallel 30

# Ternary mode (uncertain cases flagged for human review)
python scripts/baselines/llm_zeroshot.py --input data/samples/sample_1000.json --parallel 30 --ternary

# With options
python scripts/baselines/llm_zeroshot.py --input data/samples/sample_1000.json --parallel 30 --model gpt-5.2 --reasoning high
```

Requires `OPENAI_API_KEY` in `.env` file.

## Results (N=1000 stratified sample, 800 test pairs)

| Method | F1 Score | Precision | Recall | FN | FP |
|--------|----------|-----------|--------|----|----|
| Nomenklatura RegressionV1 | 90.61% | 82.84% | 100% | 0 | 127 |
| **LLM Zero-Shot (GPT-5-nano)** | **93.10%** | 87.84% | 99.02% | 6 | 84 |
| LLM Ternary Mode | 99.53%* | 99.38% | 99.69% | 1 | 2 |

*Ternary mode: 62.8% coverage (502/800 automated), uncertain cases flagged for human review.

### Complementary Failure Modes

The methods have different failure modes, making them suitable for different use cases:

| Behavior | Nomenklatura | LLM Binary |
|----------|--------------|------------|
| False Positives | 127 | 84 |
| False Negatives | 0 | 6 |
| Strategy | Never miss a match | Balanced |

**Deployment Guidance:**
- **High-recall workflow:** Use Nomenklatura (0% FN, 16% FP)
- **Balanced workflow:** Use LLM Binary (0.7% FN, 10% FP)
- **Human-in-the-loop:** Use LLM Ternary (99.4% accuracy on 63% of cases, route uncertain to human)