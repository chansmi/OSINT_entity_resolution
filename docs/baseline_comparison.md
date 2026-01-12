# Baseline Entity Resolution: Experimental Comparison

This document compares two pairwise entity resolution baselines on OSINT sanctions data, with analysis of how sampling strategy affects results.

## Key Finding: Sampling Bias

> [!WARNING]
> **Unshuffled samples are severely biased and inflate performance metrics.**

The source data file has a non-random ordering. Taking the "first N" records produces samples with artificially high positive rates:

| Sample | Positive Rate | Negative Rate | Representative? |
|--------|---------------|---------------|-----------------|
| **Full dataset** | 76.9% | 23.1% | ✅ Ground truth |
| 1k Unshuffled | 85.5% | 14.5% | ❌ Biased |
| 10k Unshuffled | 95.9% | 4.1% | ❌ **Severely biased** |
| 1k Shuffled | 76.6% | 23.4% | ✅ Representative |
| 10k Shuffled | 76.2% | 23.8% | ✅ Representative |

The unshuffled samples contain mostly "easy" positive matches, making resolution trivially accurate.

---

## Results: N = 1,000

| Method | Shuffled | Threshold | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|----------|-----------|--------|-----|
| Simple Fuzzy | ❌ No | 0.60 | 0.9240 | 0.9543 | 0.9587 | 0.9565 |
| Simple Fuzzy | ✅ Yes | 0.65 | 0.5940 | 0.9202 | 0.5131 | **0.6588** |
| Nomenklatura | ❌ No | 0.60 | 0.9460 | 0.9534 | 0.9862 | 0.9696 |
| Nomenklatura | ✅ Yes | 0.20 | 0.8640 | 0.8568 | 0.9869 | **0.9173** |

**On representative (shuffled) data:**
- Nomenklatura achieves **0.92 F1** vs Simple Fuzzy's **0.66 F1** — a 26 percentage point gap
- Simple Fuzzy has high precision (0.92) but low recall (0.51) — it misses half the true matches
- Nomenklatura maintains near-perfect recall (0.99) with reasonable precision (0.86)

---

## Results: N = 10,000

| Method | Shuffled | Threshold | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|----------|-----------|--------|-----|
| Simple Fuzzy | ❌ No | 0.45 | 0.9600 | 0.9798 | 0.9786 | 0.9792 |
| Simple Fuzzy | ✅ Yes | 0.65 | 0.5908 | 0.9378 | 0.5035 | **0.6552** |
| Nomenklatura | ❌ No | 0.15 | 0.9736 | 0.9734 | 0.9998 | 0.9864 |
| Nomenklatura | ✅ Yes | 0.20 | 0.8666 | 0.8566 | 0.9935 | **0.9200** |

**On representative (shuffled) data:**
- Results are consistent with N=1,000 — sampling strategy matters more than sample size
- Nomenklatura: **0.92 F1** (stable across sample sizes)
- Simple Fuzzy: **0.66 F1** (stable across sample sizes)

---

## Why Does Shuffling Hurt Performance?

**It doesn't hurt performance — it reveals the true performance.**

### The Unshuffled Samples Are "Easy"

1. **Fewer hard negatives**: With only 4-14% negatives, most pairs are matches. A model that predicts "positive" for everything would score >85% accuracy.

2. **Correlated ordering**: The source file groups similar entities together. The first N records may be from the same dataset/source, with more consistent formatting and easier name matching.

3. **Threshold overfitting**: With biased dev sets, the tuned threshold works for that distribution but fails on representative data.

### The Shuffled Samples Reveal True Difficulty

The dataset contains many **hard negatives** — entity pairs that share names or attributes but refer to different people. With the true 23% negative rate:

- Simple Fuzzy's 3-feature approach can't distinguish these cases
- Nomenklatura's 18 features (including negative signals like `gender_mismatch`, `dob_year_disjoint`) correctly reject them

---

## Method Comparison

### Simple Fuzzy
**Features (3):**
- Name similarity (token-sorted, weighted 0.7)
- Country/nationality match (+0.1)  
- Birth date/year match (+0.1 or +0.2)

**Limitation:** No negative signals — can only add evidence, never subtract.

### Nomenklatura RegressionV1
**Features (18):** Includes all of Simple Fuzzy's plus:
- First/last name matching separately
- Phone, email, identifier matching
- **Negative signals**: gender mismatch, birth year disjoint, country mismatch

**Advantage:** Can actively penalize contradictory evidence.

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Dev/Test Split | 50/50 |
| Random Seed | 42 |
| Threshold Range | 0.05 - 0.95 (step 0.05) |
| Optimization Metric | F1 on dev set |

### Commands Used

```bash
# Create samples
python scripts/load_data.py --n 1000 --output data/samples/sample_1000_unshuffled.json
python scripts/load_data.py --n 1000 --output data/samples/sample_1000_shuffled.json --shuffle

# Run baselines
python scripts/baselines/simple_fuzzy.py --input data/samples/sample_1000_shuffled.json
python scripts/baselines/nomenklatura_v1.py --input data/samples/sample_1000_shuffled.json
```

---

## Recommendations

1. **Always use shuffled samples** for evaluation — unshuffled results are not representative
2. **Use Nomenklatura for production** — 0.92 F1 vs 0.66 F1 on real data
3. **Sample size is less important** than sampling method for these baselines
4. **Consider precision/recall tradeoffs** — Nomenklatura favors recall (missing fewer matches at cost of some false positives)
