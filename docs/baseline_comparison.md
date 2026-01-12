# Baseline Entity Resolution Methods: Comparison

This document compares two baseline approaches for pairwise entity resolution on OSINT sanctions data.

## Results Summary

| Method | Sample Size | Threshold | Accuracy | Precision | Recall | F1 Score |
|--------|-------------|-----------|----------|-----------|--------|----------|
| Simple Fuzzy | 1,000 | 0.60 | 0.9240 | 0.9543 | 0.9587 | 0.9565 |
| Simple Fuzzy | 10,000 | 0.45 | 0.9600 | 0.9798 | 0.9786 | **0.9792** |
| Nomenklatura RegressionV1 | 1,000 | 0.60 | 0.9460 | 0.9534 | 0.9862 | 0.9696 |
| Nomenklatura RegressionV1 | 10,000 | 0.15 | **0.9736** | 0.9734 | **0.9998** | **0.9864** |

### Key Observations

1. **Nomenklatura achieves higher F1** on both sample sizes (+1.3% on N=1k, +0.7% on N=10k)
2. **Nomenklatura has near-perfect recall** (99.98%) at the cost of slightly lower precision
3. **Simple Fuzzy is more balanced** between precision and recall
4. **Threshold varies significantly** — Nomenklatura drops to 0.15 on larger data, suggesting its scores are more spread out

---

## Data Schema

Each entity pair in the dataset follows the [FollowTheMoney](https://followthemoney.tech/) schema:

```json
{
  "judgement": "positive" | "negative",
  "left": { /* Entity */ },
  "right": { /* Entity */ }
}
```

### Entity Structure

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., `eu-fsf-eu-3298-47`) |
| `caption` | string | Display name |
| `schema` | string | Entity type (e.g., `Person`, `Company`) |
| `datasets` | array | Source datasets (e.g., `eu_fsf`, `un_sc_sanctions`) |
| `properties` | object | Key-value attributes (see below) |

### Person Properties (used for matching)

| Property | Example | Used by Simple Fuzzy | Used by RegressionV1 |
|----------|---------|---------------------|---------------------|
| `name` | `["John Smith", "Smith, John"]` | ✅ Primary (70%) | ✅ Multiple features |
| `firstName` | `["John"]` | ❌ | ✅ `first_name_match` |
| `lastName` | `["Smith"]` | ❌ | ✅ `family_name_match` |
| `birthDate` | `["1969-06-16", "1969"]` | ✅ (+10-20%) | ✅ `dob_matches`, `dob_year_matches` |
| `country` | `["iq", "eg"]` | ✅ (+10%) | ✅ `country_mismatch` |
| `nationality` | `["iq"]` | ✅ (+10%) | ✅ `country_mismatch` |
| `gender` | `["male"]` | ❌ | ✅ `gender_mismatch` |
| `idNumber` | `["1729765"]` | ❌ | ✅ `identifier_match` |
| `phone` | `["+1234567890"]` | ❌ | ✅ `phone_match` |
| `email` | `["john@example.com"]` | ❌ | ✅ `email_match` |
| `address` | `["123 Main St"]` | ❌ | ✅ `address_match` |

---

## Method 1: Simple Fuzzy Match

**File:** [`scripts/baselines/simple_fuzzy.py`](../scripts/baselines/simple_fuzzy.py)

### Scoring Formula

```
score = (name_similarity × 0.7) + country_bonus + date_bonus
```

Capped at 1.0.

### Features

#### 1. Name Similarity (max 0.70)

**Algorithm:** Token-sorted sequence matching

```python
# Tokenize, sort, then compare
t1 = " ".join(sorted("John Smith".lower().split()))   # → "john smith"
t2 = " ".join(sorted("Smith, John".lower().split()))  # → "john, smith"
score = difflib.SequenceMatcher(None, t1, t2).ratio() # → 0.92
```

**Purpose:** Handles name ordering variations ("First Last" vs "Last, First")

**Properties used:** `properties.name[]`

#### 2. Country/Nationality Match (+0.10)

**Algorithm:** Set intersection

```python
countries_left = properties.country + properties.nationality
countries_right = properties.country + properties.nationality
bonus = 0.1 if intersection(countries_left, countries_right) else 0
```

**Purpose:** Boosts score when entities share geographic attribution

**Properties used:** `properties.country[]`, `properties.nationality[]`

#### 3. Birth Date Match (+0.10 or +0.20)

**Algorithm:** 
- Exact date match → +0.20
- Year-only match (first 4 digits) → +0.10

```python
if exact_match(birthDate_left, birthDate_right):
    bonus = 0.2
elif year_match(birthDate_left, birthDate_right):
    bonus = 0.1
```

**Properties used:** `properties.birthDate[]`

### Strengths
- No external dependencies (uses Python stdlib `difflib`)
- Fully interpretable scoring
- Fast execution
- Handles most common name variations

### Limitations
- No negative signals (can't penalize mismatches)
- Ignores structured name components (`firstName`, `lastName`)
- Ignores identifiers, emails, phones, addresses

---

## Method 2: Nomenklatura RegressionV1

**File:** [`scripts/baselines/nomenklatura_v1.py`](../scripts/baselines/nomenklatura_v1.py)

**Library:** [OpenSanctions Nomenklatura](https://github.com/opensanctions/nomenklatura)

### Scoring Method

**Algorithm:** Logistic regression trained on OpenSanctions manual deduplication decisions

```python
score = logistic_regression.predict_proba(features)[1]  # probability of match
```

### Features (18 total)

#### Name Features (6)

| Feature | Description | Signal |
|---------|-------------|--------|
| `name_match` | Exact match in any name pair | Strong positive |
| `name_token_overlap` | Proportion of shared tokens | Positive |
| `name_levenshtein` | Edit distance as fraction of length | Positive |
| `name_numbers` | Numeric discrepancy in names (e.g., "Company 1" vs "Company 2") | Negative |
| `first_name_match` | Match on `firstName` property | Positive |
| `family_name_match` | Match on `lastName` property | Positive |

#### Date Features (3)

| Feature | Description | Signal |
|---------|-------------|--------|
| `dob_matches` | Exact birth date match | Strong positive |
| `dob_year_matches` | Birth year match | Positive |
| `dob_year_disjoint` | Birth years don't overlap | **Negative** |

#### Identifier Features (3)

| Feature | Description | Signal |
|---------|-------------|--------|
| `identifier_match` | Match on passport, national ID, etc. | Strong positive |
| `org_identifier_match` | Match on registration/tax numbers | Strong positive |
| `phone_match` | Phone number match | Strong positive |

#### Contact & Location Features (4)

| Feature | Description | Signal |
|---------|-------------|--------|
| `email_match` | Email address match | Strong positive |
| `address_match` | Address text overlap | Positive |
| `address_numbers` | Address number discrepancy | Negative |
| `birth_place` | Birth place match | Positive |

#### Demographic Features (2)

| Feature | Description | Signal |
|---------|-------------|--------|
| `gender_mismatch` | Gender fields don't match | **Negative** |
| `country_mismatch` | Country/nationality mismatch | **Negative** |

### Key Advantages Over Simple Fuzzy

1. **Learned weights** — Coefficients derived from real deduplication data
2. **Negative signals** — Gender mismatch or birth year disjoint actively reduce score
3. **Structured name matching** — Uses `firstName`, `lastName` separately
4. **Identifier matching** — Can use strong ID signals when available
5. **More features** — 18 vs 3

### Why Near-Perfect Recall?

The threshold of 0.15 on the 10k sample suggests the model is very conservative about rejecting matches. This is intentional for sanctions matching where **false negatives are costly** (missing a match could mean failing to flag a sanctioned entity).

---

## Threshold Optimization

Both methods use the same tuning approach:

1. Split data 50/50 into dev and test sets
2. On dev set: test thresholds from 0.05 to 0.95
3. Select threshold that maximizes F1 score
4. Evaluate on held-out test set

```python
def optimize_threshold(scores, labels):
    best_f1, best_thresh = 0.0, 0.5
    for thresh in [i/100.0 for i in range(5, 100, 5)]:
        predictions = [1 if s >= thresh else 0 for s in scores]
        f1 = compute_f1(labels, predictions)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_thresh
```

---

## When to Use Each Method

| Use Case | Recommended |
|----------|-------------|
| Quick prototyping, no dependencies | Simple Fuzzy |
| Production sanctions screening | Nomenklatura |
| Maximum recall (minimize false negatives) | Nomenklatura |
| Balanced precision/recall | Simple Fuzzy |
| Entities with rich identifier data | Nomenklatura |
| Name-only matching | Either (similar performance) |

---

## Running the Baselines

```bash
# Simple Fuzzy
python scripts/baselines/simple_fuzzy.py --input data/samples/sample_1000.json

# Nomenklatura RegressionV1
python scripts/baselines/nomenklatura_v1.py --input data/samples/sample_1000.json
```

### Requirements

- **Simple Fuzzy:** Python stdlib only
- **Nomenklatura:** `pip install "nomenklatura<3.10" followthemoney` (for Python 3.10)
