# OSINT Entity Resolution Benchmark

A minimal benchmark for evaluating entity resolution approaches on OSINT data.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download and Prepare Data

**Download the raw data file** and place it in `data/raw/`:

```bash
# Download pairs-20251209.json.gz (391 MB compressed)
# Place it in: data/raw/pairs-20251209.json.gz
```

The dataset contains **755,540 entity pairs** with the following distribution:
- **Positive matches**: 581,149 (76.9%)
- **Negative matches**: 174,391 (23.1%)

### 3. Choose Your Data Loading Strategy

#### Option A: Use Sample Data (Recommended for Quick Start)

Create a small sample (1,000 entries) for rapid exploration:

```bash
# Creates data/samples/sample_1000.json
python scripts/load_data.py --n 1000 --output data/samples/sample_1000.json
```

**Memory usage**: ~4 MB  
**Load time**: <1 second

#### Option B: Use Full Dataset

Load all 755k entries into memory with caching for faster subsequent loads:

```bash
# First run: loads from .gz and creates cache (~43s)
# Subsequent runs: loads from cache (~5-10s)
python scripts/cache_full_dataset.py
```

This creates `data/raw/pairs_full.json` (2.1 GB, gitignored) for faster loading.

**Memory usage**: ~10 GB  
**Load time**: 
- First run (from .gz): ~30-60 seconds
- Cached runs: ~5-10 seconds

### 4. Explore the Data

```bash
# Open Jupyter notebook
jupyter notebook notebooks/explore_data.ipynb

# Toggle between sample and full dataset in the notebook
# Set USE_SAMPLE = True for sample (1k entries)
# Set USE_SAMPLE = False for full dataset (755k entries)
```

## Data Format

The dataset consists of entity pairs with ground truth labels:

```json
{
  "judgement": "positive",  // or "negative"
  "left": { ... },          // Entity data
  "right": { ... }          // Entity data
}
```

- **positive**: Entities refer to the same real-world entity
- **negative**: Entities refer to different real-world entities

## Repository Structure

```
├── data/
│   ├── raw/                          # Large data files (gitignored)
│   │   ├── pairs-20251209.json.gz    # Original compressed dataset (391 MB)
│   │   └── pairs_full.json           # Cached uncompressed dataset (2.1 GB, created by cache_full_dataset.py)
│   └── samples/                      # Small samples (tracked in git)
│       └── sample_1000.json          # Sample of 1,000 pairs for quick testing
├── notebooks/
│   └── explore_data.ipynb            # EDA notebook with sample/full dataset toggle
├── scripts/
│   ├── load_data.py                  # Data loading utilities (library)
│   ├── cache_full_dataset.py         # Script to create full dataset cache
│   └── evaluate.py                   # Evaluation metrics
└── requirements.txt                  # Python dependencies
```

## Usage

### Loading Data

```python
from scripts.load_data import load_pairs, create_sample, load_sample, load_full_dataset

# Option 1: Memory-efficient iterator (doesn't load everything at once)
for pair in load_pairs("data/raw/pairs-20251209.json.gz"):
    print(pair["judgement"], pair["left"], pair["right"])

# Option 2: Create and load a small sample
create_sample("data/raw/pairs-20251209.json.gz", "data/samples/sample_100.json", n=100)
sample_data = load_sample("data/samples/sample_100.json")

# Option 3: Load full dataset into memory with caching (~10 GB)
full_data = load_full_dataset(
    input_path="data/raw/pairs-20251209.json.gz",
    cache_path="data/raw/pairs_full.json",
    use_cache=True  # Uses cache if available, creates it if not
)
print(f"Loaded {len(full_data):,} pairs")
```

### Evaluation

```python
from scripts.evaluate import evaluate

results = evaluate(
    ground_truth=["positive", "negative", "positive"],
    predictions=["positive", "positive", "positive"]
)
print(results)
# Output: {'accuracy': 0.67, 'precision': 0.67, 'recall': 1.0, 'f1': 0.80}
```

## Contributing

This is a collaborative project. Keep things simple:
- Add notebooks for experiments in `notebooks/`
- Add utility scripts to `scripts/`
- Document your approach in notebook markdown cells
- Update this README when adding major features

## License

See [LICENSE](LICENSE) for details.
