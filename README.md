# OSINT Entity Resolution Benchmark

A minimal benchmark for evaluating entity resolution approaches on OSINT data.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a small data sample
python scripts/load_data.py

# 4. Explore the data
jupyter notebook notebooks/explore_data.ipynb
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
│   ├── raw/                   # Large data files (gitignored)
│   │   └── pairs-20251209.json.gz
│   └── samples/               # Small samples (tracked in git)
├── requirements.txt           # Python dependencies
├── notebooks/                 # Jupyter notebooks for exploration
└── scripts/                   # Data loading and evaluation utilities
```

## Usage

### Loading Data

```python
from scripts.load_data import load_pairs, create_sample

# Load full dataset (memory-efficient iterator)
for pair in load_pairs("data/raw/pairs-20251209.json.gz"):
    print(pair["judgement"], pair["left"], pair["right"])

# Create small sample for testing
create_sample("data/raw/pairs-20251209.json.gz", "data/samples/sample_100.json", n=100)
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
