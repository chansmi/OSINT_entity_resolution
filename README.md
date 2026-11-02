# OpenSanctions Pairs

Code and evaluation artifacts for **OpenSanctions Pairs: A Large-Scale Dataset for Pairwise Entity Matching**.

The task: given two entity records (persons, organisations, vessels, ...) drawn from sanctions, PEP and watchlist sources, decide whether they refer to the same real-world entity. The corpus has 755,540 labelled pairs from [OpenSanctions](https://www.opensanctions.org/docs/opensource/pairs/).

- **Dataset:** [huggingface.co/datasets/sanctions-er-anon/opensanctions_pairs](https://huggingface.co/datasets/sanctions-er-anon/opensanctions_pairs) (CC BY-NC 4.0)
- **Croissant metadata (with RAI fields):** [`data/croissant/croissant.json`](data/croissant/croissant.json), rebuilt from the Hub with `data/croissant/finalize_croissant.py`
- **Frozen evaluation splits:** [`data/splits/`](data/splits/), checksummed in `manifest.json`
- **Every model prediction behind the reported numbers:** [`results/predictions/`](results/predictions/)

## Results

All models on the same analyst-judged split (N=3000). F1 with 95% bootstrap CI, McNemar test against the best model. Regenerate with `python eval/make_tables.py`.

| Model | F1 (95% CI) | Precision | Recall | vs best (McNemar) |
|---|---|---|---|---|
| gpt-5.2-pro | 98.05 [97.58, 98.47] | 97.08 | 99.03 | best |
| claude-sonnet-4.5 | 97.83 [97.36, 98.29] | 98.59 | 97.08 | p=0.505 (n.s.) |
| claude-opus-4.5 | 97.75 [97.26, 98.22] | 98.64 | 96.87 | p=0.332 (n.s.) |
| gpt-4o | 97.51 [96.97, 97.98] | 96.72 | 98.31 | p=0.030 |
| deepseek-r1-qwen-14b | 94.27 [93.50, 95.00] | 90.59 | 98.26 | p<1e-28 |
| gpt-5-nano | 89.57 [88.60, 90.48] | 82.65 | 97.74 | p<1e-77 |
| nomenklatura RegressionV1 | 85.80 [84.76, 86.89] | 75.16 | 99.95 | p<1e-116 |
| llama-3.1-8b | 76.53 [75.01, 78.08] | 84.79 | 69.74 | p<1e-151 |

Robustness across the entity-disjoint, source-disjoint and temporal splits is in [`results/tables/`](results/tables/); per-script F1, error taxonomy and cost are in [`results/`](results/).

## Setup

```bash
git clone --recurse-submodules https://github.com/chansmi/OSINT_entity_resolution.git
cd OSINT_entity_resolution
uv sync                       # core: data loading, API baselines, evaluation
cp .env.example .env          # add API keys only if you re-run API models
```

Optional extras: `uv sync --extra local` (open-source models on GPU), `--extra finetune`, `--extra dspy`, `--extra notebooks`. Without uv: `pip install -r requirements.txt`. Commands below assume an activated environment (or prefix them with `uv run`).

## Data

### Download the corpus

```bash
python scripts/download_data.py                         # from the Hugging Face Hub
python scripts/download_data.py --source opensanctions  # or directly from OpenSanctions
```

Both write `data/raw/pairs-20251209.json.gz` and verify its sha256 against `data/splits/manifest.json`.

### Load it

```python
import sys; sys.path.insert(0, "scripts")
from load_data import load_pairs

for pair in load_pairs("data/raw/pairs-20251209.json.gz"):   # streams, low memory
    pair["left"], pair["right"], pair["judgement"]            # "positive" / "negative"
```

`load_pairs` reads JSON Lines (`.jsonl`, `.json.gz`), JSON sample files and Parquet, and always yields the same nested dicts.

Straight from the Hub with `datasets` (`pip install datasets`):

```python
from datasets import load_dataset
pairs  = load_dataset("sanctions-er-anon/opensanctions_pairs", split="train")                 # 755,540 pairs
sample = load_dataset("sanctions-er-anon/opensanctions_pairs", "sample_1000", split="train")  # 1,000-pair sample
```

The Hub also serves an auto-converted Parquet copy of both configs on the `refs/convert/parquet` branch, which the dataset viewer and Croissant metadata use.

### Parquet

Entity `properties` vary by schema, so the Parquet layout keeps flat columns for filtering (`judgement`, `left_id`, `left_schema`, `left_caption`, `left_datasets`, and the same for `right`) plus the full entity as `left_json` / `right_json`, which round-trips losslessly.

```bash
python scripts/export_parquet.py --splits     # data/parquet/splits/*.parquet
python scripts/export_parquet.py data/raw/pairs-20251209.json.gz data/parquet/pairs.parquet
```

```python
import json, pandas as pd
df = pd.read_parquet("data/parquet/splits/common_analyst_judged.parquet")
left = json.loads(df.left_json[0])
```

Parquet files are generated locally and not committed.

### Record format

```json
{
  "left": {
    "id": "ofac-40604",
    "caption": "Aliasghar Norouzi",
    "schema": "Person",
    "datasets": ["us_ofac_sdn", "us_trade_csl"],
    "first_seen": "2023-04-20T10:27:20",
    "properties": {"alias": ["Ali Asghar Norowzi"], "birthDate": ["1962-11-11"], "nationality": ["ir"]}
  },
  "right": {"...": "..."},
  "judgement": "positive"
}
```

Split files also carry `pair_key = "{left_id}__{right_id}__{corpus_line}"` to join back to the corpus.

## Evaluation splits

Built deterministically (seed 42) by `eval/build_splits.py` from the corpus and `eval/config.yaml`. The development samples in `data/samples/` are excluded from every split.

| Split | N | Contents |
|---|---|---|
| `common_analyst_judged` | 3,000 | Primary split. Analyst-judged pairs only, natural label ratio. |
| `common_auto_merge` | 1,000 | Auto-merge positives (relationship schemas, positive by construction). Combined with the above for overall F1. |
| `entity_disjoint` | 3,000 | At most one pair per canonical cluster, removing near-duplicate repetition. |
| `source_disjoint` | 3,000 | Left and right share no source dataset. |
| `temporal` | 2,000 | Both entities first seen on or after 2025-01-01. |
| `script_eval` | 2,000 | 400 per script bucket: Latin, Cyrillic, Arabic, CJK, cross-script. |

## Reproduce the results

Everything below runs on a laptop except open-source model inference.

```bash
# 1. Rebuild the splits; they match the committed files byte for byte (about 40 s)
python scripts/download_data.py
python eval/build_splits.py --config eval/config.yaml
git diff --stat data/splits            # only the generated_at timestamp changes

# 2. Recompute every table from the committed predictions (no API keys needed)
python eval/make_tables.py
python eval/significance.py $(for f in results/predictions/*__common_analyst_judged.jsonl; do \
    m=$(basename $f); echo --pred ${m%%__*}=$f; done)
python eval/analysis.py per-script --label gpt-4o \
    --pred results/predictions/gpt-4o__script_eval.jsonl --split data/splits/script_eval.jsonl
python eval/analysis.py clusters

# 3. Re-run a model on a split (writes results/predictions/<model>__<split>.jsonl)
python eval/run_model.py --runner zeroshot --model gpt-4o --split common_analyst_judged --parallel 20
python eval/run_model.py --runner local --model llama-8b --split common_analyst_judged   # GPU
python eval/run_nomenklatura.py --input data/splits/common_analyst_judged.jsonl \
    --output results/predictions/nomenklatura-regv1__common_analyst_judged.jsonl
```

Predictions align to their split by line position (`pair_index`). Re-running API models can shift results slightly because provider models are not fully deterministic.

### Baselines

- `scripts/baselines/llm_zeroshot.py`, `llm_fewshot.py`: API models (model ids in `scripts/baselines/config.py`).
- `scripts/baselines/llm_local.py`: Llama 3.1 8B and DeepSeek-R1-Distill-Qwen-14B via `transformers`. `python scripts/download_models.py` fetches weights; gated models need `HF_TOKEN`. Set `PRETRAINED_MODELS_DIR` to reuse an existing model directory.
- `eval/run_nomenklatura.py`: OpenSanctions' `RegressionV1` matcher from the pinned `scripts/vendor/nomenklatura` submodule (Python 3.11+, ICU).
- `scripts/baselines/simple_fuzzy.py`: deterministic weighted string-similarity baseline.
- `scripts/baselines/dspy_er/`: DSPy MIPROv2 prompt optimisation; optimised prompts are in `data/prompts/`.
- `scripts/finetune/`: LoRA fine-tuning with entity-level train/test splits.

Sample-based scripts take `--offset 200` to skip the 200 development pairs of `data/samples/sample_1000.json`.

### Earlier sample-based results

`results/sample_1000/` holds per-pair predictions for the original 1,000-pair sample (800 test pairs) and the manifest that `scripts/build_tables.py` uses to recompute those numbers with confidence intervals.

## Repository layout

```
data/
  croissant/          Croissant + RAI metadata
  splits/             frozen evaluation splits (JSONL) + manifest.json with sha256s
  samples/            1k / 10k development samples (excluded from all splits)
  prompts/            optimised prompts per model, registry.json
  examples/           curated few-shot examples
  raw/, parquet/      downloaded corpus and local Parquet exports (gitignored)
eval/
  config.yaml         seed, split definitions, model list, pricing
  build_splits.py     corpus -> splits + manifest + cluster statistics
  run_model.py        run a baseline on a split -> results/predictions/
  run_nomenklatura.py rule baseline on a split
  significance.py     bootstrap CIs, exact McNemar, Wilson intervals
  analysis.py         per-script F1, cost frontier, error taxonomy, cluster sizes
  make_tables.py      predictions -> results/tables/*.md
results/
  predictions/        <model>__<split>.jsonl for every reported number
  tables/             generated markdown tables
  *.csv, *.json       metrics, significance, per-script, taxonomy, cost, cluster sizes
  sample_1000/        predictions and manifest for the earlier sample-based evaluation
scripts/
  download_data.py    fetch and verify the corpus
  export_parquet.py   JSONL -> Parquet
  load_data.py        streaming loader (JSONL, JSON, Parquet)
  baselines/          rule, fuzzy, API, local and DSPy baselines
  finetune/           LoRA fine-tuning
  vendor/nomenklatura pinned submodule
notebooks/            exploratory data analysis
```

## Citation

If you use the dataset or code, please cite:

```bibtex
@inproceedings{smith2026opensanctions,
  title     = {OpenSanctions Pairs: A Large-Scale Dataset for Pairwise Entity Matching},
  author    = {Smith, Chandler and Sesodia, Magnus and Lindenberg, Friedrich and Schroeder de Witt, Christian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Track on Evaluations and Datasets},
  year      = {2026}
}
```

Chandler Smith and Magnus Sesodia contributed equally.

## License

Code: MIT, see [LICENSE](LICENSE). Data: CC BY-NC 4.0, following the OpenSanctions non-commercial licence; commercial use requires a licence from OpenSanctions.
