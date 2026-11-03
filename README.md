# OpenSanctions Pairs

Code and evaluation artifacts for **OpenSanctions Pairs: A Large-Scale Dataset for Pairwise Entity Matching**.

The task: given two entity records (persons, organisations, vessels, ...) drawn from sanctions, PEP and watchlist sources, decide whether they refer to the same real-world entity. The corpus has 755,540 labelled pairs from [OpenSanctions](https://www.opensanctions.org/docs/opensource/pairs/).

- **Dataset:** [huggingface.co/datasets/sanctions-er-anon/opensanctions_pairs](https://huggingface.co/datasets/sanctions-er-anon/opensanctions_pairs) (CC BY-NC 4.0, gated: accept the terms of use once, approval is automatic)
- **Synthetic preview** (real schema, fabricated values, no login needed): [`data/preview/synthetic_pairs.json`](data/preview/synthetic_pairs.json)
- **Croissant metadata (with RAI fields):** [`data/croissant/croissant.json`](data/croissant/croissant.json), rebuilt from the Hub with `data/croissant/finalize_croissant.py`
- **Frozen evaluation splits:** rebuilt deterministically from the corpus and verified against the checksums in [`data/splits/manifest.json`](data/splits/manifest.json)
- **Every model prediction behind the reported numbers:** [`results/predictions/`](results/predictions/)

The records name real people and organisations taken from official sanctions, watchlist and PEP sources, so they are distributed only through the gated Hugging Face release. This repository holds code, checksums, predictions and results, never the records themselves. See the dataset card for the terms of use, legal basis and how to request corrections.

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

The Hugging Face dataset is gated. Log in once (`hf auth login`, or set `HF_TOKEN`) and accept the terms of use on the [dataset page](https://huggingface.co/datasets/sanctions-er-anon/opensanctions_pairs); approval is automatic.

```bash
python scripts/download_data.py                         # Hugging Face Hub, release v1
python scripts/download_data.py --sample                # also the 1,000-pair dev sample -> data/samples/
python scripts/download_data.py --source opensanctions  # or the same snapshot from OpenSanctions
```

Each writes `data/raw/pairs-20251209.json.gz` and verifies its sha256 against `data/splits/manifest.json`.

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
from datasets import load_dataset   # after logging in and accepting the terms
pairs  = load_dataset("sanctions-er-anon/opensanctions_pairs", split="train", revision="v1")                 # 755,540 pairs
sample = load_dataset("sanctions-er-anon/opensanctions_pairs", "sample_1000", split="train", revision="v1")  # 1,000-pair sample
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

The example below is from the synthetic preview, so every value is invented; real records have the same structure.

```json
{
  "left": {
    "id": "synthetic-0021",
    "caption": "Lomfen Teltelis",
    "schema": "Person",
    "datasets": ["synthetic_list_a"],
    "first_seen": "2022-12-11T23:00:00",
    "properties": {"birthDate": ["1974-08-16"], "nationality": ["xb"]}
  },
  "right": {
    "id": "synthetic-0022",
    "caption": "Ломфен Телтелис",
    "properties": {"alias": ["LOMFEN TELTELIS"], "birthDate": ["1974-08-16"]}
  },
  "judgement": "positive"
}
```

Split files also carry `pair_key = "{left_id}__{right_id}__{corpus_line}"` to join back to the corpus.

## Evaluation splits

Built deterministically (seed 42) by `eval/build_splits.py` from the corpus and `eval/config.yaml`, then checked against the published checksums. The split files contain real records, so they are rebuilt locally rather than committed. The development samples used during prompt development are excluded from every split; `data/splits/excluded_dev_pairs.json` stores them as hashed entity-id pairs.

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
# 1. Download the corpus and rebuild the splits (about 40 s); the script checks
#    every split against the sha256 in data/splits/manifest.json
python scripts/download_data.py
python eval/build_splits.py --config eval/config.yaml

# 2. Recompute every table from the committed predictions (no API keys needed)
python eval/make_tables.py
python eval/significance.py $(for f in results/predictions/*__common_analyst_judged.jsonl; do \
    m=$(basename $f); echo --pred ${m%%__*}=$f; done)
python eval/analysis.py per-script --label gpt-4o \
    --pred results/predictions/gpt-4o__script_eval.jsonl --split data/splits/script_eval.jsonl
python eval/analysis.py clusters
python eval/make_figure.py             # Figure 2 (needs --extra notebooks)
python eval/export_clusters.py         # entity -> cluster mapping, data/clusters/clusters.parquet

# 3. Re-run everything reported in the paper (skips existing prediction files)
bash eval/run_all.sh api      # closed models, via an OpenAI-compatible endpoint
bash eval/run_all.sh local    # Llama 3.1 8B, DeepSeek-R1-Distill-Qwen-14B (GPU)
bash eval/run_all.sh rule     # nomenklatura RegressionV1

# Or one model on one split (writes results/predictions/<model>__<split>.jsonl)
python eval/run_model.py --runner zeroshot --model gpt-4o --split common_analyst_judged --parallel 20
python eval/run_model.py --runner local --model llama-8b --split common_analyst_judged   # GPU
python eval/run_nomenklatura.py --input data/splits/common_analyst_judged.jsonl \
    --output results/predictions/nomenklatura-regv1__common_analyst_judged.jsonl
```

API models were called through a [LiteLLM](https://github.com/BerriAI/litellm) proxy, so a single OpenAI-compatible client reaches both OpenAI and Anthropic models: set `OPENAI_BASE_URL` and `OPENAI_API_KEY` in `.env`. Error categories in `results/taxonomy_*.csv` come from `eval/analysis.py taxonomy`, a keyword rule over each model's stated rationale. The released prediction files keep each error's category but not the rationale text, which quotes the records. A response that cannot be parsed is scored as a negative prediction (27 of GPT-5 Nano's 3,000).

Predictions align to their split by line position (`pair_index`). Re-running API models can shift results slightly because provider models are not fully deterministic.

### Baselines

- `scripts/baselines/llm_zeroshot.py`, `llm_fewshot.py`: API models (model ids in `scripts/baselines/config.py`).
- `scripts/baselines/llm_local.py`: Llama 3.1 8B and DeepSeek-R1-Distill-Qwen-14B via `transformers`. `python scripts/download_models.py` fetches weights; gated models need `HF_TOKEN`. Set `PRETRAINED_MODELS_DIR` to reuse an existing model directory.
- `eval/run_nomenklatura.py`: OpenSanctions' `RegressionV1` matcher from the pinned `scripts/vendor/nomenklatura` submodule (Python 3.11+, ICU).
- `scripts/baselines/simple_fuzzy.py`: deterministic weighted string-similarity baseline.
- `scripts/baselines/dspy_er/`: DSPy MIPROv2 prompt optimisation; optimised prompts are in `data/prompts/`. Their few-shot demonstrations were real records and have been removed; rerun the optimiser to regenerate them.
- `scripts/finetune/`: LoRA fine-tuning with entity-level train/test splits.

Sample-based scripts read `data/samples/sample_1000.json` (fetch it with `python scripts/download_data.py --sample`) and take `--offset 200` to skip its 200 development pairs.

### Earlier sample-based results

`results/sample_1000/` holds per-pair predictions for the original 1,000-pair sample (800 test pairs) and the manifest that `scripts/build_tables.py` uses to recompute those numbers with confidence intervals.

## Repository layout

```
data/
  croissant/          Croissant + RAI metadata, copy of the dataset card
  preview/            synthetic preview (fabricated values, real schema)
  splits/             manifest.json (sha256 of every split) + hashed dev-sample exclusion list
  samples/            dev sample, downloaded with --sample (gitignored)
  prompts/            optimised prompts per model, registry.json
  examples/           curated few-shot examples
  raw/, parquet/, clusters/   downloaded corpus and local derived files (gitignored)
eval/
  config.yaml         seed, split definitions, model list, pricing
  build_splits.py     corpus -> splits (verified against manifest) + cluster statistics
  export_clusters.py  entity -> canonical cluster mapping (n-ary structure)
  run_model.py        run a baseline on a split -> results/predictions/
  run_nomenklatura.py rule baseline on a split
  significance.py     bootstrap CIs, exact McNemar, Wilson intervals
  analysis.py         per-script F1, cost frontier, error taxonomy, cluster sizes
  make_tables.py      predictions -> results/tables/*.md
  make_figure.py      results/metrics.csv -> Figure 2
  run_all.sh          every (model, split) in the paper, then all tables
results/
  predictions/        <model>__<split>.jsonl for every reported number
  tables/             generated markdown tables
  *.csv, *.json       metrics, significance, per-script, taxonomy, cost, cluster sizes
  sample_1000/        predictions and manifest for the earlier sample-based evaluation
scripts/
  download_data.py    fetch and verify the corpus (gated Hub or OpenSanctions)
  make_synthetic_preview.py  regenerate data/preview/
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
@inproceedings{smith2026opensanctionspairs,
  title     = {OpenSanctions Pairs: A Large-Scale Dataset for Pairwise Entity Matching},
  author    = {Smith, Chandler and Sesodia, Magnus and Lindenberg, Friedrich and Schroeder de Witt, Christian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Evaluations and Datasets Track},
  year      = {2026}
}
```

Chandler Smith and Magnus Sesodia contributed equally.

## License

Code: MIT, see [LICENSE](LICENSE). Data: CC BY-NC 4.0, following the OpenSanctions non-commercial licence; commercial use requires a licence from OpenSanctions.
