#!/usr/bin/env bash
# Re-run every (model, split) reported in the paper, then rebuild all tables.
#
# API models are called through an OpenAI-compatible endpoint. The paper's runs used a
# LiteLLM proxy (https://github.com/BerriAI/litellm), so one client reaches both OpenAI
# and Anthropic models: set OPENAI_BASE_URL and OPENAI_API_KEY in .env. With plain
# OpenAI credentials only the openai/* models will work.
#
# Open models need a GPU and `uv sync --extra local`; weights are fetched into
# PRETRAINED_MODELS_DIR (Llama 3.1 is gated: set HF_TOKEN).
#
# Idempotent: any (model, split) whose prediction file already exists is skipped, so
# delete a file under results/predictions/ to re-run it.
#
#   bash eval/run_all.sh api        # closed models
#   bash eval/run_all.sh local      # open models (GPU)
#   bash eval/run_all.sh rule       # nomenklatura RegressionV1 (needs the submodule)
#   bash eval/run_all.sh tables     # aggregate only
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a

ALL_SPLITS=(common_analyst_judged common_auto_merge entity_disjoint source_disjoint temporal script_eval)
CORE=(common_analyst_judged common_auto_merge)

run_api() {  # model_id name split parallel
  local out="results/predictions/$2__$3.jsonl"
  [ -s "$out" ] && { echo "skip $out"; return; }
  python eval/run_model.py --runner zeroshot --model "$1" --name "$2" --split "$3" --parallel "$4"
}
run_local() {  # model_key name split
  local out="results/predictions/$2__$3.jsonl"
  [ -s "$out" ] && { echo "skip $out"; return; }
  python eval/run_model.py --runner local --model "$1" --name "$2" --split "$3"
}

api() {
  # model id (as sent to the endpoint) | name in results | parallel requests
  local MODELS=(
    "openai/gpt-4o|gpt-4o|15"
    "openai/gpt-5.2-pro|gpt-5.2-pro|8"        # reasoning effort: medium (runner default)
    "anthropic/claude-opus-4-5-20251101|claude-opus-4.5|10"
    "anthropic/claude-sonnet-4-5|claude-sonnet-4.5|12"
    "openai/gpt-5-nano|gpt-5-nano|12"
  )
  for spec in "${MODELS[@]}"; do
    IFS='|' read -r id name par <<< "$spec"
    for s in "${CORE[@]}"; do run_api "$id" "$name" "$s" "$par"; done
  done
  for s in entity_disjoint source_disjoint temporal script_eval; do run_api openai/gpt-4o gpt-4o "$s" 15; done
  run_api openai/gpt-5-nano gpt-5-nano script_eval 12
}

local_models() {
  for s in "${ALL_SPLITS[@]}"; do run_local llama-8b llama-3.1-8b "$s"; done
  for s in "${CORE[@]}"; do run_local deepseek-14b deepseek-r1-qwen-14b "$s"; done
}

rule() {
  for s in "${ALL_SPLITS[@]}"; do
    out="results/predictions/nomenklatura-regv1__$s.jsonl"
    [ -s "$out" ] && { echo "skip $out"; continue; }
    python eval/run_nomenklatura.py --input "data/splits/$s.jsonl" --output "$out"
  done
}

tables() {
  python eval/significance.py $(for f in results/predictions/*__common_analyst_judged.jsonl; do
    m=$(basename "$f"); echo --pred "${m%%__*}=$f"; done)
  python eval/make_tables.py
  for m in gpt-4o llama-3.1-8b; do
    python eval/analysis.py per-script --label "$m" \
      --pred "results/predictions/${m}__script_eval.jsonl" --split data/splits/script_eval.jsonl
    python eval/analysis.py taxonomy --label "$m" --pred "results/predictions/${m}__common_analyst_judged.jsonl"
  done
}

case "${1:-tables}" in
  api) api; tables ;;
  local) local_models; tables ;;
  rule) rule; tables ;;
  tables) tables ;;
  *) echo "usage: $0 {api|local|rule|tables}"; exit 1 ;;
esac
