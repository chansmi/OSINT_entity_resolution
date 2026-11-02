#!/usr/bin/env python3
"""Thin wrapper that runs a baseline on a frozen split and writes the prediction to the
canonical path results/predictions/<model>__<split>.jsonl (so make_tables can glob them).

It shells out to the existing runners (no logic duplicated) and moves their auto-named
output to the canonical name.

    # API model (local, needs OPENAI_API_KEY / ANTHROPIC_API_KEY):
    python eval/run_model.py --runner zeroshot --model gpt-4o \
        --split common_analyst_judged --parallel 20

    # Open-source model on a local GPU (optionally wrap with a scheduler via --launcher):
    python eval/run_model.py --runner local --model llama-8b \
        --split common_analyst_judged --launcher "flux run -N1 -n1 -g8"
"""
import argparse
import glob
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = {
    "zeroshot": "scripts/baselines/llm_zeroshot.py",
    "local": "scripts/baselines/llm_local.py",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True, choices=list(RUNNERS))
    ap.add_argument("--model", required=True, help="model id (zeroshot) or key (local: llama-8b/deepseek-14b)")
    ap.add_argument("--split", required=True, help="split name under data/splits/")
    ap.add_argument("--name", help="canonical model name for the output file (default: --model)")
    ap.add_argument("--parallel", type=int)
    ap.add_argument("--reasoning")
    ap.add_argument("--limit", type=int, help="cap pairs (for smoke tests)")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--launcher", default="", help="prefix e.g. 'flux run -N1 -n1 -g8'")
    args = ap.parse_args()

    name = args.name or args.model
    split_file = ROOT / "data" / "splits" / f"{args.split}.jsonl"
    assert split_file.exists(), f"missing split {split_file}"
    raw_dir = ROOT / "results" / "predictions" / "_raw" / f"{name}__{args.split}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    final = ROOT / "results" / "predictions" / f"{name}__{args.split}.jsonl"

    # Our splits are JSONL; llm_local.py uses a plain json.load, so materialize a JSON
    # array input (order preserved -> position-aligned predictions). --limit slices here.
    rows = [json.loads(l) for l in open(split_file, encoding="utf-8") if l.strip()]
    if args.limit:
        import random
        random.Random(42).shuffle(rows)  # stratified subset (split file is ordered pos-then-neg)
        rows = rows[: args.limit]
    input_file = raw_dir / "input.json"
    json.dump(rows, open(input_file, "w", encoding="utf-8"))

    cmd = shlex.split(args.launcher) + [
        sys.executable, str(ROOT / RUNNERS[args.runner]),
        "--model", args.model,
        "--input", str(input_file),
        "--output", str(raw_dir),
        "--shots", str(args.shots),
    ]
    if args.parallel:
        cmd += ["--parallel", str(args.parallel)]
    if args.reasoning:
        cmd += ["--reasoning", args.reasoning]

    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    produced = sorted(glob.glob(str(raw_dir / "*.jsonl")), key=lambda p: Path(p).stat().st_mtime)
    assert produced, f"runner produced no .jsonl in {raw_dir}"
    Path(produced[-1]).replace(final)
    print(f"-> {final}")


if __name__ == "__main__":
    main()
