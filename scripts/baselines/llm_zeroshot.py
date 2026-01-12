#!/usr/bin/env python3
"""
LLM Zero-Shot Entity Resolution Baseline

Uses GPT-5 models to classify entity pairs as same/different entities.
Supports parallel execution for ~30x speedup.

Usage:
    # Sequential (slow)
    python scripts/baselines/llm_zeroshot.py --input data/samples/sample_1000.json

    # Parallel (fast, recommended)
    python scripts/baselines/llm_zeroshot.py --input data/samples/sample_1000.json --parallel 30

    # With options
    python scripts/baselines/llm_zeroshot.py --input data/samples/sample_1000.json --parallel 30 --model gpt-5.2 --reasoning high
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.evaluate import evaluate, print_evaluation_report
from scripts.experiment import Experiment

load_dotenv()

# --- Prompts ---
# Optimized prompt: frames task as conflict detection (default positive)
# This reduces false negatives by 97% compared to the conservative approach

SYSTEM_PROMPT = """You are an expert entity resolution system for sanctions screening. Your task is to determine whether two entity records refer to the same real-world person or organization.

Your primary task is to identify CONFLICTS, not similarities.

Key Principles:
- Name variations (transliterations, nicknames, titles) are common
- Missing fields are normal - absence of data is NOT evidence of difference
- Same entity often appears across multiple sources with variations

Decision Process:
1. Look for CONTRADICTORY evidence (different dates, conflicting IDs, incompatible attributes)
2. If NO contradictions found -> POSITIVE (same entity)
3. Only NEGATIVE if explicit conflicts exist

The DEFAULT is POSITIVE unless you find proof of difference."""

USER_TEMPLATE = """Compare these two entities:

=== ENTITY A ===
{entity_a}

=== ENTITY B ===
{entity_b}

Search for CONTRADICTIONS. If none found, classify as POSITIVE."""


def get_response_schema(ternary: bool = False) -> dict:
    """Return JSON schema for structured output."""
    classification_enum = ["positive", "negative", "uncertain"] if ternary else ["positive", "negative"]

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "entity_match_result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "classification": {
                        "type": "string",
                        "enum": classification_enum,
                        "description": "Whether entities are the same (positive), different (negative), or uncertain"
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Confidence level in the classification"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Detailed explanation of the decision including key matching/mismatching factors"
                    }
                },
                "required": ["classification", "confidence", "reasoning"],
                "additionalProperties": False
            }
        }
    }


def format_entity(entity: dict) -> str:
    """Convert entity dict to readable text for the prompt."""
    props = entity.get("properties", {})
    lines = [f"Type: {entity.get('schema', 'Unknown')}"]

    # Field definitions: (key, label, limit, separator)
    field_specs = [
        ("name", "Names", 8, ", "),
        ("alias", "Aliases", 4, ", "),
        ("birthDate", "Birth Date", None, ", "),
        ("birthPlace", "Birth Place", 2, ", "),
        ("nationality", "Nationality", None, ", "),
        ("country", "Country", None, ", "),
        ("address", "Address", 3, "; "),
        ("idNumber", "ID Numbers", 3, ", "),
        ("passportNumber", "Passport", 2, ", "),
        ("gender", "Gender", None, ", "),
        ("position", "Position", 2, ", "),
        ("firstName", "First Name", None, ", "),
        ("lastName", "Last Name", None, ", "),
    ]

    for key, label, limit, sep in field_specs:
        if key in props:
            values = props[key][:limit] if limit else props[key]
            lines.append(f"{label}: {sep.join(values)}")

    return "\n".join(lines)


# --- Classification Helpers ---

def build_messages(pair: dict) -> list:
    """Build the chat messages for a classification request."""
    prompt = USER_TEMPLATE.format(
        entity_a=format_entity(pair["left"]),
        entity_b=format_entity(pair["right"])
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]


def parse_response(response, pair: dict, pair_index: int) -> dict:
    """Parse API response into result dict."""
    result = json.loads(response.choices[0].message.content)
    result["tokens_used"] = {
        "prompt": response.usage.prompt_tokens,
        "completion": response.usage.completion_tokens,
        "total": response.usage.total_tokens
    }
    result["ground_truth"] = pair["judgement"]
    result["pair_index"] = pair_index
    return result


def make_error_result(pair: dict, pair_index: int, error: Exception) -> dict:
    """Create error result dict."""
    return {
        "classification": "error",
        "ground_truth": pair["judgement"],
        "pair_index": pair_index,
        "error": str(error)
    }


# --- Async Classification ---

async def classify_pair_async(
    client: AsyncOpenAI,
    pair: dict,
    pair_index: int,
    semaphore: asyncio.Semaphore,
    model: str,
    reasoning_effort: str,
    ternary: bool
) -> dict:
    """Async classify a single entity pair."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=build_messages(pair),
                response_format=get_response_schema(ternary),
                max_completion_tokens=4000,
                reasoning_effort=reasoning_effort
            )
            return parse_response(response, pair, pair_index)
        except Exception as e:
            return make_error_result(pair, pair_index, e)


async def run_parallel_experiment(
    data: list,
    model: str,
    reasoning_effort: str,
    ternary: bool,
    max_concurrent: int,
    output_file: Path
) -> list:
    """Run classification on all pairs in parallel."""
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks
    tasks = [
        classify_pair_async(client, pair, i, semaphore, model, reasoning_effort, ternary)
        for i, pair in enumerate(data)
    ]

    # Run with progress bar
    results = []
    with open(output_file, 'w') as f:
        for coro in tqdm_asyncio.as_completed(tasks, desc=f"Classifying ({max_concurrent} parallel)"):
            result = await coro
            results.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()

    # Sort by pair_index to maintain order
    results.sort(key=lambda x: x.get("pair_index", 0))
    return results


# --- Sync Classification (fallback) ---

def classify_pair_sync(
    client: OpenAI,
    pair: dict,
    pair_index: int,
    model: str,
    reasoning_effort: str,
    ternary: bool
) -> dict:
    """Sync classify a single entity pair."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=build_messages(pair),
            response_format=get_response_schema(ternary),
            max_completion_tokens=4000,
            reasoning_effort=reasoning_effort
        )
        return parse_response(response, pair, pair_index)
    except Exception as e:
        return make_error_result(pair, pair_index, e)


def run_sequential_experiment(
    data: list,
    model: str,
    reasoning_effort: str,
    ternary: bool,
    output_file: Path
) -> list:
    """Run classification sequentially (slow but simple)."""
    client = OpenAI()
    results = []

    with open(output_file, 'w') as f:
        for i, pair in enumerate(tqdm(data, desc="Classifying (sequential)")):
            result = classify_pair_sync(client, pair, i, model, reasoning_effort, ternary)
            results.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()

    return results


# --- Main ---

def load_data(filepath: str) -> list:
    """Load entity pairs from JSON file.

    Supports:
    - v1 format: JSON array of pairs
    - v2 format: JSON object with 'metadata' and 'pairs' keys
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle v2 format (with metadata wrapper)
    if isinstance(data, dict) and 'pairs' in data:
        return data['pairs']

    return data


def run_experiment(
    input_path: str,
    output_dir: str,
    model: str = "gpt-5-nano",
    reasoning_effort: str = "medium",
    ternary: bool = False,
    limit: int = None,
    offset: int = 0,
    parallel: int = None
) -> dict:
    """Run the full experiment."""
    data = load_data(input_path)

    # Apply offset first (skip first N pairs, e.g., for using same test set as baselines)
    if offset:
        data = data[offset:]

    if limit:
        data = data[:limit]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "ternary" if ternary else "binary"
    responses_file = output_dir / f"llm_responses_{mode}_{model.replace('.', '_')}.jsonl"

    print(f"\nRunning {mode} classification on {len(data)} pairs...")
    print(f"Model: {model}")
    print(f"Reasoning effort: {reasoning_effort}")
    print(f"Parallel: {parallel if parallel else 'No (sequential)'}")
    print(f"Output: {responses_file}\n")

    start_time = time.time()

    if parallel:
        results = asyncio.run(run_parallel_experiment(
            data, model, reasoning_effort, ternary, parallel, responses_file
        ))
    else:
        results = run_sequential_experiment(
            data, model, reasoning_effort, ternary, responses_file
        )

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/len(data):.2f}s per pair)")

    # Evaluate
    predictions = []
    ground_truth = []
    error_count = 0

    for r in results:
        if r.get("classification") == "error":
            error_count += 1
            continue
        if ternary and r.get("classification") == "uncertain":
            continue
        predictions.append(r["classification"])
        ground_truth.append(r["ground_truth"])

    if not predictions:
        print("No valid predictions to evaluate!")
        return {"error": "No valid predictions"}, results

    metrics = evaluate(ground_truth, predictions)
    total_tokens = sum(r.get("tokens_used", {}).get("total", 0) for r in results if "tokens_used" in r)

    # Track experiment
    exp = Experiment(method="llm_zeroshot", input_file=input_path)
    exp.set_params(
        model=model,
        mode=mode,
        reasoning_effort=reasoning_effort,
        parallel=parallel,
        total_pairs=len(results),
        evaluated_pairs=len(predictions),
        errors=error_count,
        total_tokens=total_tokens,
        elapsed_seconds=round(elapsed, 2)
    )

    if ternary:
        uncertain_count = sum(1 for r in results if r.get("classification") == "uncertain")
        coverage = len(predictions) / (len(results) - error_count) if (len(results) - error_count) > 0 else 0
        exp.set_params(uncertain_count=uncertain_count, coverage=round(coverage, 4))

    exp.set_metrics(
        accuracy=metrics['accuracy'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1=metrics['f1'],
        confusion_matrix=metrics['confusion_matrix']
    )

    result_file = exp.save(str(output_dir))
    exp.print_summary()

    # Keep responses file reference in metrics for backward compatibility
    metrics["experiment"] = exp.params
    metrics["experiment"]["responses_file"] = str(responses_file)

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="LLM Zero-Shot Entity Resolution")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", default="data/outputs", help="Output directory")
    parser.add_argument("--model", default="gpt-5-nano",
                       help="GPT model: gpt-5-nano, gpt-5, gpt-5.2, gpt-5.2-pro")
    parser.add_argument("--reasoning", default="medium",
                       choices=["none", "low", "medium", "high", "xhigh"],
                       help="Reasoning effort level")
    parser.add_argument("--ternary", action="store_true",
                       help="Use ternary mode (positive/negative/uncertain)")
    parser.add_argument("--limit", type=int, help="Limit number of pairs")
    parser.add_argument("--offset", type=int, default=0,
                       help="Skip first N pairs (use with --dev-ratio for same test set as baselines)")
    parser.add_argument("--parallel", type=int, default=None,
                       help="Number of parallel requests (recommended: 30)")

    args = parser.parse_args()

    metrics, results = run_experiment(
        input_path=args.input,
        output_dir=args.output,
        model=args.model,
        reasoning_effort=args.reasoning,
        ternary=args.ternary,
        limit=args.limit,
        offset=args.offset,
        parallel=args.parallel
    )

    if "error" in metrics:
        print(f"Experiment failed: {metrics['error']}")
        return

    # Cost estimate (experiment summary already printed by tracker)
    exp = metrics["experiment"]
    est_cost = exp['total_tokens'] * 0.0000005
    print(f"Estimated cost: ${est_cost:.4f}")


if __name__ == "__main__":
    main()
