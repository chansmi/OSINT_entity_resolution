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

load_dotenv()

# --- Prompts ---

SYSTEM_PROMPT = """You are an expert entity resolution system. Your task is to determine whether two entity records refer to the same real-world person or organization.

Consider:
- Name variations (transliterations, nicknames, titles, spelling differences)
- Matching identifiers (birth dates, ID numbers, addresses)
- Contextual clues (nationality, occupation, relationships)

Be aware that:
- Arabic names may have multiple Latin transliterations
- Dates may be partial (year only) or in different formats
- Missing fields are common - absence of data is not evidence of difference
- Same person may appear in multiple sanction lists with slight variations

Think carefully through the evidence before making your decision."""

USER_TEMPLATE = """Compare these two entities and determine if they are the SAME real-world entity.

=== ENTITY A ===
{entity_a}

=== ENTITY B ===
{entity_b}

Analyze the similarities and differences carefully, then provide your classification."""


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
    lines = []

    # Schema type
    schema = entity.get("schema", "Unknown")
    lines.append(f"Type: {schema}")

    # Names (include all for better matching)
    if "name" in props:
        names = props["name"][:8]
        lines.append(f"Names: {', '.join(names)}")

    # Aliases
    if "alias" in props:
        aliases = props["alias"][:4]
        lines.append(f"Aliases: {', '.join(aliases)}")

    # Birth info
    if "birthDate" in props:
        lines.append(f"Birth Date: {', '.join(props['birthDate'])}")
    if "birthPlace" in props:
        lines.append(f"Birth Place: {', '.join(props['birthPlace'][:2])}")

    # Location info
    if "nationality" in props:
        lines.append(f"Nationality: {', '.join(props['nationality'])}")
    if "country" in props:
        lines.append(f"Country: {', '.join(props['country'])}")
    if "address" in props:
        addresses = props["address"][:3]
        lines.append(f"Address: {'; '.join(addresses)}")

    # Identifiers
    if "idNumber" in props:
        lines.append(f"ID Numbers: {', '.join(props['idNumber'][:3])}")
    if "passportNumber" in props:
        lines.append(f"Passport: {', '.join(props['passportNumber'][:2])}")

    # Other useful fields
    if "gender" in props:
        lines.append(f"Gender: {', '.join(props['gender'])}")
    if "position" in props:
        lines.append(f"Position: {', '.join(props['position'][:2])}")

    # Name components
    if "firstName" in props:
        lines.append(f"First Name: {', '.join(props['firstName'])}")
    if "lastName" in props:
        lines.append(f"Last Name: {', '.join(props['lastName'])}")

    return "\n".join(lines)


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
            prompt = USER_TEMPLATE.format(
                entity_a=format_entity(pair["left"]),
                entity_b=format_entity(pair["right"])
            )

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format=get_response_schema(ternary),
                max_completion_tokens=4000,
                reasoning_effort=reasoning_effort
            )

            result = json.loads(response.choices[0].message.content)
            result["tokens_used"] = {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            }
            result["ground_truth"] = pair["judgement"]
            result["pair_index"] = pair_index
            return result

        except Exception as e:
            return {
                "classification": "error",
                "ground_truth": pair["judgement"],
                "pair_index": pair_index,
                "error": str(e)
            }


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
        prompt = USER_TEMPLATE.format(
            entity_a=format_entity(pair["left"]),
            entity_b=format_entity(pair["right"])
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format=get_response_schema(ternary),
            max_completion_tokens=4000,
            reasoning_effort=reasoning_effort
        )

        result = json.loads(response.choices[0].message.content)
        result["tokens_used"] = {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens
        }
        result["ground_truth"] = pair["judgement"]
        result["pair_index"] = pair_index
        return result

    except Exception as e:
        return {
            "classification": "error",
            "ground_truth": pair["judgement"],
            "pair_index": pair_index,
            "error": str(e)
        }


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
    """Load entity pairs from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_experiment(
    input_path: str,
    output_dir: str,
    model: str = "gpt-5-nano",
    reasoning_effort: str = "medium",
    ternary: bool = False,
    limit: int = None,
    parallel: int = None
) -> dict:
    """Run the full experiment."""
    data = load_data(input_path)

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

    # Add experiment metadata
    total_tokens = sum(r.get("tokens_used", {}).get("total", 0) for r in results if "tokens_used" in r)
    metrics["experiment"] = {
        "mode": mode,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "parallel": parallel,
        "total_pairs": len(results),
        "evaluated_pairs": len(predictions),
        "errors": error_count,
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed
    }

    if ternary:
        uncertain_count = sum(1 for r in results if r.get("classification") == "uncertain")
        metrics["experiment"]["uncertain_count"] = uncertain_count
        metrics["experiment"]["coverage"] = len(predictions) / (len(results) - error_count) if (len(results) - error_count) > 0 else 0

    # Save metrics
    metrics_file = output_dir / f"llm_results_{mode}_{model.replace('.', '_')}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

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
        parallel=args.parallel
    )

    if "error" in metrics:
        print(f"Experiment failed: {metrics['error']}")
        return

    # Print results
    print_evaluation_report(metrics)

    exp = metrics["experiment"]
    print(f"Experiment Details:")
    print(f"  Mode: {exp['mode']}")
    print(f"  Model: {exp['model']}")
    print(f"  Reasoning: {exp['reasoning_effort']}")
    print(f"  Parallel: {exp['parallel']}")
    print(f"  Total pairs: {exp['total_pairs']}")
    print(f"  Evaluated: {exp['evaluated_pairs']}")
    print(f"  Errors: {exp['errors']}")
    print(f"  Total tokens: {exp['total_tokens']:,}")
    print(f"  Time: {exp['elapsed_seconds']:.1f}s")

    if args.ternary:
        print(f"  Uncertain: {exp['uncertain_count']}")
        print(f"  Coverage: {exp['coverage']:.1%}")

    # Cost estimate
    est_cost = exp['total_tokens'] * 0.0000005
    print(f"  Est. cost: ${est_cost:.4f}")


if __name__ == "__main__":
    main()
