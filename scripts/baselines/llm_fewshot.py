#!/usr/bin/env python3
"""
LLM Few-Shot Entity Resolution Baseline

Uses GPT models to classify entity pairs with in-context examples.
Supports parallel execution for ~30x speedup.

Usage:
    # Basic few-shot (8 examples)
    python scripts/baselines/llm_fewshot.py --input data/samples/sample_1000.json --examples 8

    # With options
    python scripts/baselines/llm_fewshot.py --input data/samples/sample_1000.json \
        --examples 8 --strategy diverse --model gpt-5-nano --parallel 20

    # Dry run (preview prompts, no API calls)
    python scripts/baselines/llm_fewshot.py --dry-run --examples 8

    # Use predefined config
    python scripts/baselines/llm_fewshot.py --config few_shot_8
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
from scripts.load_data import load_sample
from scripts.baselines.config import ExperimentConfig, CONFIGS, add_config_args, get_config_from_args
from scripts.baselines.example_selector import get_examples, print_example_summary
from scripts.baselines.llm_zeroshot import (
    SYSTEM_PROMPT,
    format_entity,
    get_response_schema,
    get_responses_api_schema,
    parse_response,
    parse_responses_api_response,
    make_error_result,
    uses_responses_api,
    RESPONSES_API_MODELS,
)

load_dotenv()


# --- Few-Shot Prompt Building ---

def format_example(pair: dict, include_reasoning: bool = True) -> str:
    """Format a single example for in-context learning.

    Args:
        pair: Entity pair dict with left, right, judgement
        include_reasoning: Whether to include reasoning text

    Returns:
        Formatted example string
    """
    entity_a = format_entity(pair["left"])
    entity_b = format_entity(pair["right"])
    label = pair["judgement"]

    example = f"""=== EXAMPLE ===
ENTITY A:
{entity_a}

ENTITY B:
{entity_b}

CLASSIFICATION: {label}"""

    if include_reasoning:
        # Generate basic reasoning based on judgement
        if label == "positive":
            example += "\nREASONING: Names are consistent (may be variations/transliterations), no contradictory evidence found. Default to positive when no conflicts exist."
        else:
            example += "\nREASONING: Conflicting evidence found (different dates, identifiers, or other attributes) indicates different entities."

    return example


def build_fewshot_user_prompt(examples: list, test_pair: dict) -> str:
    """Build the user prompt content with few-shot examples.

    Args:
        examples: List of example pairs for in-context learning
        test_pair: The pair to classify

    Returns:
        Formatted user prompt string
    """
    example_text = "\n\n".join(format_example(ex) for ex in examples)

    return f"""Here are some examples of entity matching decisions:

{example_text}

=== NOW CLASSIFY THIS PAIR ===

Compare these two entities:

=== ENTITY A ===
{format_entity(test_pair["left"])}

=== ENTITY B ===
{format_entity(test_pair["right"])}

Search for CONTRADICTIONS. If none found, classify as POSITIVE."""


def build_fewshot_messages(examples: list, test_pair: dict, use_responses_api: bool = False) -> list:
    """Build chat messages with few-shot examples.

    Args:
        examples: List of example pairs for in-context learning
        test_pair: The pair to classify
        use_responses_api: If True, use 'developer' role instead of 'system'

    Returns:
        List of message dicts for the API
    """
    system_role = "developer" if use_responses_api else "system"
    user_prompt = build_fewshot_user_prompt(examples, test_pair)

    return [
        {"role": system_role, "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]


# --- Async Classification ---

async def classify_pair_async(
    client: AsyncOpenAI,
    pair: dict,
    pair_index: int,
    examples: list,
    semaphore: asyncio.Semaphore,
    model: str,
    reasoning_effort: str,
    ternary: bool
) -> dict:
    """Async classify a single entity pair with few-shot examples."""
    async with semaphore:
        try:
            if uses_responses_api(model):
                # Use Responses API for gpt-5.2-pro and similar models
                response = await client.responses.create(
                    model=model,
                    input=build_fewshot_messages(examples, pair, use_responses_api=True),
                    text=get_responses_api_schema(ternary),
                    reasoning={"effort": reasoning_effort},
                    max_output_tokens=4000
                )
                return parse_responses_api_response(response, pair, pair_index)
            else:
                # Use Chat Completions API for other models
                response = await client.chat.completions.create(
                    model=model,
                    messages=build_fewshot_messages(examples, pair),
                    response_format=get_response_schema(ternary),
                    max_completion_tokens=4000,
                    reasoning_effort=reasoning_effort
                )
                return parse_response(response, pair, pair_index)
        except Exception as e:
            return make_error_result(pair, pair_index, e)


async def run_parallel_experiment(
    data: list,
    examples: list,
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
        classify_pair_async(
            client, pair, i, examples, semaphore, model, reasoning_effort, ternary
        )
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
    examples: list,
    model: str,
    reasoning_effort: str,
    ternary: bool
) -> dict:
    """Sync classify a single entity pair."""
    try:
        if uses_responses_api(model):
            # Use Responses API for gpt-5.2-pro and similar models
            response = client.responses.create(
                model=model,
                input=build_fewshot_messages(examples, pair, use_responses_api=True),
                text=get_responses_api_schema(ternary),
                reasoning={"effort": reasoning_effort},
                max_output_tokens=4000
            )
            return parse_responses_api_response(response, pair, pair_index)
        else:
            # Use Chat Completions API for other models
            response = client.chat.completions.create(
                model=model,
                messages=build_fewshot_messages(examples, pair),
                response_format=get_response_schema(ternary),
                max_completion_tokens=4000,
                reasoning_effort=reasoning_effort
            )
            return parse_response(response, pair, pair_index)
    except Exception as e:
        return make_error_result(pair, pair_index, e)


def run_sequential_experiment(
    data: list,
    examples: list,
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
            result = classify_pair_sync(
                client, pair, i, examples, model, reasoning_effort, ternary
            )
            results.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()

    return results


# --- Dry Run ---

def validate_and_preview(config: ExperimentConfig, data: list, examples: list) -> dict:
    """Validate configuration and preview prompts without API calls.

    Args:
        config: Experiment configuration
        data: Test data pairs
        examples: Few-shot examples

    Returns:
        Dict with validation results and previews
    """
    print("\n" + "=" * 60)
    print("DRY RUN MODE - No API calls will be made")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(config.describe())

    print(f"\nFew-shot examples ({len(examples)}):")
    print_example_summary(examples)

    # Preview prompts
    sample_pairs = data[:config.sample_n]
    print(f"\nSample prompts ({len(sample_pairs)} pairs):")

    sample_prompts = []
    total_tokens = 0

    for i, pair in enumerate(sample_pairs):
        messages = build_fewshot_messages(examples, pair)
        prompt_text = messages[0]["content"] + "\n\n" + messages[1]["content"]
        sample_prompts.append(prompt_text)

        # Rough token estimate (4 chars per token)
        tokens = len(prompt_text) // 4
        total_tokens += tokens

        print(f"\n--- Pair {i+1} ---")
        print(f"Left: {pair['left'].get('caption', 'Unknown')}")
        print(f"Right: {pair['right'].get('caption', 'Unknown')}")
        print(f"Ground truth: {pair['judgement']}")
        print(f"Estimated tokens: ~{tokens}")

        if i == 0:
            print(f"\nFull prompt preview (first pair only):")
            print("-" * 40)
            # Show truncated prompt
            if len(prompt_text) > 2000:
                print(prompt_text[:1000])
                print("\n... [truncated] ...\n")
                print(prompt_text[-500:])
            else:
                print(prompt_text)
            print("-" * 40)

    avg_tokens = total_tokens // len(sample_pairs) if sample_pairs else 0
    cost_estimate = config.estimate_cost(len(data))

    print(f"\nToken Estimates:")
    print(f"  Average per pair: ~{avg_tokens}")
    print(f"  Total for {len(data)} pairs: ~{avg_tokens * len(data)}")

    print(f"\nCost Estimate:")
    print(f"  Estimated total: ${cost_estimate['estimated_cost_usd']}")
    print(f"  Per pair: ${cost_estimate['cost_per_pair_usd']}")

    print(f"\nTo run the experiment, remove --dry-run flag")

    return {
        "valid": True,
        "sample_prompts": sample_prompts,
        "avg_tokens_per_pair": avg_tokens,
        "estimated_cost": cost_estimate,
    }


# --- Main ---

def run_experiment(config: ExperimentConfig) -> tuple:
    """Run the full few-shot experiment.

    Args:
        config: Experiment configuration

    Returns:
        Tuple of (metrics_dict, results_list). For dry runs, results_list is empty.
        On error, metrics_dict contains an "error" key.
    """
    # Load data
    data = load_sample(config.input_file)

    # Apply offset and limit
    if config.offset:
        data = data[config.offset:]
    if config.limit:
        data = data[:config.limit]

    # Load examples
    examples = get_examples(
        strategy=config.example_strategy,
        n=config.n_examples,
        seed=config.example_seed,
        input_file=config.input_file
    )

    # Dry run mode
    if config.dry_run:
        validation = validate_and_preview(config, data, examples)
        return validation, []

    # Setup output
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "ternary" if config.ternary else "binary"
    model_safe = config.model.replace(".", "_").replace("-", "_")
    responses_file = output_dir / f"llm_fewshot_{config.n_examples}ex_{mode}_{model_safe}.jsonl"

    api_type = "Responses API" if uses_responses_api(config.model) else "Chat Completions API"
    print(f"\nRunning {mode} few-shot classification on {len(data)} pairs...")
    print(f"Model: {config.model}")
    print(f"API: {api_type}")
    print(f"Reasoning: {config.reasoning_effort}")
    print(f"Examples: {config.n_examples} ({config.example_strategy})")
    print(f"Parallel: {config.parallel}")
    print(f"Output: {responses_file}\n")

    print_example_summary(examples)

    start_time = time.time()

    if config.parallel > 1:
        results = asyncio.run(run_parallel_experiment(
            data, examples, config.model, config.reasoning_effort,
            config.ternary, config.parallel, responses_file
        ))
    else:
        results = run_sequential_experiment(
            data, examples, config.model, config.reasoning_effort,
            config.ternary, responses_file
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
        if config.ternary and r.get("classification") == "uncertain":
            continue
        predictions.append(r["classification"])
        ground_truth.append(r["ground_truth"])

    if not predictions:
        print("No valid predictions to evaluate!")
        return {"error": "No valid predictions"}, results

    metrics = evaluate(ground_truth, predictions)
    total_tokens = sum(r.get("tokens_used", {}).get("total", 0) for r in results if "tokens_used" in r)

    # Track experiment
    exp = Experiment(method="llm_fewshot", input_file=config.input_file)
    exp.set_params(
        model=config.model,
        mode=mode,
        reasoning_effort=config.reasoning_effort,
        n_examples=config.n_examples,
        example_strategy=config.example_strategy,
        parallel=config.parallel,
        total_pairs=len(results),
        evaluated_pairs=len(predictions),
        errors=error_count,
        total_tokens=total_tokens,
        elapsed_seconds=round(elapsed, 2)
    )

    if config.ternary:
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

    # Add responses file reference
    metrics["experiment"] = exp.params
    metrics["experiment"]["responses_file"] = str(responses_file)

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="LLM Few-Shot Entity Resolution")
    add_config_args(parser)

    args = parser.parse_args()

    # Get config (from predefined or args)
    if args.config:
        config = CONFIGS[args.config]
        # Override with explicit args
        if args.input != "data/samples/sample_1000.json":
            config.input_file = args.input
        if args.model != "gpt-5-nano":
            config.model = args.model
        if args.examples != 0:
            config.n_examples = args.examples
        if args.dry_run:
            config.dry_run = True
    else:
        config = ExperimentConfig.from_args(args)

    # Ensure we have examples for few-shot
    if config.n_examples == 0:
        config.n_examples = 8  # Default for few-shot
        print(f"Note: Using default {config.n_examples} examples for few-shot")

    metrics, results = run_experiment(config)

    # Skip cost output for dry runs or errors
    if config.dry_run or "error" in metrics:
        return

    # Print cost estimate
    exp = metrics.get("experiment", {})
    total_tokens = exp.get("total_tokens", 0)
    if total_tokens > 0:
        est_cost = total_tokens * 0.0000005
        print(f"Estimated cost: ${est_cost:.4f}")


if __name__ == "__main__":
    main()
