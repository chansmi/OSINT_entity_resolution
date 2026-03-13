#!/usr/bin/env python3
"""
Local LLM Entity Resolution Baseline (Zero-Shot and Few-Shot)

Evaluates entity resolution using locally-hosted models (Llama-8B, DeepSeek-14B).
Uses the same prompting strategy as llm_zeroshot.py but with local transformers inference.

Usage:
    # Zero-shot with Llama-8B
    python scripts/baselines/llm_local.py --model llama-8b --input data/samples/sample_1000.json

    # Few-shot with DeepSeek-14B
    python scripts/baselines/llm_local.py --model deepseek-14b --input data/samples/sample_1000.json --shots 4

    # Via Flux (required for GPU access)
    flux run -N 1 -n 1 -g 8 python scripts/baselines/llm_local.py --model llama-8b --input data/samples/sample_1000.json
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.evaluate import evaluate, print_evaluation_report
from scripts.experiment import Experiment
from scripts.baselines.llm_zeroshot import format_entity
from scripts.finetune.config import PRETRAINED_MODELS_DIR, SYSTEM_PROMPT

# Model configurations
MODEL_CONFIGS = {
    "llama-8b": {
        "name": "Llama-3.1-8B-Instruct",
        "path": PRETRAINED_MODELS_DIR / "meta-llama--Llama-3.1-8B-Instruct",
    },
    "deepseek-14b": {
        "name": "DeepSeek-R1-Distill-Qwen-14B",
        "path": PRETRAINED_MODELS_DIR / "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B",
    },
}


def load_data(filepath: str) -> list:
    """Load entity pairs from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'pairs' in data:
        return data['pairs']
    return data


def select_few_shot_examples(
    data: List[Dict[str, Any]],
    n_shots: int,
    seed: int = 42,
    exclude_indices: Optional[set] = None
) -> List[Dict[str, Any]]:
    """Select balanced few-shot examples from the data."""
    random.seed(seed)
    exclude_indices = exclude_indices or set()

    positives = [(i, p) for i, p in enumerate(data)
                 if p.get('judgement') == 'positive' and i not in exclude_indices]
    negatives = [(i, p) for i, p in enumerate(data)
                 if p.get('judgement') == 'negative' and i not in exclude_indices]

    n_pos = n_shots // 2
    n_neg = n_shots - n_pos

    random.shuffle(positives)
    random.shuffle(negatives)

    selected = []
    for _, pair in positives[:n_pos]:
        example = pair.copy()
        example['reasoning'] = "No contradictions found. Same entity."
        selected.append(example)

    for _, pair in negatives[:n_neg]:
        example = pair.copy()
        example['reasoning'] = "Contradictions found in identifying attributes. Different entities."
        selected.append(example)

    random.shuffle(selected)
    return selected


def format_few_shot_examples(examples: List[Dict[str, Any]]) -> str:
    """Format few-shot examples into a string for the prompt."""
    if not examples:
        return ""

    formatted = ["Here are some examples:\n"]
    for i, ex in enumerate(examples, 1):
        formatted.append(f"""Example {i}:
=== ENTITY A ===
{format_entity(ex['left'])}

=== ENTITY B ===
{format_entity(ex['right'])}

Classification: {ex['judgement']}
Reasoning: {ex.get('reasoning', 'Based on available evidence.')}
---""")
    formatted.append("\nNow classify this pair:")
    return "\n".join(formatted)


def build_prompt(pair: dict, few_shot_examples: Optional[List[Dict[str, Any]]] = None) -> str:
    """Build the user prompt for classification."""
    current_pair = f"""Compare these two entities:

=== ENTITY A ===
{format_entity(pair["left"])}

=== ENTITY B ===
{format_entity(pair["right"])}

Search for CONTRADICTIONS. If none found, classify as POSITIVE.
Respond with only "positive" or "negative"."""

    if few_shot_examples:
        few_shot_text = format_few_shot_examples(few_shot_examples)
        return f"{few_shot_text}\n\n{current_pair}"
    return current_pair


def load_model(model_key: str):
    """Load model and tokenizer from local storage."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_key]
    model_path = str(config["path"])

    # CRITICAL: Verify path exists before loading to avoid HuggingFace repo ID validation error
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}\n"
            f"Set PRETRAINED_MODELS_DIR or use --model-path to specify the correct location.\n"
            f"Use: flux run -N 1 -n 1 -g 8 python scripts/baselines/llm_local.py ..."
        )

    print(f"Loading model from {model_path}...")
    print(f"Path exists: {os.path.exists(model_path)}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.eval()
    print(f"Model loaded. Device: {next(model.parameters()).device}")

    return model, tokenizer, config["name"]


def predict_single(
    model,
    tokenizer,
    pair: Dict[str, Any],
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    max_new_tokens: int = 20,
) -> str:
    """Run inference on a single entity pair."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(pair, few_shot_examples)},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()

    if "positive" in response:
        return "positive"
    elif "negative" in response:
        return "negative"
    else:
        # Default to negative if unclear (conservative)
        return "negative"


def run_experiment(
    input_path: str,
    output_dir: str,
    model_key: str = "llama-8b",
    limit: int = None,
    offset: int = 0,
    n_shots: int = 0,
    examples_file: str = None
) -> dict:
    """Run the full evaluation experiment."""
    # Load data
    data = load_data(input_path)

    if offset:
        data = data[offset:]
    if limit:
        data = data[:limit]

    # Prepare few-shot examples
    few_shot_examples = None
    if n_shots > 0:
        examples_source = examples_file or input_path
        examples_data = load_data(examples_source)
        eval_indices = set(range(offset, offset + len(data)))
        few_shot_examples = select_few_shot_examples(
            examples_data, n_shots=n_shots, seed=42, exclude_indices=eval_indices
        )
        print(f"Selected {len(few_shot_examples)} few-shot examples")

    # Load model
    model, tokenizer, model_name = load_model(model_key)

    # Setup output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shot_label = f"{n_shots}shot" if n_shots > 0 else "0shot"
    responses_file = output_dir / f"llm_local_{model_key}_{shot_label}_responses.jsonl"

    print(f"\nRunning classification on {len(data)} pairs...")
    print(f"Model: {model_name}")
    print(f"Few-shot: {n_shots} examples" if n_shots > 0 else "Zero-shot")
    print(f"Output: {responses_file}\n")

    start_time = time.time()

    results = []
    with open(responses_file, 'w') as f:
        for i, pair in enumerate(tqdm(data, desc="Classifying")):
            pred = predict_single(model, tokenizer, pair, few_shot_examples)
            result = {
                "classification": pred,
                "ground_truth": pair["judgement"],
                "pair_index": i,
            }
            results.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/len(data):.2f}s per pair)")

    # Evaluate
    predictions = [r["classification"] for r in results]
    ground_truth = [r["ground_truth"] for r in results]
    metrics = evaluate(ground_truth, predictions)

    # Track experiment
    method_name = f"llm_local_{n_shots}shot" if n_shots > 0 else "llm_local_zeroshot"
    exp = Experiment(method=method_name, input_file=input_path)
    exp.set_params(
        model=model_name,
        model_key=model_key,
        n_shots=n_shots,
        total_pairs=len(results),
        elapsed_seconds=round(elapsed, 2)
    )
    exp.set_metrics(
        accuracy=metrics['accuracy'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1=metrics['f1'],
        confusion_matrix=metrics['confusion_matrix']
    )

    exp.save(str(output_dir))
    exp.print_summary()

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Local LLM Entity Resolution")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                       help="Model to use: llama-8b or deepseek-14b")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", default="data/outputs", help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit number of pairs")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N pairs")
    parser.add_argument("--shots", type=int, default=0, choices=[0, 4, 8],
                       help="Number of few-shot examples (0=zero-shot)")
    parser.add_argument("--examples-file", type=str, default=None,
                       help="Optional file for few-shot examples")

    args = parser.parse_args()

    metrics, results = run_experiment(
        input_path=args.input,
        output_dir=args.output,
        model_key=args.model,
        limit=args.limit,
        offset=args.offset,
        n_shots=args.shots,
        examples_file=args.examples_file
    )

    print(f"\nFinal F1 Score: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
