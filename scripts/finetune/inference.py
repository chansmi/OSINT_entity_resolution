#!/usr/bin/env python3
"""
Inference script for fine-tuned entity resolution models.

Loads a base model with a LoRA adapter and runs evaluation on test data.
Uses the universal test set (test_pairs.json) that's compatible with ALL methods.

Usage:
    # Evaluate fine-tuned model on universal test set
    python scripts/finetune/inference.py \
        --model llama-8b \
        --adapter-path ./checkpoints/llama-8b_*/final \
        --input data/processed/test_pairs.json

    # Quick test with limited samples
    python scripts/finetune/inference.py \
        --model llama-8b \
        --adapter-path ./checkpoint/final \
        --input data/processed/test_pairs.json \
        --limit 100
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.finetune.config import get_model_config, SYSTEM_PROMPT
from scripts.finetune.data_pipeline import format_entity_for_prompt, compute_file_checksum
from scripts.evaluate import evaluate, print_evaluation_report
from scripts.experiment import Experiment


def verify_test_set_checksum(input_path: str, metadata_path: Optional[str] = None) -> str:
    """
    Verify test set integrity and return checksum.

    Args:
        input_path: Path to test set file
        metadata_path: Optional path to metadata.json with expected checksum

    Returns:
        SHA256 checksum of the test set

    Raises:
        ValueError: If checksum doesn't match expected value
    """
    actual_checksum = compute_file_checksum(input_path)

    # Try to find expected checksum from metadata
    if metadata_path is None:
        metadata_path = Path(input_path).parent / "metadata.json"

    if Path(metadata_path).exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        if "test_set_checksum" in metadata:
            expected_checksum = metadata["test_set_checksum"]
            if actual_checksum != expected_checksum:
                raise ValueError(
                    f"Test set checksum mismatch!\n"
                    f"  Expected: {expected_checksum}\n"
                    f"  Actual:   {actual_checksum}\n"
                    f"  This may indicate the test set was modified."
                )
            print(f"✓ Test set checksum verified: {actual_checksum[:16]}...")
    else:
        print(f"Test set checksum: {actual_checksum}")

    return actual_checksum


def load_test_pairs_v2(input_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load test pairs from V2 JSON format (universal format for all methods).

    Supports both:
    - V2 format with metadata: {"metadata": {...}, "pairs": [...]}
    - Plain JSON array: [...]
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    if 'pairs' in data:
        # V2 format with metadata
        pairs = data['pairs']
        print(f"Loaded V2 format test set with metadata")
        if 'metadata' in data:
            meta = data['metadata']
            print(f"  Description: {meta.get('description', 'N/A')}")
            print(f"  Pairs: {meta.get('n_pairs', len(pairs)):,}")
    else:
        # Plain JSON array
        pairs = data

    if limit:
        pairs = pairs[:limit]

    return pairs


def load_model_with_adapter(
    model_path: str,
    adapter_path: str,
    use_4bit: bool = True,
):
    """Load base model and apply LoRA adapter."""

    print(f"Loading base model from {model_path}...")

    # QLoRA quantization config (must match training)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model_kwargs = {
        "local_files_only": True,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }

    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    # Load LoRA adapter
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)

    # Merge adapter for faster inference (optional, but recommended)
    # model = model.merge_and_unload()  # Uncomment if you want to merge

    model.eval()

    return model, tokenizer


def create_prompt(pair: Dict[str, Any]) -> str:
    """Create inference prompt for an entity pair."""
    user_content = f"""Compare these two entities:

=== ENTITY A ===
{format_entity_for_prompt(pair['left'])}

=== ENTITY B ===
{format_entity_for_prompt(pair['right'])}

Are these the same entity?"""

    return user_content


def predict_single(
    model,
    tokenizer,
    pair: Dict[str, Any],
    max_new_tokens: int = 10,
) -> str:
    """
    Run inference on a single entity pair.

    Returns:
        Predicted label: 'positive' or 'negative'
    """
    # Create messages in chat format
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": create_prompt(pair)},
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()

    # Parse response
    if "positive" in response:
        return "positive"
    elif "negative" in response:
        return "negative"
    else:
        # Default to negative if unclear (conservative)
        print(f"  Warning: Unclear response '{response}', defaulting to 'negative'")
        return "negative"


def load_test_data(input_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load test data from JSONL file."""
    data = []
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            example = json.loads(line)
            # Extract pair info from messages
            # The messages format has: system, user, assistant
            # We need to reconstruct the pair for evaluation
            data.append(example)
    return data


def load_raw_test_pairs(input_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load raw test pairs that still have left/right entity structure."""
    # Try to load from the original test data
    test_pairs_path = Path(input_path).parent / "test_pairs.jsonl"

    if test_pairs_path.exists():
        data = []
        with open(test_pairs_path, 'r') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                data.append(json.loads(line))
        return data

    # Otherwise, we need to load from raw data
    print(f"Note: Loading raw pairs from data/raw/ for evaluation...")
    raw_path = Path("data/raw/pairs-20251209.json.gz")
    if raw_path.exists():
        import gzip
        data = []
        with gzip.open(raw_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                data.append(json.loads(line))
        return data

    raise FileNotFoundError(
        f"Cannot find test pairs. Expected either:\n"
        f"  - {test_pairs_path}\n"
        f"  - {raw_path}"
    )


def run_evaluation(
    model,
    tokenizer,
    test_pairs: List[Dict[str, Any]],
    batch_progress: bool = True,
) -> Dict[str, Any]:
    """
    Run evaluation on test pairs (raw V2 format with left/right/judgement).

    Args:
        model: Fine-tuned model
        tokenizer: Model tokenizer
        test_pairs: List of pairs with 'left', 'right', 'judgement' keys
        batch_progress: Whether to show progress bar

    Returns:
        Evaluation results from scripts/evaluate.py
    """
    ground_truth = []
    predictions = []

    iterator = tqdm(test_pairs, desc="Evaluating") if batch_progress else test_pairs

    for pair in iterator:
        # Get ground truth from the judgement field
        gt_label = pair['judgement'].strip().lower()
        ground_truth.append(gt_label)

        # Run inference on raw pair (converts to ChatML on-the-fly)
        pred_label = predict_single(model, tokenizer, pair)
        predictions.append(pred_label)

    # Calculate metrics
    results = evaluate(ground_truth, predictions)

    return results


def predict_single_from_formatted(
    model,
    tokenizer,
    formatted_user_content: str,
    max_new_tokens: int = 10,
) -> str:
    """Run inference with pre-formatted user content."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": formatted_user_content},
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
        max_length=1024,
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
        return "negative"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned entity resolution model"
    )
    parser.add_argument(
        "--model",
        choices=["llama-8b", "deepseek-14b"],
        required=True,
        help="Model type (for config lookup)"
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--input",
        default="data/processed/test_pairs.json",
        help="Path to test data (V2 JSON format, universal for all methods)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test samples"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save predictions JSONL"
    )

    args = parser.parse_args()

    # Get model config
    config = get_model_config(args.model)
    model_path = str(config.model_path)

    print("=" * 60)
    print(f"Entity Resolution Inference: {config.name}")
    print("=" * 60)
    print(f"Base model: {model_path}")
    print(f"Adapter: {args.adapter_path}")
    print(f"Test data: {args.input}")
    print(f"Limit: {args.limit or 'None'}")
    print()

    # Load model with adapter
    model, tokenizer = load_model_with_adapter(
        model_path,
        args.adapter_path,
        use_4bit=not args.no_4bit,
    )

    print(f"\nModel loaded successfully")
    print(f"Device: {next(model.parameters()).device}")
    print()

    # Verify test set integrity
    print("Verifying test set integrity...")
    verify_test_set_checksum(args.input)
    print()

    # Load test pairs (V2 format with left/right/judgement)
    print("Loading test data...")
    test_pairs = load_test_pairs_v2(args.input, limit=args.limit)
    print(f"Loaded {len(test_pairs)} test pairs")
    print()

    # Run evaluation
    print("Running evaluation...")
    results = run_evaluation(model, tokenizer, test_pairs)

    # Print results
    print_evaluation_report(results)

    # Log experiment
    exp = Experiment(
        method=f"finetune_{args.model}_eval",
        input_file=args.input,
    )
    exp.set_params(
        model=config.name,
        adapter_path=args.adapter_path,
        test_samples=len(test_pairs),
        use_4bit=not args.no_4bit,
    )
    exp.set_metrics(
        accuracy=results['accuracy'],
        precision=results['precision'],
        recall=results['recall'],
        f1=results['f1'],
        true_positive=results['confusion_matrix']['true_positive'],
        true_negative=results['confusion_matrix']['true_negative'],
        false_positive=results['confusion_matrix']['false_positive'],
        false_negative=results['confusion_matrix']['false_negative'],
    )
    exp.save()
    exp.print_summary()

    print(f"\n{'=' * 60}")
    print("Evaluation Complete!")
    print(f"{'=' * 60}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")

    return results


if __name__ == "__main__":
    main()
