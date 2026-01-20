#!/usr/bin/env python3
"""
DSPy Entity Resolution Baseline Runner.

Main CLI for running DSPy-based entity resolution experiments.

Usage:
    # Zero-shot baseline (validate against existing 94.95% F1)
    python scripts/baselines/dspy_er/run_dspy.py --model gpt-5-nano --mode zero-shot

    # Bootstrap few-shot optimization
    python scripts/baselines/dspy_er/run_dspy.py --model gpt-5-nano --mode bootstrap --demos 8

    # Bootstrap with teacher-student distillation (improve Llama-8B)
    flux run -N1 -g8 python scripts/baselines/dspy_er/run_dspy.py \
        --model llama-8b --mode bootstrap --teacher gpt-5-nano --demos 8

    # MIPROv2 instruction optimization
    python scripts/baselines/dspy_er/run_dspy.py --model gpt-5-nano --mode mipro --candidates 15

    # Evaluate saved compiled module
    python scripts/baselines/dspy_er/run_dspy.py --load data/outputs/dspy_compiled/bootstrap_8.json

    # Dry run (preview prompts without API calls)
    python scripts/baselines/dspy_er/run_dspy.py --dry-run --mode zero-shot
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import dspy

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.experiment import Experiment

from .data import load_dspy_examples, format_entity, get_raw_pairs
from .metrics import (
    exact_match,
    f1_metric,
    BatchEvaluator,
    classification_accuracy,
    print_metrics,
)
from .modules import get_module, MODULES
from .optimize import (
    configure_lm,
    configure_teacher_student,
    run_bootstrap_fewshot,
    run_bootstrap_random_search,
    run_mipro,
    save_compiled_module,
    load_compiled_module,
    is_local_model,
)

# Import prompt store for exporting optimized prompts
try:
    from ..prompt_optimization import save_prompt, extract_dspy_prompt
    PROMPT_STORE_AVAILABLE = True
except ImportError:
    PROMPT_STORE_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DSPy Entity Resolution Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Zero-shot validation
  python run_dspy.py --model gpt-5-nano --mode zero-shot

  # Bootstrap optimization with 8 demos
  python run_dspy.py --model gpt-5-nano --mode bootstrap --demos 8

  # Teacher-student distillation for Llama-8B
  python run_dspy.py --model llama-8b --teacher gpt-5-nano --mode bootstrap

  # MIPROv2 joint optimization
  python run_dspy.py --model gpt-5-nano --mode mipro --candidates 15
        """,
    )

    # Input/Output
    parser.add_argument("--input", default="data/samples/sample_1000.json",
                        help="Input sample file (default: sample_1000.json)")
    parser.add_argument("--output", default="data/outputs",
                        help="Output directory")
    parser.add_argument("--dev-size", type=int, default=200,
                        help="Dev set size (default: 200)")

    # Model configuration
    parser.add_argument("--model", default="gpt-5-nano",
                        help="Model name (gpt-5-nano, llama-8b, etc.)")
    parser.add_argument("--model-path", default=None,
                        help="Explicit path to model directory (for local models)")
    parser.add_argument("--teacher", default=None,
                        help="Teacher model for distillation (optional)")
    parser.add_argument("--api-base", default=None,
                        help="API base URL for vLLM server mode")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM server instead of direct HuggingFace (for local models)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum tokens to generate (for local models)")

    # Mode selection
    parser.add_argument("--mode", default="zero-shot",
                        choices=["zero-shot", "bootstrap", "bootstrap-search", "mipro"],
                        help="Optimization mode")

    # Module selection
    parser.add_argument("--module", default="cot",
                        choices=list(MODULES.keys()),
                        help="Module type (basic, cot, two_stage, conflict_focused)")

    # Optimization parameters
    parser.add_argument("--demos", type=int, default=8,
                        help="Number of demonstrations for optimization")
    parser.add_argument("--candidates", type=int, default=10,
                        help="Number of candidates for search/mipro")
    parser.add_argument("--threads", type=int, default=4,
                        help="Parallel threads for optimization")

    # Evaluation
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit test set size")

    # Save/Load
    parser.add_argument("--save", default=None,
                        help="Save compiled module to path")
    parser.add_argument("--load", default=None,
                        help="Load compiled module from path")
    parser.add_argument("--export-prompt", action="store_true",
                        help="Export optimized prompt to portable JSON and register")
    parser.add_argument("--eval-demos", type=int, default=None,
                        help="Limit number of demos when evaluating loaded module (for ablation studies)")

    # Debug/Preview
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview prompts without API calls")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")

    return parser.parse_args()


def setup_lm(args):
    """Set up the language model based on arguments.

    Returns either a DSPy LM or HuggingFaceLanguageModel depending on model type.
    """
    # Get model_path from args if provided
    model_path = getattr(args, 'model_path', None)
    use_vllm = getattr(args, 'use_vllm', False)
    max_new_tokens = getattr(args, 'max_new_tokens', 512)

    lm = configure_lm(
        model=args.model,
        api_base=args.api_base,
        temperature=args.temperature,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        use_vllm=use_vllm,
    )
    dspy.configure(lm=lm)
    return lm


def run_zero_shot(args, module: dspy.Module, test_examples: list) -> dict:
    """Run zero-shot evaluation without optimization.

    Args:
        args: Command line arguments
        module: DSPy module to evaluate
        test_examples: Test set examples

    Returns:
        Dict with metrics and results
    """
    print(f"\n{'='*60}")
    print("Running Zero-Shot Evaluation")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Module: {args.module}")
    print(f"Test examples: {len(test_examples)}")

    evaluator = BatchEvaluator(module)
    start_time = time.time()
    results = evaluator.evaluate(test_examples)
    elapsed = time.time() - start_time

    print_metrics(results["metrics"], "Zero-Shot Results")
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/len(test_examples):.2f}s per example)")

    return {
        "mode": "zero-shot",
        "elapsed_seconds": elapsed,
        **results,
    }


def run_bootstrap(args, module: dspy.Module, dev_examples: list, test_examples: list) -> dict:
    """Run bootstrap few-shot optimization.

    Args:
        args: Command line arguments
        module: DSPy module to optimize
        dev_examples: Development set for optimization
        test_examples: Test set for evaluation

    Returns:
        Dict with metrics and results
    """
    print(f"\n{'='*60}")
    print("Running Bootstrap Few-Shot Optimization")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    if args.teacher:
        print(f"Teacher: {args.teacher}")
    print(f"Dev examples: {len(dev_examples)}")
    print(f"Target demos: {args.demos}")

    # Configure teacher if specified
    if args.teacher:
        teacher_lm = configure_lm(
            args.teacher,
            use_vllm=getattr(args, 'use_vllm', False),
        )
        # Use teacher for bootstrap, then switch to student for inference
        dspy.configure(lm=teacher_lm)

    start_time = time.time()
    compiled = run_bootstrap_fewshot(
        module=module,
        trainset=dev_examples,
        metric=f1_metric,
        max_demos=args.demos,
        max_labeled_demos=args.demos,
    )
    optimize_time = time.time() - start_time
    print(f"Optimization time: {optimize_time:.1f}s")

    # Switch back to target model for evaluation
    if args.teacher:
        target_lm = configure_lm(
            args.model,
            api_base=args.api_base,
            model_path=getattr(args, 'model_path', None),
            max_new_tokens=getattr(args, 'max_new_tokens', 512),
            use_vllm=getattr(args, 'use_vllm', False),
        )
        dspy.configure(lm=target_lm)

    # Save if requested
    if args.save:
        save_compiled_module(compiled, args.save)

    # Evaluate
    print("\nEvaluating compiled module...")
    evaluator = BatchEvaluator(compiled)
    start_time = time.time()
    results = evaluator.evaluate(test_examples)
    eval_time = time.time() - start_time

    print_metrics(results["metrics"], "Bootstrap Results")
    print(f"Eval time: {eval_time:.1f}s ({eval_time/len(test_examples):.2f}s per example)")

    return {
        "mode": "bootstrap",
        "optimize_seconds": optimize_time,
        "eval_seconds": eval_time,
        "demos": args.demos,
        "teacher": args.teacher,
        "compiled_module": compiled,
        **results,
    }


def run_bootstrap_search(args, module: dspy.Module, dev_examples: list, test_examples: list) -> dict:
    """Run bootstrap with random search optimization.

    Args:
        args: Command line arguments
        module: DSPy module to optimize
        dev_examples: Development set for optimization
        test_examples: Test set for evaluation

    Returns:
        Dict with metrics and results
    """
    print(f"\n{'='*60}")
    print("Running Bootstrap with Random Search")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Dev examples: {len(dev_examples)}")
    print(f"Candidates: {args.candidates}")

    start_time = time.time()
    compiled = run_bootstrap_random_search(
        module=module,
        trainset=dev_examples,
        metric=f1_metric,
        max_demos=args.demos,
        num_candidate_programs=args.candidates,
        num_threads=args.threads,
    )
    optimize_time = time.time() - start_time
    print(f"Optimization time: {optimize_time:.1f}s")

    if args.save:
        save_compiled_module(compiled, args.save)

    # Evaluate
    print("\nEvaluating compiled module...")
    evaluator = BatchEvaluator(compiled)
    start_time = time.time()
    results = evaluator.evaluate(test_examples)
    eval_time = time.time() - start_time

    print_metrics(results["metrics"], "Bootstrap Search Results")

    return {
        "mode": "bootstrap-search",
        "optimize_seconds": optimize_time,
        "eval_seconds": eval_time,
        "demos": args.demos,
        "candidates": args.candidates,
        "compiled_module": compiled,
        **results,
    }


def run_mipro_optimization(args, module: dspy.Module, dev_examples: list, test_examples: list) -> dict:
    """Run MIPROv2 joint optimization.

    Args:
        args: Command line arguments
        module: DSPy module to optimize
        dev_examples: Development set for optimization
        test_examples: Test set for evaluation

    Returns:
        Dict with metrics and results
    """
    print(f"\n{'='*60}")
    print("Running MIPROv2 Optimization")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Dev examples: {len(dev_examples)}")
    print(f"Candidates: {args.candidates}")

    start_time = time.time()
    compiled = run_mipro(
        module=module,
        trainset=dev_examples,
        metric=f1_metric,
        num_candidates=args.candidates,
        num_threads=args.threads,
        max_demos=args.demos,
    )
    optimize_time = time.time() - start_time
    print(f"Optimization time: {optimize_time:.1f}s")

    if args.save:
        save_compiled_module(compiled, args.save)

    # Evaluate
    print("\nEvaluating compiled module...")
    evaluator = BatchEvaluator(compiled)
    start_time = time.time()
    results = evaluator.evaluate(test_examples)
    eval_time = time.time() - start_time

    print_metrics(results["metrics"], "MIPROv2 Results")

    return {
        "mode": "mipro",
        "optimize_seconds": optimize_time,
        "eval_seconds": eval_time,
        "demos": args.demos,
        "candidates": args.candidates,
        "compiled_module": compiled,
        **results,
    }


def run_dry_run(args, module: dspy.Module, dev_examples: list, test_examples: list) -> None:
    """Preview prompts and configuration without making API calls.

    Args:
        args: Command line arguments
        module: DSPy module to preview
        dev_examples: Development set examples
        test_examples: Test set examples
    """
    print(f"\n{'='*60}")
    print("DRY RUN - Preview Mode")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Module: {args.module}")
    print(f"Mode: {args.mode}")
    print(f"Dev examples: {len(dev_examples)}")
    print(f"Test examples: {len(test_examples)}")

    # Show sample input
    if test_examples:
        ex = test_examples[0]
        print(f"\n--- Sample Input ---")
        print(f"Entity A:\n{ex.entity_a[:500]}...")
        print(f"\nEntity B:\n{ex.entity_b[:500]}...")
        print(f"\nGround Truth: {ex.classification}")

    # Show module structure
    print(f"\n--- Module Structure ---")
    print(f"Type: {type(module).__name__}")
    for name, predictor in module.named_predictors():
        print(f"  {name}: {type(predictor).__name__}")

    # Estimate costs
    n_test = len(test_examples)
    tokens_per_pair = 1500  # Rough estimate
    total_tokens = n_test * tokens_per_pair

    if args.mode == "bootstrap":
        # Bootstrap also processes dev set
        total_tokens += len(dev_examples) * tokens_per_pair * 2

    print(f"\n--- Cost Estimate ---")
    print(f"Estimated tokens: ~{total_tokens:,}")
    print(f"At $0.30/1M tokens (gpt-5-nano): ~${total_tokens * 0.30 / 1_000_000:.2f}")


def export_optimized_prompt(
    args,
    compiled_module: dspy.Module,
    results: dict,
    save_path: Optional[str] = None,
) -> Optional[str]:
    """Export an optimized prompt to the prompt registry.

    Extracts the optimized prompt from a compiled DSPy module and saves it
    to the prompt store for later use.

    Args:
        args: Command line arguments
        compiled_module: Compiled DSPy module with optimized demos
        results: Results dict with metrics
        save_path: Optional path where DSPy module was saved

    Returns:
        Prompt ID if exported successfully, None otherwise
    """
    if not PROMPT_STORE_AVAILABLE:
        print("Warning: prompt_store not available, skipping export")
        return None

    try:
        # Extract demonstrations from the compiled module
        demonstrations = []
        for name, predictor in compiled_module.named_predictors():
            demos = getattr(predictor, 'demos', [])
            for demo in demos:
                if hasattr(demo, '__dict__'):
                    demo_dict = {
                        k: v for k, v in demo.__dict__.items()
                        if not k.startswith('_')
                    }
                else:
                    demo_dict = dict(demo) if isinstance(demo, dict) else {}
                demonstrations.append(demo_dict)

        # Get the instruction from the signature if available
        system_prompt = ""
        for name, predictor in compiled_module.named_predictors():
            sig = getattr(predictor, 'signature', None)
            if sig and hasattr(sig, '__doc__'):
                system_prompt = sig.__doc__ or ""
                break

        # Build the prompt dict
        prompt_dict = {
            "system_prompt": system_prompt.strip(),
            "user_template": "Entity A:\n{entity_a}\n\nEntity B:\n{entity_b}",
            "demonstrations": demonstrations,
            "dspy_module_path": save_path,
        }

        # Get dev F1 from results
        dev_f1 = results.get("metrics", {}).get("f1", 0)

        # Save to prompt store
        from ..prompt_optimization import save_prompt
        prompt_id = save_prompt(
            prompt_dict,
            model=args.model,
            optimizer=args.mode,
            description=f"DSPy {args.mode} optimized prompt for {args.model}",
            dev_f1=dev_f1,
        )

        print(f"\nExported prompt to registry: {prompt_id}")
        return prompt_id

    except Exception as e:
        print(f"Warning: Failed to export prompt: {e}")
        return None


def track_experiment(args, results: dict, output_dir: str) -> None:
    """Track experiment results using the standard tracking system.

    Args:
        args: Command line arguments
        results: Results dict from evaluation
        output_dir: Output directory
    """
    method = f"dspy_{args.mode.replace('-', '_')}"
    exp = Experiment(method=method, input_file=args.input)

    exp.set_params(
        model=args.model,
        module=args.module,
        mode=args.mode,
        demos=args.demos if args.mode != "zero-shot" else 0,
        teacher=args.teacher,
        candidates=args.candidates if args.mode in ["bootstrap-search", "mipro"] else None,
        dev_size=args.dev_size,
        test_size=results["metrics"].get("n_evaluated", 0),
    )

    metrics = results["metrics"]
    exp.set_metrics(
        accuracy=metrics.get("accuracy", 0),
        precision=metrics.get("precision", 0),
        recall=metrics.get("recall", 0),
        f1=metrics.get("f1", 0),
        confusion_matrix=metrics.get("confusion_matrix", {}),
    )

    if "optimize_seconds" in results:
        exp.set_params(optimize_seconds=results["optimize_seconds"])
    if "eval_seconds" in results:
        exp.set_params(eval_seconds=results["eval_seconds"])

    result_file = exp.save(output_dir)
    exp.print_summary()

    return exp


def main():
    args = parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    dev_examples, test_examples = load_dspy_examples(
        args.input,
        dev_size=args.dev_size,
    )

    if args.limit:
        test_examples = test_examples[:args.limit]

    print(f"Dev set: {len(dev_examples)} examples")
    print(f"Test set: {len(test_examples)} examples")

    # Handle dry run
    if args.dry_run:
        module = get_module(args.module)
        run_dry_run(args, module, dev_examples, test_examples)
        return

    # Set up language model
    setup_lm(args)

    # Load or create module
    if args.load:
        print(f"Loading compiled module from {args.load}...")
        module_class = MODULES[args.module]
        module = load_compiled_module(module_class, args.load)

        # Limit demos if --eval-demos specified (for ablation studies)
        if args.eval_demos is not None:
            print(f"Limiting demos to {args.eval_demos} for evaluation...")
            for name, predictor in module.named_predictors():
                if hasattr(predictor, 'demos'):
                    original_count = len(predictor.demos)
                    predictor.demos = predictor.demos[:args.eval_demos]
                    print(f"  {name}: {original_count} -> {len(predictor.demos)} demos")
    else:
        module = get_module(args.module)

    # Run appropriate mode
    if args.mode == "zero-shot":
        results = run_zero_shot(args, module, test_examples)
    elif args.mode == "bootstrap":
        results = run_bootstrap(args, module, dev_examples, test_examples)
    elif args.mode == "bootstrap-search":
        results = run_bootstrap_search(args, module, dev_examples, test_examples)
    elif args.mode == "mipro":
        results = run_mipro_optimization(args, module, dev_examples, test_examples)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Export optimized prompt if requested and we have a compiled module
    if getattr(args, 'export_prompt', False) and args.mode != "zero-shot":
        compiled_module = results.get("compiled_module")
        if compiled_module:
            export_optimized_prompt(
                args,
                compiled_module,
                results,
                save_path=args.save,
            )
        else:
            print("Warning: No compiled module available to export")

    # Track experiment
    track_experiment(args, results, args.output)

    # Print final summary
    metrics = results["metrics"]
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"F1:        {metrics.get('f1', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
