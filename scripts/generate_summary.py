#!/usr/bin/env python3
"""
Generate Summary Report for Entity Resolution Evaluations

Parses experiments.jsonl and generates a markdown summary with comparison tables.

Usage:
    python scripts/generate_summary.py
    python scripts/generate_summary.py --filter-method llm  # Filter by method prefix
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

EXPERIMENTS_FILE = Path("data/outputs/experiments.jsonl")
OUTPUT_FILE = Path("data/outputs/evaluation_summary.md")


def load_experiments(method_filter: str = None) -> List[Dict[str, Any]]:
    """Load experiments from the registry."""
    if not EXPERIMENTS_FILE.exists():
        return []

    experiments = []
    with open(EXPERIMENTS_FILE) as f:
        for line in f:
            if line.strip():
                exp = json.loads(line)
                if method_filter:
                    if method_filter.lower() in exp.get("method", "").lower():
                        experiments.append(exp)
                else:
                    experiments.append(exp)

    return experiments


def get_model_display_name(exp: Dict[str, Any]) -> str:
    """Extract display name for the model."""
    params = exp.get("params", {})
    model = params.get("model", "Unknown")

    # Shorten common model names
    if "gpt-4.1" in model.lower():
        return "GPT-4.1"
    elif "gpt-5" in model.lower():
        return model.replace("gpt-", "GPT-")
    elif "llama" in model.lower():
        return "Llama-8B"
    elif "deepseek" in model.lower():
        return "DeepSeek-14B"
    return model


def get_config_description(exp: Dict[str, Any]) -> str:
    """Generate a configuration description."""
    params = exp.get("params", {})
    method = exp.get("method", "")

    parts = []

    # Shot type
    n_shots = params.get("n_shots", 0)
    if n_shots > 0:
        parts.append(f"{n_shots}-shot")
    else:
        parts.append("0-shot")

    # Reasoning (only for OpenAI models)
    reasoning = params.get("reasoning_effort", "")
    if reasoning and "local" not in method:
        parts.append(f"reasoning={reasoning}")

    return ", ".join(parts)


def generate_markdown(experiments: List[Dict[str, Any]]) -> str:
    """Generate markdown summary from experiments."""
    lines = [
        "# Entity Resolution Evaluation Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Results Overview",
        "",
        "| Model | Configuration | F1 | Precision | Recall | Accuracy |",
        "|-------|---------------|---:|----------:|-------:|---------:|",
    ]

    # Sort by F1 score descending
    sorted_exps = sorted(
        experiments,
        key=lambda x: x.get("metrics", {}).get("f1", 0),
        reverse=True
    )

    for exp in sorted_exps:
        metrics = exp.get("metrics", {})
        model = get_model_display_name(exp)
        config = get_config_description(exp)

        f1 = metrics.get("f1", 0) * 100
        precision = metrics.get("precision", 0) * 100
        recall = metrics.get("recall", 0) * 100
        accuracy = metrics.get("accuracy", 0) * 100

        lines.append(
            f"| {model} | {config} | {f1:.2f}% | {precision:.2f}% | {recall:.2f}% | {accuracy:.2f}% |"
        )

    # Add detailed sections
    lines.extend([
        "",
        "## Best Configuration",
        "",
    ])

    if sorted_exps:
        best = sorted_exps[0]
        metrics = best.get("metrics", {})
        params = best.get("params", {})

        lines.extend([
            f"**Model:** {get_model_display_name(best)}",
            f"**Configuration:** {get_config_description(best)}",
            f"**F1 Score:** {metrics.get('f1', 0) * 100:.2f}%",
            "",
        ])

        # Confusion matrix
        cm = metrics.get("confusion_matrix", {})
        if cm:
            lines.extend([
                "**Confusion Matrix:**",
                f"- True Positives: {cm.get('true_positive', 0)}",
                f"- True Negatives: {cm.get('true_negative', 0)}",
                f"- False Positives: {cm.get('false_positive', 0)}",
                f"- False Negatives: {cm.get('false_negative', 0)}",
                "",
            ])

    # Group by model
    lines.extend([
        "## Results by Model",
        "",
    ])

    models = {}
    for exp in experiments:
        model = get_model_display_name(exp)
        if model not in models:
            models[model] = []
        models[model].append(exp)

    for model, exps in sorted(models.items()):
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| Configuration | F1 | Precision | Recall |")
        lines.append("|---------------|---:|----------:|-------:|")

        for exp in sorted(exps, key=lambda x: x.get("metrics", {}).get("f1", 0), reverse=True):
            metrics = exp.get("metrics", {})
            config = get_config_description(exp)
            f1 = metrics.get("f1", 0) * 100
            precision = metrics.get("precision", 0) * 100
            recall = metrics.get("recall", 0) * 100
            lines.append(f"| {config} | {f1:.2f}% | {precision:.2f}% | {recall:.2f}% |")

        lines.append("")

    # Add experiment details
    lines.extend([
        "## Experiment Details",
        "",
    ])

    for exp in sorted_exps[:10]:  # Show top 10
        run_id = exp.get("run_id", "unknown")
        timestamp = exp.get("timestamp", "")
        params = exp.get("params", {})
        metrics = exp.get("metrics", {})

        lines.extend([
            f"### {run_id}",
            f"- **Timestamp:** {timestamp}",
            f"- **Model:** {params.get('model', 'N/A')}",
            f"- **Total Pairs:** {params.get('total_pairs', 'N/A')}",
            f"- **Elapsed:** {params.get('elapsed_seconds', 'N/A')}s",
            f"- **F1:** {metrics.get('f1', 0) * 100:.2f}%",
            "",
        ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation summary")
    parser.add_argument("--filter-method", type=str, default=None,
                       help="Filter experiments by method name (e.g., 'llm')")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE),
                       help="Output markdown file path")

    args = parser.parse_args()

    print(f"Loading experiments from {EXPERIMENTS_FILE}...")
    experiments = load_experiments(method_filter=args.filter_method)

    if not experiments:
        print("No experiments found!")
        return

    print(f"Found {len(experiments)} experiments")

    markdown = generate_markdown(experiments)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(markdown)

    print(f"Summary written to {output_path}")

    # Print quick summary to console
    print("\n" + "=" * 60)
    print("TOP 5 RESULTS")
    print("=" * 60)

    sorted_exps = sorted(
        experiments,
        key=lambda x: x.get("metrics", {}).get("f1", 0),
        reverse=True
    )[:5]

    for i, exp in enumerate(sorted_exps, 1):
        metrics = exp.get("metrics", {})
        model = get_model_display_name(exp)
        config = get_config_description(exp)
        f1 = metrics.get("f1", 0) * 100
        print(f"{i}. {model} ({config}): F1={f1:.2f}%")


if __name__ == "__main__":
    main()
