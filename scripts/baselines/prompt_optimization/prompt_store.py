#!/usr/bin/env python3
"""
Prompt storage and registry management utilities.

Provides functions to save, load, and manage optimized prompts for different
models. Prompts are stored as JSON files with full provenance metadata.

Usage:
    from scripts.baselines.prompt_optimization import (
        save_prompt, load_prompt, get_best_prompt, list_prompts
    )

    # Save an optimized prompt
    save_prompt(prompt_dict, model="llama-8b", optimizer="mipro")

    # Load a specific prompt
    prompt = load_prompt("llama-8b_mipro_20260116_v1")

    # Get the best prompt for a model
    prompt = get_best_prompt("llama-8b")

    # List all prompts
    prompts = list_prompts(model="llama-8b")
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# --- Constants ---

PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "prompts"
REGISTRY_PATH = PROMPTS_DIR / "registry.json"


def _get_git_commit() -> Optional[str]:
    """Get the current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _load_registry() -> Dict[str, Any]:
    """Load the prompt registry from disk."""
    if not REGISTRY_PATH.exists():
        return {"prompts": [], "metadata": {"version": "1.0"}}

    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def _save_registry(registry: Dict[str, Any]) -> None:
    """Save the prompt registry to disk."""
    registry["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d")

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def generate_prompt_id(
    model: str,
    optimizer: str,
    version: int = 1,
) -> str:
    """Generate a unique prompt ID.

    Args:
        model: Model name (e.g., "llama-8b")
        optimizer: Optimizer name (e.g., "mipro", "bootstrap")
        version: Version number

    Returns:
        Unique prompt ID string
    """
    date_str = datetime.now().strftime("%Y%m%d")
    return f"{model}_{optimizer}_{date_str}_v{version}"


def save_prompt(
    prompt_dict: Dict[str, Any],
    model: str,
    optimizer: str,
    description: Optional[str] = None,
    dev_f1: Optional[float] = None,
    version: Optional[int] = None,
) -> str:
    """Save a prompt to disk and update the registry.

    Args:
        prompt_dict: Prompt dictionary with system_prompt, user_template, etc.
        model: Model name (e.g., "llama-8b")
        optimizer: Optimizer name (e.g., "mipro", "bootstrap")
        description: Optional description
        dev_f1: Optional F1 score on dev set
        version: Optional version number (auto-incremented if not provided)

    Returns:
        The prompt_id of the saved prompt
    """
    # Determine version if not provided
    if version is None:
        existing = list_prompts(model=model, optimizer=optimizer)
        version = len(existing) + 1

    # Generate prompt ID
    prompt_id = generate_prompt_id(model, optimizer, version)

    # Build full prompt dict with metadata
    full_prompt = {
        "prompt_id": prompt_id,
        "model": model,
        "description": description or f"{optimizer} optimized prompt for {model}",
        **prompt_dict,
        "provenance": {
            "optimizer": optimizer,
            "dev_f1": dev_f1,
            "optimization_date": datetime.now().strftime("%Y-%m-%d"),
            "git_commit": _get_git_commit(),
            **(prompt_dict.get("provenance", {})),
        },
    }

    # Determine save path
    model_dir = PROMPTS_DIR / model
    model_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{optimizer}_{datetime.now().strftime('%Y%m%d')}_v{version}.json"
    save_path = model_dir / filename

    # Save prompt
    with open(save_path, "w") as f:
        json.dump(full_prompt, f, indent=2)

    print(f"Saved prompt to {save_path}")

    # Update registry
    update_registry(full_prompt, str(save_path.relative_to(PROMPTS_DIR)))

    return prompt_id


def load_prompt(prompt_id_or_path: str) -> Dict[str, Any]:
    """Load a prompt by ID or path.

    Args:
        prompt_id_or_path: Either a prompt ID (e.g., "llama-8b_mipro_20260116_v1")
            or a path to a prompt JSON file

    Returns:
        Prompt dictionary

    Raises:
        FileNotFoundError: If prompt not found
    """
    # Check if it's a direct path
    if os.path.exists(prompt_id_or_path):
        with open(prompt_id_or_path, "r") as f:
            return json.load(f)

    # Check if path is relative to PROMPTS_DIR
    full_path = PROMPTS_DIR / prompt_id_or_path
    if full_path.exists():
        with open(full_path, "r") as f:
            return json.load(f)

    # Look up in registry
    registry = _load_registry()
    for entry in registry.get("prompts", []):
        if entry.get("prompt_id") == prompt_id_or_path:
            path = PROMPTS_DIR / entry["path"]
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)

    raise FileNotFoundError(f"Prompt not found: {prompt_id_or_path}")


def get_best_prompt(
    model: str,
    metric: str = "dev_f1",
) -> Optional[Dict[str, Any]]:
    """Get the best prompt for a model based on a metric.

    Args:
        model: Model name (e.g., "llama-8b")
        metric: Metric to optimize (default: "dev_f1")

    Returns:
        Best prompt dict, or None if no prompts found for model
    """
    prompts = list_prompts(model=model)

    if not prompts:
        # Try default prompts
        prompts = list_prompts(model="all")
        if not prompts:
            return None

    # Sort by metric (descending)
    best = None
    best_score = -1

    for entry in prompts:
        try:
            prompt = load_prompt(entry["prompt_id"])
            score = prompt.get("provenance", {}).get(metric, 0) or 0
            if score > best_score:
                best_score = score
                best = prompt
        except Exception:
            continue

    # If no scored prompts, return the first one
    if best is None and prompts:
        try:
            best = load_prompt(prompts[0]["prompt_id"])
        except Exception:
            pass

    return best


def update_registry(
    prompt_dict: Dict[str, Any],
    path: str,
) -> None:
    """Add or update a prompt entry in the registry.

    Args:
        prompt_dict: Full prompt dictionary
        path: Relative path to the prompt file (from PROMPTS_DIR)
    """
    registry = _load_registry()

    # Check if entry already exists
    prompt_id = prompt_dict["prompt_id"]
    existing_idx = None
    for i, entry in enumerate(registry.get("prompts", [])):
        if entry.get("prompt_id") == prompt_id:
            existing_idx = i
            break

    # Build registry entry
    entry = {
        "prompt_id": prompt_id,
        "model": prompt_dict.get("model", "unknown"),
        "description": prompt_dict.get("description", ""),
        "path": path,
        "created_date": prompt_dict.get("provenance", {}).get(
            "optimization_date", datetime.now().strftime("%Y-%m-%d")
        ),
        "provenance": {
            "optimizer": prompt_dict.get("provenance", {}).get("optimizer"),
            "dev_f1": prompt_dict.get("provenance", {}).get("dev_f1"),
        },
    }

    # Update or append
    if existing_idx is not None:
        registry["prompts"][existing_idx] = entry
    else:
        registry["prompts"].append(entry)

    _save_registry(registry)
    print(f"Updated registry with prompt: {prompt_id}")


def list_prompts(
    model: Optional[str] = None,
    optimizer: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List prompts matching the given criteria.

    Args:
        model: Filter by model name (optional)
        optimizer: Filter by optimizer name (optional)

    Returns:
        List of registry entries matching criteria
    """
    registry = _load_registry()
    prompts = registry.get("prompts", [])

    if model is not None:
        prompts = [p for p in prompts if p.get("model") == model]

    if optimizer is not None:
        prompts = [
            p for p in prompts
            if p.get("provenance", {}).get("optimizer") == optimizer
        ]

    return prompts


def extract_dspy_prompt(
    dspy_module_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract a portable prompt from a saved DSPy module.

    Converts a DSPy compiled module to a portable JSON format that can be
    used without DSPy.

    Args:
        dspy_module_path: Path to saved DSPy module JSON
        output_path: Optional path to save extracted prompt

    Returns:
        Portable prompt dictionary
    """
    with open(dspy_module_path, "r") as f:
        dspy_state = json.load(f)

    # Extract demonstrations from DSPy format
    demonstrations = []
    for key, value in dspy_state.items():
        if "demos" in key.lower() and isinstance(value, list):
            for demo in value:
                if isinstance(demo, dict):
                    demonstrations.append({
                        "entity_a": demo.get("entity_a", ""),
                        "entity_b": demo.get("entity_b", ""),
                        "classification": demo.get("classification", ""),
                        "reasoning": demo.get("reasoning", ""),
                    })

    # Build portable prompt
    prompt_dict = {
        "system_prompt": dspy_state.get("instructions", ""),
        "user_template": "Entity A:\n{entity_a}\n\nEntity B:\n{entity_b}",
        "demonstrations": demonstrations,
        "dspy_state_path": dspy_module_path,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(prompt_dict, f, indent=2)
        print(f"Extracted prompt to {output_path}")

    return prompt_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prompt Store CLI")
    subparsers = parser.add_subparsers(dest="command")

    # List command
    list_parser = subparsers.add_parser("list", help="List prompts")
    list_parser.add_argument("--model", help="Filter by model")
    list_parser.add_argument("--optimizer", help="Filter by optimizer")

    # Load command
    load_parser = subparsers.add_parser("load", help="Load a prompt")
    load_parser.add_argument("prompt_id", help="Prompt ID or path")

    # Best command
    best_parser = subparsers.add_parser("best", help="Get best prompt for model")
    best_parser.add_argument("model", help="Model name")

    args = parser.parse_args()

    if args.command == "list":
        prompts = list_prompts(model=args.model, optimizer=args.optimizer)
        print(f"Found {len(prompts)} prompts:")
        for p in prompts:
            print(f"  - {p['prompt_id']}: {p.get('description', '')[:50]}")
            if p.get("provenance", {}).get("dev_f1"):
                print(f"    Dev F1: {p['provenance']['dev_f1']:.4f}")

    elif args.command == "load":
        prompt = load_prompt(args.prompt_id)
        print(json.dumps(prompt, indent=2))

    elif args.command == "best":
        prompt = get_best_prompt(args.model)
        if prompt:
            print(f"Best prompt for {args.model}: {prompt['prompt_id']}")
            print(json.dumps(prompt, indent=2)[:500] + "...")
        else:
            print(f"No prompts found for {args.model}")

    else:
        parser.print_help()
