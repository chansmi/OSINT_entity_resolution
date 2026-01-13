#!/usr/bin/env python3
"""
Configuration module for entity resolution experiments.

Provides a type-safe ExperimentConfig dataclass and predefined configurations
for common experiments. Supports both CLI and notebook usage.

Usage:
    # CLI: configurations passed via argparse
    python scripts/baselines/llm_experiment.py --model gpt-5.2-pro --reasoning high

    # Notebook: import and use directly
    from scripts.baselines.config import ExperimentConfig, CONFIGS
    config = CONFIGS["gpt52pro"]
    # or
    config = ExperimentConfig(model="gpt-5.2-pro", reasoning_effort="high")
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, List
import json


@dataclass
class ExperimentConfig:
    """Configuration for entity resolution experiments.

    All experiment parameters in one place. Can be:
    - Constructed directly in Python
    - Created from CLI args via from_args()
    - Serialized/deserialized for experiment tracking
    """

    # Input/Output
    input_file: str = "data/samples/sample_1000.json"
    output_dir: str = "data/outputs"

    # Data split
    offset: int = 200  # Skip dev set (first 200 pairs)
    limit: Optional[int] = None  # None = all remaining pairs

    # Model settings
    model: str = "gpt-5-nano"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "medium"

    # Few-shot settings
    n_examples: int = 0  # 0 = zero-shot
    example_strategy: Literal["random", "diverse", "curated"] = "diverse"
    example_seed: int = 42  # Seed for example selection

    # Execution settings
    parallel: int = 20
    ternary: bool = False  # If True, allow "uncertain" classification

    # Dry run (for setup validation)
    dry_run: bool = False  # If True, only validate config, don't call API
    sample_n: int = 5  # For dry run, how many pairs to format (for prompt preview)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_args(cls, args) -> "ExperimentConfig":
        """Create from argparse namespace.

        Maps CLI argument names to config field names.
        """
        mapping = {
            "input": "input_file",
            "output": "output_dir",
            "reasoning": "reasoning_effort",
            "examples": "n_examples",
            "strategy": "example_strategy",
            "dry_run": "dry_run",
        }

        kwargs = {}
        for arg_name, value in vars(args).items():
            if value is None:
                continue
            # Map CLI names to config names
            config_name = mapping.get(arg_name, arg_name)
            if config_name in cls.__dataclass_fields__:
                kwargs[config_name] = value

        return cls(**kwargs)

    def describe(self) -> str:
        """Human-readable description of configuration."""
        lines = [
            f"Model: {self.model}",
            f"Reasoning: {self.reasoning_effort}",
            f"Examples: {self.n_examples} ({self.example_strategy})" if self.n_examples > 0 else "Examples: zero-shot",
            f"Mode: {'ternary' if self.ternary else 'binary'}",
            f"Input: {self.input_file}",
            f"Offset: {self.offset}, Limit: {self.limit or 'all'}",
            f"Parallel: {self.parallel}",
        ]
        if self.dry_run:
            lines.append("DRY RUN MODE (no API calls)")
        return "\n".join(lines)

    def estimate_cost(self, n_pairs: int = 800) -> dict:
        """Estimate cost for running this configuration.

        Returns dict with token and cost estimates.
        """
        # Rough estimates based on typical entity pair sizes
        base_tokens_per_pair = 1500  # System + user prompt
        example_tokens = 300 * self.n_examples  # ~300 tokens per example
        reasoning_multiplier = {
            "none": 0.5,
            "low": 0.8,
            "medium": 1.0,
            "high": 1.5,
            "xhigh": 2.0,
        }.get(self.reasoning_effort, 1.0)

        total_input_tokens = (base_tokens_per_pair + example_tokens) * n_pairs
        total_output_tokens = int(500 * n_pairs * reasoning_multiplier)  # ~500 base output

        # Rough pricing (varies by model)
        pricing = {
            "gpt-5-nano": (0.30, 1.20),  # per 1M tokens (input, output)
            "gpt-5": (1.50, 6.00),
            "gpt-5.2": (2.50, 10.00),
            "gpt-5.2-pro": (15.00, 60.00),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.50, 10.00),
        }

        input_price, output_price = pricing.get(self.model, (1.00, 4.00))
        estimated_cost = (
            total_input_tokens * input_price / 1_000_000
            + total_output_tokens * output_price / 1_000_000
        )

        return {
            "n_pairs": n_pairs,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "estimated_cost_usd": round(estimated_cost, 2),
            "cost_per_pair_usd": round(estimated_cost / n_pairs, 4),
        }


# Predefined configurations for common experiments
CONFIGS = {
    # Baseline (current best)
    "baseline": ExperimentConfig(),

    # Model upgrades
    "gpt52pro": ExperimentConfig(model="gpt-5.2-pro", parallel=10),
    "gpt52": ExperimentConfig(model="gpt-5.2", parallel=15),
    "gpt5": ExperimentConfig(model="gpt-5", parallel=15),

    # Reasoning levels
    "reasoning_none": ExperimentConfig(reasoning_effort="none"),
    "reasoning_low": ExperimentConfig(reasoning_effort="low"),
    "reasoning_medium": ExperimentConfig(reasoning_effort="medium"),
    "reasoning_high": ExperimentConfig(reasoning_effort="high"),
    "reasoning_xhigh": ExperimentConfig(reasoning_effort="xhigh"),

    # Few-shot configurations
    "few_shot_4": ExperimentConfig(n_examples=4),
    "few_shot_6": ExperimentConfig(n_examples=6),
    "few_shot_8": ExperimentConfig(n_examples=8),
    "few_shot_10": ExperimentConfig(n_examples=10),
    "few_shot_curated": ExperimentConfig(n_examples=8, example_strategy="curated"),
    "few_shot_random": ExperimentConfig(n_examples=8, example_strategy="random"),

    # Ternary mode
    "ternary": ExperimentConfig(ternary=True),
    "ternary_gpt52pro": ExperimentConfig(model="gpt-5.2-pro", ternary=True, parallel=10),

    # Quick test (5 pairs)
    "quick_test": ExperimentConfig(limit=5, parallel=5),
    "quick_test_gpt52pro": ExperimentConfig(model="gpt-5.2-pro", limit=5, parallel=5),
}


def add_config_args(parser) -> None:
    """Add standard config arguments to an argparse parser.

    Use this to ensure consistent CLI interface across all experiment scripts.
    """
    parser.add_argument("--input", default="data/samples/sample_1000.json",
                        help="Path to input JSON file")
    parser.add_argument("--output", default="data/outputs",
                        help="Output directory")
    parser.add_argument("--offset", type=int, default=200,
                        help="Skip first N pairs (default: 200 = dev set)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of pairs (default: all)")
    parser.add_argument("--model", default="gpt-5-nano",
                        help="Model to use (gpt-5-nano, gpt-5.2-pro, etc.)")
    parser.add_argument("--reasoning", default="medium",
                        choices=["none", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort level")
    parser.add_argument("--examples", type=int, default=0,
                        help="Number of few-shot examples (0 = zero-shot)")
    parser.add_argument("--strategy", default="diverse",
                        choices=["random", "diverse", "curated"],
                        help="Example selection strategy")
    parser.add_argument("--parallel", type=int, default=20,
                        help="Number of parallel requests")
    parser.add_argument("--ternary", action="store_true",
                        help="Use ternary mode (positive/negative/uncertain)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config and preview prompts without API calls")
    parser.add_argument("--config", default=None,
                        choices=list(CONFIGS.keys()),
                        help="Use predefined configuration")


def get_config_from_args(args) -> ExperimentConfig:
    """Get ExperimentConfig from parsed args, with predefined config support."""
    if args.config:
        # Start with predefined config
        config = CONFIGS[args.config]
        # Override with any explicitly set args
        for key in ["input", "output", "offset", "limit", "model", "reasoning",
                    "examples", "strategy", "parallel", "ternary", "dry_run"]:
            arg_value = getattr(args, key.replace("_", "-").replace("-", "_"), None)
            if arg_value is not None and arg_value != getattr(config, key, None):
                # Arg was explicitly set, override config
                pass  # TODO: implement selective override
        return config
    else:
        return ExperimentConfig.from_args(args)


if __name__ == "__main__":
    # Print available configs when run directly
    print("Available predefined configurations:\n")
    for name, config in CONFIGS.items():
        cost = config.estimate_cost()
        print(f"  {name}:")
        print(f"    Model: {config.model}, Reasoning: {config.reasoning_effort}")
        print(f"    Examples: {config.n_examples} ({config.example_strategy})")
        print(f"    Est. cost (800 pairs): ${cost['estimated_cost_usd']}")
        print()
