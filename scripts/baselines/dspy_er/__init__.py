"""
DSPy Entity Resolution Module.

Provides DSPy-based entity resolution baselines with automatic prompt optimization.

Main components:
- signatures: Typed input/output signatures (EntityResolution, etc.)
- modules: DSPy modules (EntityResolver, ChainOfThoughtResolver, etc.)
- data: Data loading utilities (load_dspy_examples, format_entity)
- metrics: Evaluation metrics (exact_match, f1_metric, BatchEvaluator)
- optimize: Optimizer wrappers (bootstrap, mipro, etc.)
- run_dspy: Main CLI runner

Usage:
    # From command line
    python -m scripts.baselines.dspy_er.run_dspy --model gpt-5-nano --mode zero-shot

    # From Python
    from scripts.baselines.dspy_er import (
        EntityResolution,
        ChainOfThoughtResolver,
        load_dspy_examples,
        BatchEvaluator,
    )
"""

from .signatures import (
    EntityResolution,
    EntityResolutionWithConfidence,
    EntityResolutionTernary,
    ConflictDetection,
)

from .modules import (
    EntityResolver,
    ChainOfThoughtResolver,
    TwoStageResolver,
    ConflictFocusedResolver,
    get_module,
    MODULES,
)

from .data import (
    format_entity,
    pair_to_example,
    load_dspy_examples,
    load_all_examples,
    get_raw_pairs,
)

from .metrics import (
    exact_match,
    f1_metric,
    classification_accuracy,
    BatchEvaluator,
    print_metrics,
)

from .optimize import (
    configure_lm,
    configure_teacher_student,
    run_bootstrap_fewshot,
    run_bootstrap_random_search,
    run_mipro,
    save_compiled_module,
    load_compiled_module,
    OPTIMIZERS,
)

__all__ = [
    # Signatures
    "EntityResolution",
    "EntityResolutionWithConfidence",
    "EntityResolutionTernary",
    "ConflictDetection",
    # Modules
    "EntityResolver",
    "ChainOfThoughtResolver",
    "TwoStageResolver",
    "ConflictFocusedResolver",
    "get_module",
    "MODULES",
    # Data
    "format_entity",
    "pair_to_example",
    "load_dspy_examples",
    "load_all_examples",
    "get_raw_pairs",
    # Metrics
    "exact_match",
    "f1_metric",
    "classification_accuracy",
    "BatchEvaluator",
    "print_metrics",
    # Optimize
    "configure_lm",
    "configure_teacher_student",
    "run_bootstrap_fewshot",
    "run_bootstrap_random_search",
    "run_mipro",
    "save_compiled_module",
    "load_compiled_module",
    "OPTIMIZERS",
]
