#!/usr/bin/env python3
"""
DSPy Optimizer configurations for entity resolution.

Provides wrapper functions for different DSPy optimizers:
- BootstrapFewShot: Basic few-shot example optimization
- BootstrapFewShotWithRandomSearch: Enhanced with random search
- MIPROv2: Joint instruction and example optimization
- SIMBA: Self-improving with failure analysis

Also provides LM configuration for both API-based models (OpenAI, Anthropic via LiteLLM)
and local HuggingFace models via the HuggingFaceLanguageModel wrapper.
"""

import os
from pathlib import Path
from typing import List, Optional, Callable, Union

from dotenv import load_dotenv
import dspy

# Load environment variables from .env file
load_dotenv()
from dspy.teleprompt import (
    BootstrapFewShot,
    BootstrapFewShotWithRandomSearch,
    MIPROv2,
)

from .metrics import exact_match, f1_metric


# List of model name patterns that should use local HuggingFace inference
LOCAL_MODEL_PATTERNS = ["llama", "deepseek", "qwen", "mixtral"]


def is_local_model(model: str) -> bool:
    """Check if a model should use local HuggingFace inference.

    Args:
        model: Model name

    Returns:
        True if model should use local inference
    """
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in LOCAL_MODEL_PATTERNS)


def configure_lm(
    model: str = "gpt-5-nano",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    model_path: Optional[str] = None,
    max_new_tokens: int = 512,
    use_vllm: bool = False,
) -> Union[dspy.LM, "HuggingFaceLanguageModel"]:
    """Configure a DSPy language model.

    For local models (Llama, DeepSeek, etc.), uses HuggingFace transformers
    directly by default. Can optionally use vLLM server if use_vllm=True.

    For API models (OpenAI, Anthropic via LiteLLM), uses the standard DSPy
    OpenAI-compatible integration. Supports provider prefixes like:
    - "anthropic/claude-opus-4-5-20251101" -> Anthropic via LiteLLM
    - "openai/gpt-5.2-pro" -> OpenAI via LiteLLM

    Args:
        model: Model name (e.g., "gpt-5-nano", "llama-8b", "anthropic/claude-opus-4-5-20251101")
        api_base: Optional API base URL (defaults to LITELLM_BASE_URL env var)
        api_key: Optional API key (defaults to provider-specific env var)
        temperature: Sampling temperature
        model_path: Optional explicit path to model (for local models)
        max_new_tokens: Maximum tokens to generate (for local models)
        use_vllm: If True, use vLLM server instead of direct HuggingFace

    Returns:
        Configured DSPy LM instance (or HuggingFaceLanguageModel for local)
    """
    # Handle local models (check before stripping prefix)
    if is_local_model(model):
        if use_vllm:
            # Use vLLM server (requires separate server process)
            base_url = api_base or "http://localhost:8000/v1"
            return dspy.LM(
                model=f"openai/{model}",
                api_base=base_url,
                api_key="dummy",  # vLLM doesn't need real key
                temperature=temperature,
            )
        else:
            # Use HuggingFace transformers directly (default)
            from .hf_lm import HuggingFaceLanguageModel, get_model_path

            # Resolve model path
            if model_path is None:
                model_path = get_model_path(model)

            return HuggingFaceLanguageModel(
                model_path=model_path,
                model_name=model,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

    # API models - detect provider from prefix or default to OpenAI
    provider = "openai"  # default
    model_name = model

    if model.startswith("anthropic/"):
        provider = "anthropic"
        model_name = model[len("anthropic/"):]
    elif model.startswith("openai/"):
        provider = "openai"
        model_name = model[len("openai/"):]

    # Get API base URL (use LiteLLM endpoint by default)
    if api_base is None:
        api_base = os.environ.get("LITELLM_BASE_URL")

    # Get API key from environment if not provided
    if api_key is None:
        if provider == "anthropic":
            api_key = os.environ.get("LITELLM_API_KEY_ANTHROPIC")
        else:
            api_key = os.environ.get("LITELLM_API_KEY_OPENAI")

    # Build the model string for DSPy
    # When using LiteLLM proxy with OpenAI-compatible API, use "openai/" prefix
    # and pass the provider/model as the model name (LiteLLM routes by model name)
    if api_base:
        # Using LiteLLM proxy - use OpenAI format, LiteLLM routes by model name
        # The model name should be "anthropic/claude-opus-4-5-20251101" for LiteLLM routing
        dspy_model = f"openai/{provider}/{model_name}"
    else:
        # Direct API - use appropriate provider prefix
        dspy_model = f"{provider}/{model_name}"

    # Build kwargs for dspy.LM
    lm_kwargs = {
        "model": dspy_model,
    }

    # GPT-5 models don't support temperature=0.0, only temperature=1.0
    # For these models, we omit the temperature parameter entirely
    if "gpt-5" in model_name.lower() and not "gpt-5.1" in model_name.lower():
        # GPT-5 base models: don't set temperature (defaults to 1.0)
        pass
    else:
        lm_kwargs["temperature"] = temperature

    if api_base:
        lm_kwargs["api_base"] = api_base
    if api_key:
        lm_kwargs["api_key"] = api_key

    print(f"Configuring LM: model={dspy_model}, api_base={api_base or 'default'}")
    return dspy.LM(**lm_kwargs)


def configure_teacher_student(
    teacher_model: str = "gpt-5-nano",
    student_model: str = "llama-8b",
    teacher_api_base: Optional[str] = None,
    student_api_base: Optional[str] = None,
    teacher_model_path: Optional[str] = None,
    student_model_path: Optional[str] = None,
    use_vllm: bool = False,
) -> tuple:
    """Configure teacher and student LMs for distillation.

    Args:
        teacher_model: Teacher model name (typically larger/API model)
        student_model: Student model name (typically smaller/local model)
        teacher_api_base: API base for teacher (if using vLLM)
        student_api_base: API base for student (if using vLLM)
        teacher_model_path: Explicit path for teacher model
        student_model_path: Explicit path for student model
        use_vllm: If True, use vLLM server for local models

    Returns:
        Tuple of (teacher_lm, student_lm)
    """
    teacher = configure_lm(
        teacher_model,
        api_base=teacher_api_base,
        model_path=teacher_model_path,
        use_vllm=use_vllm,
    )
    student = configure_lm(
        student_model,
        api_base=student_api_base,
        model_path=student_model_path,
        use_vllm=use_vllm,
    )
    return teacher, student


def run_bootstrap_fewshot(
    module: dspy.Module,
    trainset: List[dspy.Example],
    metric: Callable = exact_match,
    max_demos: int = 8,
    max_labeled_demos: int = 8,
    max_rounds: int = 1,
    max_errors: int = 5,
) -> dspy.Module:
    """Run BootstrapFewShot optimization.

    Finds effective few-shot demonstrations from the trainset.

    Args:
        module: DSPy module to optimize
        trainset: Training examples (dev set)
        metric: Metric function for optimization
        max_demos: Maximum number of demonstrations to include
        max_labeled_demos: Maximum labeled demos
        max_rounds: Number of bootstrap rounds
        max_errors: Maximum errors before stopping

    Returns:
        Optimized module with compiled demonstrations
    """
    optimizer = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=max_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=max_rounds,
        max_errors=max_errors,
    )

    print(f"Running BootstrapFewShot with {len(trainset)} examples...")
    print(f"  max_demos: {max_demos}")
    print(f"  max_rounds: {max_rounds}")

    compiled = optimizer.compile(module, trainset=trainset)
    return compiled


def run_bootstrap_random_search(
    module: dspy.Module,
    trainset: List[dspy.Example],
    valset: Optional[List[dspy.Example]] = None,
    metric: Callable = exact_match,
    max_demos: int = 8,
    num_candidate_programs: int = 10,
    num_threads: int = 4,
) -> dspy.Module:
    """Run BootstrapFewShot with random search for better results.

    Generates multiple candidate programs and selects the best.

    Args:
        module: DSPy module to optimize
        trainset: Training examples
        valset: Validation set (if None, uses subset of trainset)
        metric: Metric function for optimization
        max_demos: Maximum demonstrations per candidate
        num_candidate_programs: Number of candidate programs to try
        num_threads: Parallel threads for evaluation

    Returns:
        Best optimized module
    """
    if valset is None:
        # Split trainset 80/20 for train/val
        split_idx = int(len(trainset) * 0.8)
        valset = trainset[split_idx:]
        trainset = trainset[:split_idx]

    optimizer = BootstrapFewShotWithRandomSearch(
        metric=metric,
        max_bootstrapped_demos=max_demos,
        num_candidate_programs=num_candidate_programs,
        num_threads=num_threads,
    )

    print(f"Running BootstrapFewShotWithRandomSearch...")
    print(f"  trainset: {len(trainset)} examples")
    print(f"  valset: {len(valset)} examples")
    print(f"  num_candidates: {num_candidate_programs}")

    compiled = optimizer.compile(module, trainset=trainset, valset=valset)
    return compiled


def run_mipro(
    module: dspy.Module,
    trainset: List[dspy.Example],
    valset: Optional[List[dspy.Example]] = None,
    metric: Callable = exact_match,
    num_candidates: int = 15,
    init_temperature: float = 1.0,
    num_threads: int = 4,
    max_demos: int = 8,
    auto: Optional[str] = None,
) -> dspy.Module:
    """Run MIPROv2 optimization (joint instruction + example optimization).

    MIPROv2 uses Bayesian optimization to find both good instructions
    and demonstrations. This is the most powerful optimizer but also
    the most expensive.

    Args:
        module: DSPy module to optimize
        trainset: Training examples
        valset: Validation set (if None, uses subset of trainset)
        metric: Metric function for optimization
        num_candidates: Number of instruction candidates to generate (ignored if auto is set)
        init_temperature: Temperature for candidate generation
        num_threads: Parallel threads
        max_demos: Maximum demonstrations
        auto: Auto mode ("light", "medium", "heavy") - if set, overrides num_candidates

    Returns:
        Optimized module with best instructions and demos
    """
    if valset is None:
        split_idx = int(len(trainset) * 0.8)
        valset = trainset[split_idx:]
        trainset = trainset[:split_idx]

    # MIPROv2 doesn't allow num_candidates when auto is set
    # MIPROv2 defaults to auto='light', so we must explicitly set auto=None to use num_candidates
    # NOTE: num_trials is passed to compile(), not __init__()
    if auto is not None:
        optimizer = MIPROv2(
            metric=metric,
            init_temperature=init_temperature,
            num_threads=num_threads,
            auto=auto,
        )
        num_trials = None  # Will be determined by auto setting
        print(f"Running MIPROv2 optimization (auto={auto})...")
    else:
        # When auto=None, num_trials is required in compile(). Recommended: num_trials ~= 1.5 * num_candidates
        num_trials = int(num_candidates * 1.5)
        optimizer = MIPROv2(
            metric=metric,
            num_candidates=num_candidates,
            init_temperature=init_temperature,
            num_threads=num_threads,
            auto=None,  # Explicitly set to None to allow num_candidates
        )
        print(f"Running MIPROv2 optimization (num_candidates={num_candidates}, num_trials={num_trials})...")

    print(f"  trainset: {len(trainset)} examples")
    print(f"  valset: {len(valset)} examples")

    # Build compile kwargs - num_trials is only needed when auto=None
    compile_kwargs = {
        "trainset": trainset,
        "valset": valset,
        "max_bootstrapped_demos": max_demos,
        "max_labeled_demos": max_demos,
    }
    if num_trials is not None:
        compile_kwargs["num_trials"] = num_trials

    compiled = optimizer.compile(module, **compile_kwargs)
    return compiled


def save_compiled_module(module: dspy.Module, path: str) -> None:
    """Save a compiled module to disk.

    Args:
        module: Compiled DSPy module
        path: Path to save (will create .json file)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    module.save(str(path))
    print(f"Saved compiled module to {path}")


def load_compiled_module(module_class: type, path: str) -> dspy.Module:
    """Load a compiled module from disk.

    Args:
        module_class: The module class to instantiate
        path: Path to the saved module

    Returns:
        Loaded module with compiled prompts/demos
    """
    module = module_class()
    module.load(str(path))
    print(f"Loaded compiled module from {path}")
    return module


# --- Optimizer Registry ---

OPTIMIZERS = {
    "bootstrap": run_bootstrap_fewshot,
    "bootstrap_search": run_bootstrap_random_search,
    "mipro": run_mipro,
}


def get_optimizer(name: str) -> Callable:
    """Get an optimizer function by name.

    Args:
        name: Optimizer name

    Returns:
        Optimizer function

    Raises:
        ValueError: If optimizer name is unknown
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(OPTIMIZERS.keys())}")
    return OPTIMIZERS[name]
