#!/usr/bin/env python3
"""
DSPy Language Model wrapper for HuggingFace Transformers.

Provides a DSPy-compatible LM that uses HuggingFace transformers directly,
without requiring vLLM or SGLang server setup. This is slower but simpler
and works well for MIPROv2 optimization on GPU clusters.

Usage:
    from scripts.baselines.dspy_er.hf_lm import HuggingFaceLanguageModel
    import dspy

    lm = HuggingFaceLanguageModel(
        model_path='/p/vast1/smith585/models/pretrained/meta-llama--Llama-3.1-8B-Instruct',
        model_name='llama-8b'
    )
    dspy.configure(lm=lm)
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import dspy


# --- DeepSeek R1 Output Processing ---
# Note: We do NOT parse verbose output. If the model doesn't produce valid JSON,
# it should fail - this keeps evaluation fair across all models.


# --- Response classes to mimic OpenAI format ---

@dataclass
class HFMessage:
    """Mimics OpenAI message format."""
    content: str
    role: str = "assistant"


@dataclass
class HFChoice:
    """Mimics OpenAI choice format."""
    message: HFMessage
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class HFUsage:
    """Mimics OpenAI usage format. Supports dict() conversion."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def keys(self):
        return ['prompt_tokens', 'completion_tokens', 'total_tokens']

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.keys())


@dataclass
class HFResponse:
    """Mimics OpenAI completion response format for DSPy compatibility."""
    choices: List[HFChoice]
    model: str
    usage: HFUsage = field(default_factory=HFUsage)


class HuggingFaceLanguageModel(dspy.LM):
    """DSPy LM using HuggingFace Transformers directly (no vLLM/SGLang).

    This class inherits from dspy.LM and provides local model inference
    without running a separate server process.

    Attributes:
        model_path: Path to the pretrained model directory
        model_name: Human-readable model identifier
        temperature: Sampling temperature (0.0 = greedy)
        max_new_tokens: Maximum tokens to generate
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = "local-model",
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        max_input_length: int = 4096,
    ):
        """Initialize the HuggingFace language model.

        Args:
            model_path: Path to the pretrained model directory
            model_name: Human-readable model identifier
            temperature: Sampling temperature (0.0 = greedy decoding)
            max_new_tokens: Maximum tokens to generate per request
            torch_dtype: Data type for model weights (default: bfloat16)
            device_map: Device placement strategy (default: "auto")
            trust_remote_code: Allow custom model code (default: True)
            max_input_length: Maximum input sequence length for truncation
        """
        # Initialize parent class with model name
        super().__init__(
            model=f"huggingface/{model_name}",
            model_type="chat",
            temperature=temperature,
            max_tokens=max_new_tokens,
            cache=False,  # Disable caching for local models
        )

        # Store our custom attributes
        self.model_path = model_path
        self.max_input_length = max_input_length

        # Lazy loading - model loaded on first request
        self._hf_model = None
        self._tokenizer = None
        self._torch_dtype = torch_dtype or torch.bfloat16
        self._device_map = device_map
        self._trust_remote_code = trust_remote_code

        # Detect DeepSeek R1 models (they use <think>...</think> format)
        self._is_deepseek_r1 = self._detect_deepseek_r1(model_path, model_name)

        # DeepSeek R1 models need more tokens for reasoning + answer
        if self._is_deepseek_r1 and max_new_tokens < 2048:
            max_new_tokens = 2048
            # Update the parent class's stored value
            self.kwargs["max_tokens"] = max_new_tokens

        print(f"HuggingFaceLanguageModel initialized (lazy loading)")
        print(f"  Model path: {model_path}")
        print(f"  Model name: {model_name}")
        print(f"  Temperature: {temperature}")
        print(f"  Max new tokens: {max_new_tokens}")
        if self._is_deepseek_r1:
            print(f"  DeepSeek R1 mode: enabled (will extract answer from <think> tags)")

    def _detect_deepseek_r1(self, model_path: str, model_name: str) -> bool:
        """Detect if this is a DeepSeek R1 reasoning model.

        DeepSeek R1 models produce output in <think>...</think> format
        and need special post-processing to extract the final answer.
        """
        # Check model name or path for DeepSeek R1 indicators
        path_lower = model_path.lower()
        name_lower = model_name.lower()

        deepseek_r1_indicators = [
            "deepseek-r1",
            "deepseek_r1",
            "r1-distill",
            "r1_distill",
        ]

        for indicator in deepseek_r1_indicators:
            if indicator in path_lower or indicator in name_lower:
                return True

        return False

    def _add_format_constraint(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add explicit format constraint for DeepSeek R1 models.

        DeepSeek R1 models tend to produce verbose reasoning. This method adds
        explicit instructions to output ONLY structured JSON without thinking.

        Args:
            messages: Original chat messages

        Returns:
            Modified messages with format constraint added
        """
        # Create a copy to avoid modifying the original
        messages = [dict(m) for m in messages]

        # Format constraint to add - be very explicit about JSON-only output
        format_constraint = (
            "\n\n**CRITICAL OUTPUT FORMAT**: You MUST respond with ONLY a valid JSON object. "
            "NO thinking, NO explanation, NO text before or after the JSON. "
            "Your ENTIRE response must be valid JSON starting with '{' and ending with '}'. "
            "Example format: {\"reasoning\": \"brief reason\", \"classification\": \"positive\"}"
        )

        # Add constraint to the last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages[i]["content"] = messages[i]["content"] + format_constraint
                break

        return messages

    def _ensure_loaded(self) -> None:
        """Lazy load the model and tokenizer on first use."""
        if self._hf_model is not None:
            return

        print(f"Loading model from {self.model_path}...")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=self._trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            device_map=self._device_map,
            torch_dtype=self._torch_dtype,
            trust_remote_code=self._trust_remote_code,
        )
        self._hf_model.eval()

        print(f"Model loaded successfully on {next(self._hf_model.parameters()).device}")

    def forward(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate completions for the given prompt or messages.

        This is the main DSPy interface method. Called by dspy.LM.__call__.

        Args:
            prompt: Text prompt (used if messages not provided)
            messages: Chat messages in OpenAI format [{"role": "...", "content": "..."}]
            **kwargs: Additional generation parameters

        Returns:
            List of completion dicts, each with a 'text' key containing the generated text
        """
        self._ensure_loaded()

        # Handle different input formats
        if messages is not None:
            # For DeepSeek R1 models, add explicit format constraint
            if self._is_deepseek_r1:
                messages = self._add_format_constraint(messages)

            # Convert chat messages to prompt using chat template
            if hasattr(self._tokenizer, 'apply_chat_template'):
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # Fallback: concatenate messages
                prompt = "\n".join(
                    f"{m['role']}: {m['content']}" for m in messages
                )
                prompt += "\nassistant:"
        elif prompt is None:
            raise ValueError("Either prompt or messages must be provided")

        # Get generation parameters from kwargs or defaults
        temperature = kwargs.get("temperature", self.kwargs.get("temperature", 0.0))
        max_tokens = kwargs.get("max_tokens", self.kwargs.get("max_tokens", 512))
        n = kwargs.get("n", 1)  # Number of completions

        # Tokenize input
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self._hf_model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            if temperature is None or temperature <= 0.01:
                # Greedy decoding
                output_ids = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    num_return_sequences=n,
                )
            else:
                # Sampling
                output_ids = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                    num_return_sequences=n,
                )

        # Decode outputs and build OpenAI-compatible response
        choices = []
        total_completion_tokens = 0
        for i in range(n):
            generated_ids = output_ids[i][input_length:]
            generated_text = self._tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            choices.append(HFChoice(
                message=HFMessage(content=generated_text),
                index=i,
            ))
            total_completion_tokens += len(generated_ids)

        # Build response object that mimics OpenAI format
        response = HFResponse(
            choices=choices,
            model=self.model,
            usage=HFUsage(
                prompt_tokens=input_length,
                completion_tokens=total_completion_tokens,
                total_tokens=input_length + total_completion_tokens,
            ),
        )

        return response


# --- Model Path Registry ---

MODEL_PATHS = {
    "llama-8b": "meta-llama--Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama--Llama-3.3-70B-Instruct",
    "deepseek-14b": "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-32b": "deepseek-ai--DeepSeek-R1-Distill-Qwen-32B",
    "qwen-235b": "Qwen--Qwen3-235B-A22B-Instruct-2507",
    "mixtral-8x22b": "mistralai--Mixtral-8x22B-Instruct-v0.1",
}

DEFAULT_MODEL_BASE = "/p/vast1/smith585/models/pretrained"


def get_model_path(model_name: str, base_path: Optional[str] = None) -> str:
    """Get the full path for a model name.

    Args:
        model_name: Short model name (e.g., "llama-8b") or full path
        base_path: Base directory for models (default: from env or DEFAULT_MODEL_BASE)

    Returns:
        Full path to model directory

    Raises:
        ValueError: If model name not recognized and not a valid path
    """
    # If it's already a path, return it
    if os.path.isdir(model_name):
        return model_name

    # Get base path from environment or default
    base = base_path or os.environ.get("PRETRAINED_MODELS", DEFAULT_MODEL_BASE)

    # Look up in registry
    if model_name.lower() in MODEL_PATHS:
        return os.path.join(base, MODEL_PATHS[model_name.lower()])

    # Try direct combination
    potential_path = os.path.join(base, model_name)
    if os.path.isdir(potential_path):
        return potential_path

    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Available models: {list(MODEL_PATHS.keys())}. "
        f"Or provide a full path to a model directory."
    )


def create_hf_lm(
    model: str,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    base_path: Optional[str] = None,
    **kwargs,
) -> HuggingFaceLanguageModel:
    """Convenience function to create a HuggingFace LM.

    Args:
        model: Model name (e.g., "llama-8b") or full path
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
        base_path: Base directory for models
        **kwargs: Additional arguments for HuggingFaceLanguageModel

    Returns:
        Configured HuggingFaceLanguageModel instance
    """
    model_path = get_model_path(model, base_path)

    return HuggingFaceLanguageModel(
        model_path=model_path,
        model_name=model,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )


if __name__ == "__main__":
    # Quick test
    import argparse

    parser = argparse.ArgumentParser(description="Test HuggingFace LM wrapper")
    parser.add_argument("--model", default="llama-8b", help="Model name or path")
    parser.add_argument("--prompt", default="What is 2+2? Answer with just the number:", help="Test prompt")
    args = parser.parse_args()

    print(f"Testing HuggingFaceLanguageModel with {args.model}")

    lm = create_hf_lm(args.model)

    print(f"\nRunning inference...")
    result = lm(prompt=args.prompt)

    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {result[0]['text']}")

    # Test chat format
    print(f"\nTesting chat format...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    result = lm(messages=messages)
    print(f"Generated: {result[0]['text']}")
