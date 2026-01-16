#!/usr/bin/env python3
"""
Configuration for fine-tuning entity resolution models.

Defines model paths, hyperparameters, and prompts for both:
- Llama-3.1-8B-Instruct
- DeepSeek-R1-Distill-Qwen-14B
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

# Base paths
PRETRAINED_MODELS_DIR = Path("/p/vast1/smith585/models/pretrained")
CHECKPOINTS_DIR = Path("/p/vast1/smith585/checkpoints/entity_resolution")

# System prompt for entity resolution
SYSTEM_PROMPT = """You are an expert entity resolution system for sanctions screening. Your task is to determine whether two entity records refer to the same real-world person or organization.

Your primary task is to identify CONFLICTS, not similarities.

Key Principles:
- Name variations (transliterations, nicknames, titles) are common
- Missing fields are normal - absence of data is NOT evidence of difference
- Same entity often appears across multiple sources with variations

Decision Process:
1. Look for CONTRADICTORY evidence (different dates, conflicting IDs, incompatible attributes)
2. If NO contradictions found -> POSITIVE (same entity)
3. Only NEGATIVE if explicit conflicts exist

Respond with only "positive" if they are the same entity, or "negative" if different."""

# Fields to extract from entities
FIELD_SPECS = [
    ("name", "Names", 8),
    ("alias", "Aliases", 4),
    ("birthDate", "Birth Date", None),
    ("birthPlace", "Birth Place", 2),
    ("nationality", "Nationality", None),
    ("country", "Country", None),
    ("idNumber", "ID Numbers", 3),
    ("passportNumber", "Passport", 2),
    ("gender", "Gender", None),
    ("position", "Position", 2),
    ("firstName", "First Name", None),
    ("lastName", "Last Name", None),
]


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    model_path: Path
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


# Model configurations
MODEL_CONFIGS = {
    "llama-8b": ModelConfig(
        name="Llama-3.1-8B-Instruct",
        model_path=PRETRAINED_MODELS_DIR / "meta-llama--Llama-3.1-8B-Instruct",
        batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
    ),
    "deepseek-14b": ModelConfig(
        name="DeepSeek-R1-Distill-Qwen-14B",
        model_path=PRETRAINED_MODELS_DIR / "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B",
        batch_size=2,  # Smaller batch for larger model
        gradient_accumulation_steps=8,
        learning_rate=1e-4,  # Lower LR for larger model
    ),
}


def get_model_config(model_key: str) -> ModelConfig:
    """Get configuration for a specific model."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_key]


# Class weights for imbalanced data (77% positive, 23% negative)
# Computed as: weight = 1 / (2 * class_ratio)
CLASS_WEIGHTS = {
    "positive": 0.65,   # Down-weight majority class
    "negative": 2.17,   # Up-weight minority class
}


# Data split configuration (25% holdout for evaluation)
# Train: 70%, Val: 5%, Test: 25%
SPLIT_CONFIG = {
    "train_ratio": 0.70,   # 70% for training (~529k pairs)
    "val_ratio": 0.05,     # 5% for validation (~38k pairs)
    "test_ratio": 0.25,    # 25% for final evaluation (~189k pairs)
    "seed": 42,
}
