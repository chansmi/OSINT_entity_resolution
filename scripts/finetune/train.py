#!/usr/bin/env python3
"""
Fine-tune LLMs for entity resolution using QLoRA.

Supports:
- Llama-3.1-8B-Instruct
- DeepSeek-R1-Distill-Qwen-14B

Usage:
    # Quick test (local, small sample)
    python scripts/finetune/train.py --model llama-8b --quick --limit 100

    # Full training (via Flux)
    flux batch -t 8:00:00 -N 1 -n 1 jobs/train_llama8b.sh
"""

# =============================================================================
# CRITICAL: Set GPU visibility BEFORE any CUDA/HIP operations
# In multi-node DDP with AMD ROCm, accelerate sets LOCAL_RANK but doesn't
# automatically restrict GPU visibility. Each process must only see ONE GPU
# (its assigned one), otherwise all processes try to use GPU 0.
# This MUST happen before `import torch` which initializes HIP.
# =============================================================================
import os

_local_rank = int(os.environ.get("LOCAL_RANK", 0))
os.environ["HIP_VISIBLE_DEVICES"] = str(_local_rank)
os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)  # For compatibility

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.finetune.config import (
    get_model_config,
    CHECKPOINTS_DIR,
    CLASS_WEIGHTS,
    MODEL_CONFIGS,
)
from scripts.finetune.data_pipeline import compute_file_checksum
from scripts.experiment import Experiment


def verify_no_test_leakage(train_path: Path, val_path: Path) -> None:
    """
    Refuse to train if test data is accidentally included.

    Safety checks:
    1. Path names don't contain "test" (except parent directory)
    2. Train/val checksums don't match test set checksum from metadata

    Raises:
        ValueError: If test data leakage is detected
    """
    # Check 1: Path names don't accidentally point to test data
    for name, path in [("train", train_path), ("val", val_path)]:
        path_str = str(path.name).lower()
        if "test" in path_str:
            raise ValueError(
                f"SAFETY: Refusing to train on potential test data!\n"
                f"  {name} path: {path}\n"
                f"  File name contains 'test' - please verify this is correct."
            )

    # Check 2: Verify checksums don't match test set
    metadata_path = train_path.parent / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        if "test_set_checksum" in metadata:
            test_checksum = metadata["test_set_checksum"]

            for name, path in [("train", train_path), ("val", val_path)]:
                if path.exists():
                    file_checksum = compute_file_checksum(str(path))
                    if file_checksum == test_checksum:
                        raise ValueError(
                            f"SAFETY: {name} file checksum matches test set!\n"
                            f"  {name} path: {path}\n"
                            f"  Checksum: {file_checksum[:16]}...\n"
                            f"  This indicates test data may be used for training."
                        )

    print("✓ Verified: No test data leakage detected")

# Load environment variables
load_dotenv()


def load_training_data(
    train_path: str,
    val_path: str,
    limit: Optional[int] = None
) -> tuple:
    """Load training and validation datasets from JSONL files."""

    def load_jsonl(path: str, limit: Optional[int] = None) -> list:
        data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                data.append(json.loads(line))
        return data

    train_data = load_jsonl(train_path, limit)
    val_data = load_jsonl(val_path, limit // 5 if limit else None)

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    return train_dataset, val_dataset


def get_model_and_tokenizer(
    model_path: str,
    use_4bit: bool = True,
    use_flash_attention: bool = False
):
    """Load model with QLoRA quantization configuration."""

    print(f"Loading model from {model_path}...")

    # QLoRA quantization config
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

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model - NOTE: No device_map for DDP compatibility!
    # accelerate handles device placement when launched with `accelerate launch`
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    model_kwargs = {
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }

    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    # NOTE: Don't manually place model on device!
    # SFTTrainer/accelerate handles device placement in DDP mode.
    # Each process sees only its GPU as cuda:0 via HIP_VISIBLE_DEVICES.
    print(f"  Model loaded (device placement handled by accelerate)")

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def get_lora_config(config) -> LoraConfig:
    """Get LoRA configuration for QLoRA fine-tuning."""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )


def formatting_func(example):
    """Format a single example for training."""
    messages = example["messages"]
    # Return the messages directly - SFTTrainer will handle chat template
    return messages


def train(args):
    """Main training function."""

    # Get model configuration
    config = get_model_config(args.model)
    model_path = str(config.model_path)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{timestamp}"
    output_dir = CHECKPOINTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Entity Resolution Fine-Tuning: {config.name}")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Output dir: {output_dir}")
    print(f"Quick mode: {args.quick}")
    print()

    # Load data
    train_path = Path("data/processed/train.jsonl")
    val_path = Path("data/processed/val.jsonl")

    # SAFETY: Verify no test data leakage before training
    if train_path.exists():
        verify_no_test_leakage(train_path, val_path)

    if not train_path.exists():
        # Fall back to sample data for testing
        print("Warning: Processed data not found, using sample data...")
        train_path = Path("data/samples/sample_1000.json")
        val_path = train_path  # Use same file for quick test

    limit = args.limit if args.quick else None
    train_dataset, val_dataset = load_training_data(
        str(train_path),
        str(val_path),
        limit=limit
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()

    # Load model
    model, tokenizer = get_model_and_tokenizer(
        model_path,
        use_4bit=not args.no_4bit,
        use_flash_attention=args.flash_attention
    )

    # Apply LoRA
    lora_config = get_lora_config(config)
    model = get_peft_model(model, lora_config)

    print("\nTrainable parameters:")
    model.print_trainable_parameters()
    print()

    # Adjust hyperparameters for quick mode
    if args.quick:
        epochs = 1
        eval_steps = 50
        save_steps = 100
        logging_steps = 10
    else:
        epochs = config.epochs
        eval_steps = 500
        save_steps = 500
        logging_steps = 50

    # Training configuration
    training_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,

        # Training params
        num_train_epochs=epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        # Optimizer
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        max_grad_norm=0.3,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",

        # Precision - handled by accelerate via --mixed_precision=bf16
        # Setting bf16=True explicitly can fail GPU detection with HIP_VISIBLE_DEVICES
        # Let accelerate handle precision; it's already set in job scripts
        bf16=False,
        fp16=False,

        # Efficiency
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging
        logging_steps=logging_steps,
        logging_dir=str(output_dir / "logs"),

        # Evaluation
        eval_strategy="steps",
        eval_steps=eval_steps,

        # Checkpointing
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Data
        max_length=config.max_seq_length,

        # Reporting
        report_to=["wandb"] if not args.quick and not args.no_wandb else [],
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    final_path = output_dir / "final"
    print(f"\nSaving final model to {final_path}...")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Log experiment
    exp = Experiment(method=f"finetune_{args.model}", input_file=str(train_path))
    exp.set_params(
        model=config.name,
        model_path=model_path,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        epochs=epochs,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        effective_batch_size=config.batch_size * config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
        quick_mode=args.quick,
        use_4bit=not args.no_4bit,
    )

    # Get final metrics
    if trainer.state.log_history:
        final_train_loss = trainer.state.log_history[-1].get("loss", 0)
        best_eval_loss = trainer.state.best_metric if trainer.state.best_metric else 0
        exp.set_metrics(
            final_train_loss=final_train_loss,
            best_eval_loss=best_eval_loss,
        )

    exp.save()
    exp.print_summary()

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}")
    print(f"Final model: {final_path}")
    print(f"Run name: {run_name}")

    return str(final_path)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLMs for entity resolution"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="deepseek-14b",
        help="Model to fine-tune"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run with small sample"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Limit samples for quick run (default: 1000)"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (use 16-bit)"
    )
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="Use Flash Attention 2 (if available)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging"
    )

    args = parser.parse_args()

    # Set up WandB
    if not args.quick and not args.no_wandb:
        import wandb
        wandb_project = os.environ.get("WANDB_PROJECT", "osint-entity-resolution")
        print(f"WandB project: {wandb_project}")

    train(args)


if __name__ == "__main__":
    main()
