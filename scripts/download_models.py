#!/usr/bin/env python3
"""Download required models for entity resolution evaluation.

Models:
- meta-llama/Llama-3.1-8B-Instruct (gated - requires HF token)
- deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

Usage:
    # Set your HuggingFace token first
    export HF_TOKEN=your_token_here

    # Run download (use flux for compute node access to vast1)
    flux run -N 1 -n 1 python scripts/download_models.py
"""

import os
import sys
from pathlib import Path

# Set cache directories BEFORE importing transformers
CACHE_DIR = Path("/p/vast1/smith585/models/pretrained")
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)

from huggingface_hub import snapshot_download, login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODELS = [
    {
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "local_name": "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B",
        "gated": False,
    },
    {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_name": "meta-llama--Llama-3.1-8B-Instruct",
        "gated": True,
    },
]


def main():
    print("=" * 60)
    print("Model Download Script")
    print("=" * 60)

    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownload directory: {CACHE_DIR}")

    # Check for HF token (required for gated models like Llama)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("HF_TOKEN found, logging in...")
        login(token=hf_token)
    else:
        print("WARNING: HF_TOKEN not set. Gated models (Llama) will fail.")
        print("Set with: export HF_TOKEN=your_token")

    for model_info in MODELS:
        repo_id = model_info["repo_id"]
        local_name = model_info["local_name"]
        local_path = CACHE_DIR / local_name

        print(f"\n{'=' * 60}")
        print(f"Downloading: {repo_id}")
        print(f"To: {local_path}")
        print("=" * 60)

        if model_info["gated"] and not hf_token:
            print(f"SKIPPING {repo_id} - requires HF_TOKEN for gated model")
            continue

        try:
            # Use snapshot_download for clean local path
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,  # Copy files directly
                token=hf_token,
            )
            print(f"✓ Downloaded {repo_id}")

            # Verify download
            print(f"\nVerifying {local_path}...")
            files = list(local_path.glob("*.safetensors"))
            if files:
                print(f"✓ Found {len(files)} safetensor files")
                total_size = sum(f.stat().st_size for f in files) / 1e9
                print(f"✓ Total size: {total_size:.2f} GB")
            else:
                # Check for pytorch_model.bin as fallback
                bin_files = list(local_path.glob("*.bin"))
                if bin_files:
                    print(f"✓ Found {len(bin_files)} bin files")
                else:
                    print("⚠ No model weight files found!")

        except Exception as e:
            print(f"✗ Failed to download {repo_id}: {e}")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)

    # Show final state
    print("\nDownloaded models:")
    for model_info in MODELS:
        local_path = CACHE_DIR / model_info["local_name"]
        if local_path.exists():
            files = list(local_path.glob("*.safetensors")) + list(local_path.glob("*.bin"))
            if files:
                print(f"  ✓ {model_info['local_name']}: {len(files)} weight files")
            else:
                print(f"  ⚠ {model_info['local_name']}: exists but no weight files")
        else:
            print(f"  ✗ {model_info['local_name']}: not downloaded")


if __name__ == "__main__":
    main()
