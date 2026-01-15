#!/usr/bin/env python3
"""
Quick test to verify GPU setup and model loading before full evaluation.

Run with: flux run -N 1 -n 1 -g 8 python scripts/baselines/test_gpu_setup.py

This script verifies:
1. GPU visibility (how many GPUs are available)
2. Model loading from local path works
3. Model device placement (which GPUs are used)
4. GPU memory allocation
5. Single inference completes successfully
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    print("=" * 60)
    print("GPU Setup Verification Test")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print(f"Python: {sys.executable}")

    # 1. Check GPU visibility
    print(f"\n1. GPU Visibility:")
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"   torch.cuda.device_count(): {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("   WARNING: No CUDA/ROCm GPUs available!")

    # 2. Check environment visibility variables
    hip_visible = os.environ.get("HIP_VISIBLE_DEVICES", "not set")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES", "not set")
    print(f"\n2. Device Visibility Environment:")
    print(f"   HIP_VISIBLE_DEVICES: {hip_visible}")
    print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"   ROCR_VISIBLE_DEVICES: {rocr_visible}")

    # 3. Test model loading
    print(f"\n3. Testing Model Loading...")
    model_path = "/p/vast1/smith585/models/pretrained/deepseek-ai--DeepSeek-R1-Distill-Qwen-14B"

    print(f"   Target path: {model_path}")
    print(f"   Path exists: {os.path.exists(model_path)}")

    if not os.path.exists(model_path):
        print(f"   ERROR: Model path does not exist!")
        print(f"   This is expected if running from login node (no access to /p/vast1)")
        return

    # List model directory contents
    try:
        contents = os.listdir(model_path)
        print(f"   Model directory contents: {contents[:5]}..." if len(contents) > 5 else f"   Contents: {contents}")
    except Exception as e:
        print(f"   ERROR listing directory: {e}")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n   Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        print(f"   Tokenizer loaded successfully!")
        print(f"   Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"   ERROR loading tokenizer: {e}")
        return

    print(f"\n   Loading model with device_map='auto'...")
    start = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        load_time = time.time() - start
        print(f"   Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"   ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Check device placement
    print(f"\n4. Model Device Placement:")
    if hasattr(model, 'hf_device_map'):
        device_map = model.hf_device_map
        devices_used = set(device_map.values())
        print(f"   Unique devices used: {devices_used}")
        gpu_count = len([d for d in devices_used if isinstance(d, int)])
        print(f"   Number of GPUs used for model layers: {gpu_count}")

        # Show layer distribution
        layer_counts = {}
        for layer, device in device_map.items():
            if device not in layer_counts:
                layer_counts[device] = 0
            layer_counts[device] += 1
        print(f"   Layers per device: {layer_counts}")
    else:
        device = next(model.parameters()).device
        print(f"   All parameters on device: {device}")

    # 5. GPU memory usage
    print(f"\n5. GPU Memory Usage:")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.1f}GB total")

    # 6. Test inference
    print(f"\n6. Testing Inference...")
    test_prompt = "Hello, how are you?"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    print(f"   Input device: {inputs['input_ids'].device}")

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    inference_time = time.time() - start

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Inference time: {inference_time:.2f}s")
    print(f"   Input: '{test_prompt}'")
    print(f"   Response: '{response[:100]}...'")

    # 7. Test with entity resolution style prompt
    print(f"\n7. Testing Entity Resolution Prompt...")
    er_prompt = """Compare these two entities:

=== ENTITY A ===
Type: Person
Names: John Smith

=== ENTITY B ===
Type: Person
Names: J. Smith, John A. Smith

Are these the same entity? Respond with only "positive" or "negative"."""

    inputs = tokenizer(er_prompt, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    inference_time = time.time() - start

    # Get only the generated part
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print(f"   Inference time: {inference_time:.2f}s")
    print(f"   Response: '{response}'")

    print("\n" + "=" * 60)
    print("TEST COMPLETE - All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
