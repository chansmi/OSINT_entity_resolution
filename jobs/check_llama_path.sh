#!/bin/bash
echo "Checking Llama-8B model path structure..."

HF_CACHE="/p/vast1/smith585/caches/hf_home/hub"
LLAMA_PATH="$HF_CACHE/models--meta-llama--Llama-3.1-8B-Instruct"

echo "Llama-8B directory:"
ls -la "$LLAMA_PATH" 2>&1

echo ""
echo "Snapshots directory:"
ls -la "$LLAMA_PATH/snapshots/" 2>&1

echo ""
echo "First snapshot contents:"
ls -la "$LLAMA_PATH/snapshots/"*/ 2>&1 | head -20

echo ""
echo "Full path to use:"
SNAPSHOT=$(ls -1 "$LLAMA_PATH/snapshots/" | head -1)
echo "$LLAMA_PATH/snapshots/$SNAPSHOT"
