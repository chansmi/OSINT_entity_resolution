#!/bin/bash
echo "Checking /usr/workspace/smith585/hf_cache/hub/..."
ls -la /usr/workspace/smith585/hf_cache/hub/ 2>&1

echo ""
echo "Looking for model snapshots:"
find /usr/workspace/smith585/hf_cache/hub -name "snapshots" -type d 2>/dev/null

echo ""
echo "Checking if models exist with blobs:"
for dir in /usr/workspace/smith585/hf_cache/hub/models--*; do
    echo "Model: $(basename $dir)"
    ls -la "$dir" 2>&1 | head -5
done

echo ""
echo "Checking vast1 refs:"
cat /p/vast1/smith585/caches/hf_home/hub/models--meta-llama--Llama-3.1-8B-Instruct/refs/main 2>/dev/null || echo "No refs/main found"
