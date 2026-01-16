#!/bin/bash
echo "Checking HuggingFace caches..."

echo "1. /p/vast1/smith585/caches/:"
ls -la /p/vast1/smith585/caches/ 2>&1 | head -20

echo ""
echo "2. /p/vast1/smith585/caches/hf_home/:"
ls -la /p/vast1/smith585/caches/hf_home/ 2>&1 | head -20

echo ""
echo "3. Hub models directory:"
ls -la /p/vast1/smith585/caches/hf_home/hub/ 2>&1 | head -20

echo ""
echo "4. Looking for Llama or DeepSeek in hub:"
ls -la /p/vast1/smith585/caches/hf_home/hub/ 2>&1 | grep -i "llama\|deepseek" | head -10

echo ""
echo "5. /usr/workspace/smith585/hf_cache/:"
ls -la /usr/workspace/smith585/hf_cache/ 2>&1 | head -20
