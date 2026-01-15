#!/bin/bash
echo "Searching for model paths..."
echo "Hostname: $(hostname)"
echo ""

echo "Checking /p/vast1/smith585/:"
ls -la /p/vast1/smith585/ 2>&1 | head -20
echo ""

echo "Checking /p/vast1/:"
ls -la /p/vast1/ 2>&1 | head -10
echo ""

echo "Checking /usr/workspace/smith585/:"
ls -la /usr/workspace/smith585/ 2>&1 | head -10
echo ""

echo "Looking for pretrained model directories:"
find /p/vast1 -name "*pretrained*" -type d 2>/dev/null | head -10
find /usr/workspace -name "*pretrained*" -type d 2>/dev/null | head -10

echo ""
echo "Looking for Llama or DeepSeek directories:"
find /p/vast1 -name "*llama*" -type d 2>/dev/null | head -5
find /p/vast1 -name "*deepseek*" -type d 2>/dev/null | head -5
