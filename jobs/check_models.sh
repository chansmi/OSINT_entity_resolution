#!/bin/bash
echo "Checking model paths..."
echo "Hostname: $(hostname)"
echo ""
echo "Contents of /p/vast1/smith585/models/pretrained/:"
ls -la /p/vast1/smith585/models/pretrained/ 2>&1
echo ""
echo "Checking specific paths:"
echo "Llama-8B: $(ls -d /p/vast1/smith585/models/pretrained/meta-llama* 2>&1)"
echo "DeepSeek: $(ls -d /p/vast1/smith585/models/pretrained/deepseek* 2>&1)"
