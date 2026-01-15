#!/bin/bash
echo "Checking /p/vast1/smith585/models/:"
ls -la /p/vast1/smith585/models/ 2>&1

echo ""
echo "Looking for model subdirectories:"
find /p/vast1/smith585/models -maxdepth 2 -type d 2>/dev/null
