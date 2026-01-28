#!/bin/bash
echo "========================================="
echo "Cleaning Project Environment"
echo "========================================="

cd "$(dirname "$0")/.."

rm -rf venv
rm -rf temp_audio
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "Cleanup complete."
