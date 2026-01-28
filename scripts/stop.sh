#!/bin/bash

echo "==============================="
echo "Stopping and Cleaning Project"
echo "==============================="

# Move to project root
cd "$(dirname "$0")/.."

# Stop Python processes
pkill -f python 2>/dev/null

# Remove virtual environment
rm -rf venv

# Remove cache
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "Project cleaned successfully."
