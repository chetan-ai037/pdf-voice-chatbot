#!/bin/bash

echo "==============================="
echo "Project Setup and Start"
echo "==============================="

# Move to project root
cd "$(dirname "$0")/.."

# Check Python
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is not installed."
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Run project
python app.py
