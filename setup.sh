#!/bin/bash
# Setup script for Strava Guru

echo "Setting up Strava Guru..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To use:"
echo "  1. Activate the virtual environment: source .venv/bin/activate"
echo "  2. Run analyzer: python activity_analyzer.py <activity_file>"
echo "  3. Or test with: python test_analyzer.py"
