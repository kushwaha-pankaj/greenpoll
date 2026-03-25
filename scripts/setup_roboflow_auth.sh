#!/bin/bash
# Setup Roboflow authentication for dataset downloads
# Usage: source scripts/setup_roboflow_auth.sh

read -p "Enter your Roboflow API Key: " ROBOFLOW_API_KEY
export ROBOFLOW_API_KEY

echo "✓ ROBOFLOW_API_KEY set. You can now run:"
echo "  python scripts/download_datasets.py --all"
echo "  python scripts/download_datasets.py --crop tomato"
