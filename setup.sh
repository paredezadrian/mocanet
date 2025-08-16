#!/bin/bash

# MOCA-Net Setup Script
# This script sets up the environment and installs dependencies

set -e  # Exit on any error

echo "🚀 Setting up MOCA-Net environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.12"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.12+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "🔧 Installing MOCA-Net in development mode..."
pip install -e .

echo "✅ Setup completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Run tests: make test"
echo "  3. Start training: make train"
echo ""
echo "🚀 Happy training with MOCA-Net!"
