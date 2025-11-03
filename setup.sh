#!/bin/bash
set -e

echo "======================================"
echo "Klasifikasi Sampah - Setup Script"
echo "======================================"
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "[2/5] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "✓ Virtual environment already exists"
else
    python3 -m venv .venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "[3/5] Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "[4/5] Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"

# Install dependencies
echo ""
echo "[5/5] Installing dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# Create directory structure
echo ""
echo "Creating project directories..."
mkdir -p datasets/processed/all/images
mkdir -p datasets/processed/all/labels
mkdir -p datasets/processed/train/images
mkdir -p datasets/processed/train/labels
mkdir -p datasets/processed/val/images
mkdir -p datasets/processed/val/labels
mkdir -p datasets/processed/test/images
mkdir -p datasets/processed/test/labels
mkdir -p models
mkdir -p runs/logs
mkdir -p utils
mkdir -p tests
mkdir -p notebooks

echo "✓ Directory structure created"

echo ""
echo "======================================"
echo "✓ Setup complete!"
echo "======================================"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To verify installation, run:"
echo "  python -c 'import torch; import ultralytics; print(\"✓ All dependencies loaded successfully\")'"
echo ""
