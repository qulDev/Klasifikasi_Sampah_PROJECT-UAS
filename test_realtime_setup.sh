#!/bin/bash
# Quick test script to verify realtime_detect.py is ready

echo "=================================================="
echo "Testing Real-Time Detection Setup"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: ./setup.sh first"
    exit 1
fi

echo "✓ Virtual environment found"

# Check if model exists
if [ ! -f "models/best.pt" ]; then
    echo "❌ Model not found!"
    echo "   Run: python train.py first"
    exit 1
fi

echo "✓ Model found ($(du -h models/best.pt | cut -f1))"

# Check if realtime_detect.py exists
if [ ! -f "realtime_detect.py" ]; then
    echo "❌ realtime_detect.py not found!"
    exit 1
fi

echo "✓ realtime_detect.py found"

# Test CUDA availability
echo ""
echo "Checking CUDA availability..."
.venv/bin/python -c "
import torch
if torch.cuda.is_available():
    print('✓ CUDA is available')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA Version: {torch.version.cuda}')
else:
    print('⚠️  CUDA not available (will use CPU)')
"

echo ""
echo "=================================================="
echo "✅ Everything is ready!"
echo "=================================================="
echo ""
echo "To run real-time detection:"
echo ""
echo "  source .venv/bin/activate"
echo "  python realtime_detect.py"
echo ""
echo "Controls:"
echo "  Q - Quit"
echo "  S - Save frame"
echo "  C - Toggle confidence"
echo ""
echo "See REALTIME_SUMMARY.md for full instructions!"
echo "=================================================="
