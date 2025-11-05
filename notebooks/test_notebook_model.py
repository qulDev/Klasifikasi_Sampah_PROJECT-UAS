#!/usr/bin/env python3
"""
Quick test to verify model loading for the Jupyter notebook
Run this from the notebooks/ directory to test paths
"""

import sys
from pathlib import Path

print("=" * 60)
print("Notebook Model Path Diagnostics")
print("=" * 60)

# Show current directory
print(f"\n📂 Current directory: {Path.cwd()}")
print(f"📂 Script location: {Path(__file__).parent.resolve()}")

# Check both possible model paths
paths_to_check = [
    ("./models/best.pt", "Same directory as notebook"),
    ("../models/best.pt", "One level up (from notebooks/ to project root)"),
]

print("\n🔍 Checking model paths:\n")

found_model = None
for model_path, description in paths_to_check:
    path_obj = Path(model_path)
    exists = path_obj.exists()
    absolute = path_obj.resolve()
    
    status = "✓ FOUND" if exists else "✗ Not found"
    print(f"{status}: {model_path}")
    print(f"   Description: {description}")
    print(f"   Absolute path: {absolute}")
    
    if exists:
        size = path_obj.stat().st_size / (1024 * 1024)  # Convert to MB
        print(f"   Size: {size:.2f} MB")
        found_model = model_path
    
    print()

# Try to load the model if found
if found_model:
    print("=" * 60)
    print(f"🎯 Attempting to load model from: {found_model}")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        model = YOLO(found_model)
        print(f"\n✅ SUCCESS! Model loaded")
        print(f"   Classes: {model.names}")
        print(f"   Number of classes: {len(model.names)}")
    except Exception as e:
        print(f"\n❌ FAILED to load model: {e}")
        sys.exit(1)
else:
    print("=" * 60)
    print("❌ ERROR: No model found!")
    print("=" * 60)
    print("\n💡 Solution:")
    print("   1. Make sure you've trained a model: python train.py")
    print("   2. Check that models/best.pt exists in project root")
    print(f"   3. Current directory: {Path.cwd()}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All checks passed! You can use the notebook now.")
print("=" * 60)
