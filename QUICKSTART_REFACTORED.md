# QUICKSTART - Klasifikasi Sampah

## 🚀 Simple Commands

### 1. Setup Environment
```bash
bash setup.sh
source .venv/bin/activate  # Or: .venv/bin/python
```

### 2. Convert Datasets (Auto-detect format)
```bash
python convert_datasets.py
# Reads from: ./datasets/raw
# Outputs to: ./datasets/processed/all
```

### 3. Split Dataset
```bash
python split_and_prep.py
# Reads from: ./datasets/processed/all
# Outputs to: ./datasets/processed (train/val/test)
```

### 4. Train Model
```bash
python train.py
# Uses: ./datasets/processed/data.yaml
# Outputs to: ./models/best.pt
```

### 5. Detect Objects

**Real-time Webcam:**
```bash
python detect.py
# Uses webcam 0, runs on GPU if available
# Controls: Q=Quit | S=Save | C=Confidence
```

**Interactive Notebook:**
```bash
jupyter notebook notebooks/scan_image.ipynb
# Run cells 1-2, upload image, auto-detect
```

---

## 📋 File Summary

| File | Lines | Purpose | Command |
|------|-------|---------|---------|
| `convert_datasets.py` | 590 | Convert datasets to YOLO | `python convert_datasets.py` |
| `split_and_prep.py` | 380 | Split train/val/test | `python split_and_prep.py` |
| `train.py` | 250 | Train YOLOv8 | `python train.py` |
| `detect.py` | 130 | Real-time detection | `python detect.py` |
| `scan_image.ipynb` | 4 cells | Interactive detection | Jupyter notebook |

---

## 🔧 Advanced Options

### Convert with custom source
```bash
python convert_datasets.py --src ./my_datasets --dst ./output
```

### Split with custom ratio
```bash
python split_and_prep.py --split 0.7 0.15 0.15
```

### Train different model size
```bash
python train.py --model yolov8m --epochs 50
```

### Detect from file
```bash
python detect.py  # Edit CAM=0 to image path in code
```

---

## 📁 Directory Structure

```
Klasifikasi_Sampah/
├── convert_datasets.py     # Dataset converter
├── split_and_prep.py        # Dataset splitter
├── train.py                 # Model trainer
├── detect.py                # Real-time detection (NEW - simplified)
├── realtime_detect.py       # Real-time detection (OLD - verbose)
├── notebooks/
│   └── scan_image.ipynb     # Interactive detection (4 cells)
├── datasets/
│   ├── raw/                 # Put your datasets here
│   ├── processed/
│   │   ├── all/            # Converted datasets
│   │   ├── train/          # Training split
│   │   ├── val/            # Validation split
│   │   └── test/           # Test split
│   └── data.yaml           # Auto-generated config
└── models/
    └── best.pt              # Trained model
```

---

## ✅ What Changed (Refactoring)

### 1. **Simplified Commands**
- **Before**: `python convert_datasets.py --src ./datasets/raw --dst ./datasets/processed/all`
- **After**: `python convert_datasets.py` (uses smart defaults)

### 2. **Reduced Code**
- **detect.py**: 130 lines (was 358 lines in realtime_detect.py)
- **scan_image.ipynb**: 4 cells (was 10 cells)
- Same features, less complexity

### 3. **Smart Defaults**
- All scripts use sensible default paths
- No required arguments for basic usage
- Advanced users can still customize

### 4. **Cleaner Notebook**
- Cell 1: Load model
- Cell 2: Upload & auto-detect
- Cell 3: Help documentation
- Cell 4: (removed - merged into Cell 2)

---

## 🎯 Target Classes

1. `plastic`
2. `metal`
3. `glass`
4. `paper`
5. `cardboard`
6. `other`

---

## 💡 Tips

1. **Dataset Location**: Put raw datasets in `./datasets/raw/`
2. **Model Training**: First run uses pretrained COCO weights
3. **GPU Detection**: Auto-enables CUDA if available
4. **Confidence**: Default 0.25 (25%) - adjust in code if needed

---

*Updated: 2025-11-08 | Refactored for simplicity and ease of use*
