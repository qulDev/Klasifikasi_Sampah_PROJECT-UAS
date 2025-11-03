# Klasifikasi Sampah - Automated Waste Detection Pipeline

> Automated pipeline for waste classification using YOLOv8 object detection

Deteksi dan klasifikasi sampah anorganik menggunakan deep learning dengan YOLOv8. Pipeline ini secara otomatis mengonversi berbagai format dataset, melatih model deteksi objek, dan menyediakan antarmuka untuk inferensi real-time.

## 🎯 Features

- **Multi-Format Dataset Conversion**: Automatically detect and convert COCO JSON, Pascal VOC XML, YOLO TXT, class folders, and CSV annotations to unified YOLO format
- **Intelligent Label Mapping**: Fuzzy string matching and keyword-based mapping to standardize heterogeneous class names
- **Automated Data Preparation**: Deduplication via content hashing, stratified train/val/test splits
- **YOLOv8 Training**: GPU-accelerated training with pretrained weights and early stopping
- **Real-time Webcam Detection**: Live object detection on webcam or video files
- **Interactive Notebook**: Jupyter interface for image upload and detection visualization

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (optional, for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for training)

### Setup

1. **Clone repository**:
   ```bash
   git clone https://github.com/yourusername/Klasifikasi_Sampah.git
   cd Klasifikasi_Sampah
   ```

2. **Run setup script**:
   ```bash
   bash setup.sh
   ```

   This will:
   - Create virtual environment (`.venv/`)
   - Install all dependencies
   - Create project directory structure

3. **Activate environment**:
   ```bash
   source .venv/bin/activate
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; import ultralytics; print('✓ All dependencies loaded')"
   ```

## 🚀 Quick Start

### 1. Convert Datasets

Convert raw datasets to YOLO format:

```bash
# Dry run to see what would be converted
python convert_datasets.py --src ./datasets/raw --dst ./datasets/processed/all --dry-run

# Convert all datasets
python convert_datasets.py --src ./datasets/raw --dst ./datasets/processed/all
```

**Supported formats**:
- COCO JSON (Microsoft COCO)
- Pascal VOC XML
- YOLO TXT (existing YOLO datasets)
- Class folders (folder-per-class structure)
- CSV annotations (custom bbox format)

### 2. Split Dataset

Create train/val/test splits with stratification:

```bash
python split_and_prep.py --src ./datasets/processed/all --out ./datasets/processed --split 0.8 0.1 0.1
```

This generates:
- `datasets/processed/train/` - Training set (80%)
- `datasets/processed/val/` - Validation set (10%)
- `datasets/processed/test/` - Test set (10%)
- `data.yaml` - YOLO configuration file

### 3. Train Model

Train YOLOv8 object detection model:

```bash
# Quick smoke test (1 epoch)
python train.py --model yolov8n --epochs 1 --dry-run

# Train small model
python train.py --model yolov8s --epochs 100 --imgsz 640 --batch 16 --device 0

# Train medium model with early stopping
python train.py --model yolov8m --epochs 300 --patience 50
```

**Model variants**:
- `yolov8n`: Nano (fastest, lowest accuracy)
- `yolov8s`: Small (recommended for most use cases)
- `yolov8m`: Medium (good balance)
- `yolov8l`: Large (high accuracy, slower)
- `yolov8x`: Extra Large (best accuracy, slowest)

### 4. Run Inference

#### Webcam Detection

Real-time detection on webcam:

```bash
python detect_webcam.py --weights ./models/best.pt --source 0
```

Controls:
- `q` - Quit
- `s` - Save screenshot
- `p` - Pause/Resume

#### Static Image Detection

Detect objects in a single image:

```bash
python detect_webcam.py --weights ./models/best.pt --source ./test_image.jpg
```

#### Jupyter Notebook

Interactive image scanning:

```bash
jupyter notebook notebooks/scan_image.ipynb
```

## 📊 Dataset Structure

```
datasets/
├── raw/                                  # Original datasets (never modified)
│   ├── Garbage Classification (Kaggle)/
│   ├── garbage-classification-v2/
│   ├── TACO (Trash Annotations in Context)/
│   ├── TrashNet (Stanford)/
│   └── Waste Classification Data v2/
│
├── processed/                            # Converted datasets
│   ├── all/                              # Merged dataset before split
│   │   ├── images/
│   │   └── labels/
│   ├── train/                            # Training set (80%)
│   │   ├── images/
│   │   └── labels/
│   ├── val/                              # Validation set (10%)
│   │   ├── images/
│   │   └── labels/
│   └── test/                             # Test set (10%)
│       ├── images/
│       └── labels/
│
└── label_map.json                        # Label mapping documentation
```

## 🎓 Target Classes

The pipeline standardizes all labels to 6 waste categories:

1. **plastic** - Plastic bottles, bags, wrappers, PET, HDPE, etc.
2. **metal** - Aluminum cans, steel, tin, foil
3. **glass** - Glass bottles, jars
4. **paper** - Paper, newspaper, magazines, documents
5. **cardboard** - Cardboard boxes, cartons, packaging
6. **other** - Unknown or mixed waste items

Label mapping uses:
- Exact matching (case-insensitive)
- Fuzzy matching (typo tolerance)
- Substring matching (`"plastic_bottle"` → `"plastic"`)
- Material keywords (`"aluminum_can"` → `"metal"`)
- Fallback to `"other"` for unknown labels

## 📁 Project Structure

```
Klasifikasi_Sampah/
├── convert_datasets.py        # Dataset conversion script
├── split_and_prep.py          # Data splitting script
├── train.py                   # YOLOv8 training script
├── detect_webcam.py           # Real-time detection script
├── setup.sh                   # Environment setup automation
├── requirements.txt           # Python dependencies
├── data.yaml                  # YOLO dataset configuration
│
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── logger.py              # Logging configuration
│   ├── image_utils.py         # Image verification & hashing
│   ├── label_mapper.py        # Label standardization
│   ├── dataset_stats.py       # Statistics & format detection
│   └── annotation_parsers.py  # Multi-format parsers
│
├── notebooks/                 # Interactive analysis
│   └── scan_image.ipynb       # Image upload & detection UI
│
├── models/                    # Trained model weights
│   └── best.pt                # Latest best model
│
└── runs/                      # Training logs & metrics
    └── detect/
        └── train*/
```

## 🔧 Advanced Usage

### Custom Label Mapping

Create `datasets/label_map.json` with manual mappings:

```json
{
  "manual_mappings": {
    "pet_bottle": "plastic",
    "aluminum_foil": "metal",
    "glass_jar": "glass"
  }
}
```

Then run conversion:

```bash
python convert_datasets.py --src ./datasets/raw --dst ./datasets/processed/all --mapping-file datasets/label_map.json
```

### Custom Split Ratios

```bash
# 70% train, 15% val, 15% test
python split_and_prep.py --src ./datasets/processed/all --out ./datasets/processed --split 0.7 0.15 0.15
```

### Resume Training

```bash
python train.py --model yolov8s --resume
```

### Save Detection Video

```bash
python detect_webcam.py --weights ./models/best.pt --source video.mp4 --save-out output.avi
```

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size

```bash
python train.py --model yolov8s --batch 8  # Try 8 or 4
```

### Issue: Low mAP (Mean Average Precision)

**Possible causes**:
- Insufficient training epochs (try 100-300)
- Poor quality annotations
- Class imbalance (check with `--verbose`)

**Solutions**:
- Train longer: `--epochs 300`
- Use larger model: `--model yolov8m`
- Enable early stopping: `--patience 50`

### Issue: Webcam Not Opening

**Solution**: Check webcam index

```bash
# Try different indices
python detect_webcam.py --weights ./models/best.pt --source 1
python detect_webcam.py --weights ./models/best.pt --source 2
```

### Issue: No Objects Detected

**Solution**: Lower confidence threshold

```bash
python detect_webcam.py --weights ./models/best.pt --source image.jpg --conf 0.15
```

## 📊 Performance Benchmarks

**Training** (YOLOv8s, 100 epochs, RTX 3060):
- Dataset: 10K images
- Training time: ~2 hours
- mAP@0.5: 0.65-0.75 (expected)

**Inference** (YOLOv8s, RTX 3060):
- Webcam: 30-60 FPS
- Static image: 15-20 ms

**Inference** (YOLOv8s, CPU):
- Webcam: 3-5 FPS
- Static image: 200-300 ms

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 framework
- **TACO Dataset**: Trash Annotations in Context
- **TrashNet**: Stanford waste classification dataset
- **Kaggle**: Various waste classification datasets

## 📞 Support

For questions or issues:

1. Check [Troubleshooting](#-troubleshooting) section
2. Search existing issues
3. Create a new issue with:
   - Python version
   - Operating system
   - Error messages
   - Steps to reproduce

---

**Built with ❤️ for automated waste detection**
