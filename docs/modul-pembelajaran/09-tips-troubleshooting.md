# 📖 Modul 09: Tips dan Troubleshooting

## Daftar Isi

1. [Common Errors](#1-common-errors)
2. [Performance Issues](#2-performance-issues)
3. [Training Problems](#3-training-problems)
4. [Dataset Issues](#4-dataset-issues)
5. [Deployment Challenges](#5-deployment-challenges)
6. [Best Practices Summary](#6-best-practices-summary)
7. [Debug Techniques](#7-debug-techniques)
8. [Environment Setup](#8-environment-setup)
9. [FAQ](#9-faq)
10. [Quick Reference](#10-quick-reference)

---

## 1. Common Errors

### 1.1 Error Dictionary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       COMMON ERRORS REFERENCE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ERROR: ModuleNotFoundError: No module named 'ultralytics'                 │
│  ─────────────────────────────────────────────────────────                 │
│  Cause: Package belum terinstall                                           │
│  Fix:   pip install ultralytics                                            │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│  ERROR: CUDA out of memory                                                 │
│  ────────────────────────────                                              │
│  Cause: GPU memory tidak cukup untuk batch size                           │
│  Fix:                                                                       │
│    1. Kurangi batch size: batch=8 atau batch=4                            │
│    2. Kurangi image size: imgsz=320                                       │
│    3. Use CPU: device='cpu'                                               │
│    4. Restart kernel untuk free memory                                    │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│  ERROR: No such file or directory: 'models/best_model.pt'                 │
│  ──────────────────────────────────────────────────────────               │
│  Cause: Model belum di-train atau path salah                              │
│  Fix:                                                                       │
│    1. Run python train.py terlebih dahulu                                 │
│    2. Check path di script                                                 │
│    3. Copy model ke lokasi yang benar                                     │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│  ERROR: cv2.error: OpenCV(...) Can't open camera                          │
│  ───────────────────────────────────────────────                          │
│  Cause: Webcam tidak tersedia atau dipakai aplikasi lain                  │
│  Fix:                                                                       │
│    1. Tutup aplikasi lain yang pakai webcam                              │
│    2. Check camera index: cv2.VideoCapture(1) untuk external cam         │
│    3. Install/update webcam driver                                        │
│    4. Check permissions (Windows: Privacy settings)                       │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│  ERROR: RuntimeError: Expected all tensors on same device                 │
│  ──────────────────────────────────────────────────────                   │
│  Cause: Mismatch antara model device dan input device                     │
│  Fix:                                                                       │
│    model = model.to('cuda')  # atau 'cpu'                                 │
│    input_tensor = input_tensor.to(model.device)                           │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│  ERROR: PIL.UnidentifiedImageError: cannot identify image file            │
│  ─────────────────────────────────────────────────────────               │
│  Cause: File corrupt atau bukan format gambar yang valid                  │
│  Fix:                                                                       │
│    1. Validate images sebelum processing                                  │
│    2. Skip corrupt files dengan try/except                                │
│    3. Re-download atau regenerate corrupt files                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Error Handling Pattern

```python
# Robust error handling template
def safe_process_image(image_path):
    """Process image dengan proper error handling."""
    try:
        # Validate input
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = Image.open(image_path)
        image.verify()  # Check integrity

        # Reopen after verify
        image = Image.open(image_path)

        # Process
        result = model(image)
        return result

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return None

    except PIL.UnidentifiedImageError as e:
        logger.warning(f"Corrupt image: {image_path}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error processing {image_path}: {e}")
        raise  # Re-raise unexpected errors
```

---

## 2. Performance Issues

### 2.1 Slow Inference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PERFORMANCE OPTIMIZATION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SYMPTOM: Inference sangat lambat (< 5 FPS)                                │
│                                                                             │
│  DIAGNOSIS:                                                                 │
│  ──────────                                                                │
│                                                                             │
│  # Check device being used                                                 │
│  import torch                                                              │
│  print(f"CUDA available: {torch.cuda.is_available()}")                    │
│  print(f"Current device: {torch.cuda.current_device()}")                  │
│  print(f"GPU name: {torch.cuda.get_device_name()}")                       │
│                                                                             │
│  SOLUTIONS:                                                                 │
│  ──────────                                                                │
│                                                                             │
│  1. USE GPU (if available):                                                │
│     model = YOLO('model.pt')                                              │
│     model.to('cuda')  # atau 'cuda:0' untuk GPU pertama                   │
│                                                                             │
│  2. USE SMALLER MODEL:                                                     │
│     ┌──────────┬────────────┬────────────┐                               │
│     │  Model   │  Params    │  Speed*    │                               │
│     ├──────────┼────────────┼────────────┤                               │
│     │  YOLOv8n │  3.2M      │  ~80 FPS   │  ← Recommended               │
│     │  YOLOv8s │  11.2M     │  ~50 FPS   │                               │
│     │  YOLOv8m │  25.9M     │  ~30 FPS   │                               │
│     │  YOLOv8l │  43.7M     │  ~20 FPS   │                               │
│     │  YOLOv8x │  68.2M     │  ~15 FPS   │                               │
│     └──────────┴────────────┴────────────┘                               │
│     * Approximate, depends on hardware                                    │
│                                                                             │
│  3. REDUCE INPUT SIZE:                                                     │
│     results = model(image, imgsz=320)  # Instead of 640                   │
│                                                                             │
│  4. USE HALF PRECISION (GPU only):                                         │
│     model.half()  # FP16 instead of FP32                                  │
│                                                                             │
│  5. SKIP FRAMES (for video):                                               │
│     frame_skip = 2                                                        │
│     if frame_count % frame_skip != 0:                                     │
│         continue                                                           │
│                                                                             │
│  6. BATCH PROCESSING:                                                      │
│     results = model([img1, img2, img3])  # Process multiple at once       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Memory Issues

```python
# Memory profiling
import torch

def print_memory_stats():
    """Print GPU memory usage."""
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

# Clear GPU memory
def clear_memory():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Memory-efficient inference
def memory_efficient_inference(images, model, batch_size=4):
    """Process images in batches to manage memory."""
    results = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_results = model(batch)
        results.extend(batch_results)

        # Optional: clear cache between batches
        clear_memory()

    return results
```

---

## 3. Training Problems

### 3.1 Model Not Learning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TRAINING ISSUES                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SYMPTOM: Loss tidak turun / mAP tidak naik                                │
│                                                                             │
│  CHECKLIST:                                                                 │
│  ──────────                                                                │
│                                                                             │
│  ☐ 1. Data Quality                                                        │
│     - Annotations correct? (check dengan visualisasi)                     │
│     - Class balance? (tidak terlalu imbalanced)                           │
│     - Enough samples per class? (minimal 100-200)                         │
│                                                                             │
│  ☐ 2. Data Path                                                           │
│     - data.yaml paths correct?                                            │
│     - Images dan labels match?                                            │
│     - No missing files?                                                    │
│                                                                             │
│  ☐ 3. Learning Rate                                                       │
│     - Try different values: lr0=0.001, 0.01, 0.0001                       │
│     - Use warmup: warmup_epochs=3                                         │
│                                                                             │
│  ☐ 4. Batch Size                                                          │
│     - Too small = noisy gradients                                         │
│     - Too large = generalization issues                                   │
│     - Try: batch=16, 32, 64                                               │
│                                                                             │
│  ☐ 5. Epochs                                                              │
│     - Not enough training time?                                           │
│     - Try more epochs: epochs=200, 300                                    │
│     - Check early stopping settings                                       │
│                                                                             │
│  DIAGNOSTIC CODE:                                                          │
│  ────────────────                                                          │
│                                                                             │
│  # Validate dataset before training                                        │
│  from ultralytics import YOLO                                              │
│  model = YOLO('yolov8n.pt')                                               │
│  model.val(data='data.yaml')  # Check mAP on val set                      │
│                                                                             │
│  # Visualize predictions                                                   │
│  results = model.predict(source='test_images/', save=True)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Overfitting

```python
# Signs of overfitting:
# - Train loss keeps decreasing
# - Val loss starts increasing
# - Large gap between train mAP and val mAP

# Solutions:

# 1. Data Augmentation (built-in YOLO)
model.train(
    data='data.yaml',
    epochs=100,
    # Augmentation parameters
    hsv_h=0.015,      # Hue augmentation
    hsv_s=0.7,        # Saturation augmentation
    hsv_v=0.4,        # Value augmentation
    degrees=10.0,     # Rotation
    translate=0.1,    # Translation
    scale=0.5,        # Scale
    fliplr=0.5,       # Horizontal flip
    mosaic=1.0,       # Mosaic augmentation
    mixup=0.1,        # Mixup augmentation
)

# 2. Regularization
model.train(
    data='data.yaml',
    weight_decay=0.0005,  # L2 regularization
    dropout=0.1,          # Dropout (if supported)
)

# 3. More data
# - Collect more images
# - Use data augmentation offline
# - Consider transfer learning from similar domain

# 4. Simpler model
# - Use YOLOv8n instead of YOLOv8x
# - Reduce model complexity
```

### 3.3 Underfitting

```python
# Signs of underfitting:
# - Both train and val loss high
# - Low mAP on both sets
# - Model makes random predictions

# Solutions:

# 1. More complex model
model = YOLO('yolov8m.pt')  # Medium instead of nano

# 2. More training time
model.train(
    data='data.yaml',
    epochs=300,      # More epochs
    patience=50,     # Wait longer before early stop
)

# 3. Higher learning rate
model.train(
    data='data.yaml',
    lr0=0.01,        # Higher initial LR
)

# 4. Better pretrained weights
model = YOLO('yolov8m-seg.pt')  # Pretrained on COCO
model.train(data='data.yaml', pretrained=True)
```

---

## 4. Dataset Issues

### 4.1 Annotation Problems

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ANNOTATION TROUBLESHOOTING                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEM: Bounding boxes tidak match dengan objects                        │
│  ───────────────────────────────────────────────────                       │
│                                                                             │
│  Common causes:                                                            │
│  1. Coordinate format mismatch                                             │
│     - YOLO: [x_center, y_center, width, height] normalized                │
│     - COCO: [x, y, width, height] pixels                                  │
│     - VOC:  [xmin, ymin, xmax, ymax] pixels                               │
│                                                                             │
│  2. Image size mismatch during conversion                                  │
│     - Used wrong dimensions for normalization                              │
│                                                                             │
│  VERIFICATION CODE:                                                        │
│  ──────────────────                                                        │
│                                                                             │
│  import cv2                                                                │
│  import matplotlib.pyplot as plt                                           │
│                                                                             │
│  def visualize_annotation(image_path, label_path):                        │
│      img = cv2.imread(str(image_path))                                    │
│      h, w = img.shape[:2]                                                 │
│                                                                             │
│      with open(label_path) as f:                                          │
│          for line in f:                                                   │
│              parts = line.strip().split()                                 │
│              cls_id = int(parts[0])                                       │
│              x_center = float(parts[1]) * w                               │
│              y_center = float(parts[2]) * h                               │
│              box_w = float(parts[3]) * w                                  │
│              box_h = float(parts[4]) * h                                  │
│                                                                             │
│              x1 = int(x_center - box_w/2)                                 │
│              y1 = int(y_center - box_h/2)                                 │
│              x2 = int(x_center + box_w/2)                                 │
│              y2 = int(y_center + box_h/2)                                 │
│                                                                             │
│              cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)          │
│              cv2.putText(img, str(cls_id), (x1,y1-10),                   │
│                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)     │
│                                                                             │
│      plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))                     │
│      plt.show()                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Class Imbalance

```python
# Check class distribution
from collections import Counter
from pathlib import Path

def check_class_balance(labels_dir):
    """Analyze class distribution."""
    class_counts = Counter()

    for label_file in Path(labels_dir).glob('*.txt'):
        with open(label_file) as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                class_counts[class_id] += 1

    # Print statistics
    total = sum(class_counts.values())
    print(f"Total annotations: {total}")
    print(f"\nClass distribution:")

    for cls_id, count in sorted(class_counts.items()):
        percentage = count / total * 100
        print(f"  Class {cls_id}: {count:5d} ({percentage:5.2f}%)")

        # Warning for imbalance
        if percentage < 5:
            print(f"    ⚠️  Underrepresented class!")
        elif percentage > 30:
            print(f"    ⚠️  Overrepresented class!")

# Solutions for imbalance:

# 1. Oversampling (duplicate minority class images)
# 2. Class weights in training (if supported)
# 3. Collect more data for minority classes
# 4. Data augmentation focused on minority classes
```

---

## 5. Deployment Challenges

### 5.1 Environment Differences

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DEPLOYMENT CHECKLIST                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ☐ Python version matches (3.8+ recommended)                              │
│                                                                             │
│  ☐ All dependencies installed                                              │
│    pip install -r requirements.txt                                         │
│                                                                             │
│  ☐ Model file available                                                    │
│    - Include models/best_model.pt                                         │
│    - Check file permissions                                                │
│                                                                             │
│  ☐ GPU/CPU compatibility                                                   │
│    - Model trained on GPU → can run on CPU                                │
│    - Check CUDA version compatibility                                      │
│                                                                             │
│  ☐ Path handling                                                          │
│    - Use pathlib for cross-platform paths                                 │
│    - Avoid hardcoded absolute paths                                       │
│                                                                             │
│  ☐ Firewall/ports (for API)                                               │
│    - Allow incoming connections on API port                               │
│    - Configure CORS properly                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Dependency Management

```python
# requirements.txt best practices

# Pin exact versions for reproducibility
ultralytics==8.0.0
torch==2.0.0
opencv-python==4.8.0.74
streamlit==1.28.0
fastapi==0.104.0

# Or use version ranges for flexibility
# ultralytics>=8.0.0,<9.0.0
# torch>=2.0.0

# Generate requirements from current environment
# pip freeze > requirements.txt

# Create minimal requirements (just what you need)
# pip install pipreqs
# pipreqs . --force
```

### 5.3 Docker Configuration

```dockerfile
# Dockerfile for production

# Use specific Python version
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m appuser
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 6. Best Practices Summary

### 6.1 Code Quality

```python
# ✅ DO: Use type hints
def process_image(image_path: Path, confidence: float = 0.25) -> List[Detection]:
    pass

# ✅ DO: Document functions
def train_model(data_path: str, epochs: int = 100) -> Model:
    """
    Train YOLO model on custom dataset.

    Args:
        data_path: Path to data.yaml
        epochs: Number of training epochs

    Returns:
        Trained model instance

    Raises:
        FileNotFoundError: If data_path doesn't exist
    """
    pass

# ✅ DO: Handle errors gracefully
try:
    result = model(image)
except RuntimeError as e:
    logger.error(f"Inference failed: {e}")
    return fallback_result

# ✅ DO: Use logging instead of print
import logging
logger = logging.getLogger(__name__)
logger.info("Processing started")  # Not print()

# ✅ DO: Use constants for magic values
CONFIDENCE_THRESHOLD = 0.25
MAX_BATCH_SIZE = 32
MODEL_PATH = Path("models/best_model.pt")

# ❌ DON'T: Hardcode values
# result = model(image, conf=0.25)  # Magic number!
```

### 6.2 Project Organization

```
project/
├── README.md              # Documentation
├── requirements.txt       # Dependencies
├── data.yaml             # Dataset config
│
├── train.py              # Training script
├── detect.py             # Detection script
├── api.py                # REST API
├── web_app.py            # Web interface
│
├── utils/                # Utility modules
│   ├── __init__.py
│   └── ...
│
├── models/               # Model files
│   └── best_model.pt
│
├── datasets/             # Data
│   ├── raw/             # Original data
│   └── processed/       # Converted data
│
├── runs/                 # Training outputs
│   ├── detect/
│   └── logs/
│
├── tests/               # Unit tests
│   └── test_*.py
│
└── docs/                # Documentation
    └── ...
```

---

## 7. Debug Techniques

### 7.1 Visual Debugging

```python
# Visualize model predictions step by step

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def debug_detection(image_path, model_path):
    """Debug detection dengan visualisasi."""

    # Load model
    model = YOLO(model_path)

    # Load image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")

    # Run inference dengan verbose
    results = model(img, verbose=True)

    # Print detection details
    print(f"\nDetections: {len(results[0].boxes)}")

    for i, box in enumerate(results[0].boxes):
        print(f"\nDetection {i+1}:")
        print(f"  Class: {results[0].names[int(box.cls)]}")
        print(f"  Confidence: {float(box.conf):.4f}")
        print(f"  Box (xyxy): {box.xyxy.tolist()}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis('off')

    annotated = results[0].plot()
    axes[1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Detections")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
```

### 7.2 Logging for Debug

```python
import logging

# Setup debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def process_with_logging(image_path):
    """Process dengan detailed logging."""

    logger.debug(f"Starting process for: {image_path}")

    # Step 1: Load
    logger.debug("Loading image...")
    try:
        image = Image.open(image_path)
        logger.debug(f"Image loaded: {image.size}, {image.mode}")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise

    # Step 2: Preprocess
    logger.debug("Preprocessing...")
    if image.mode != 'RGB':
        image = image.convert('RGB')
        logger.debug("Converted to RGB")

    # Step 3: Inference
    logger.debug("Running inference...")
    start_time = time.time()
    results = model(image)
    inference_time = time.time() - start_time
    logger.debug(f"Inference completed in {inference_time:.3f}s")

    # Step 4: Post-process
    logger.debug(f"Found {len(results[0].boxes)} detections")

    return results
```

---

## 8. Environment Setup

### 8.1 Complete Setup Guide

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install PyTorch (with CUDA if available)
# Check CUDA version first: nvidia-smi
# Then go to https://pytorch.org/get-started/locally/

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision

# 5. Install other dependencies
pip install ultralytics opencv-python pillow streamlit fastapi uvicorn

# 6. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

### 8.2 CUDA Troubleshooting

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CUDA SETUP                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CHECK CUDA VERSION:                                                       │
│  ──────────────────                                                        │
│  nvidia-smi                                                                │
│                                                                             │
│  Output:                                                                   │
│  +-----------------------------------------------------------------------------+
│  | NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0    |
│  +-----------------------------------------------------------------------------+
│                                                                             │
│  PYTORCH + CUDA COMPATIBILITY:                                             │
│  ────────────────────────────                                              │
│  PyTorch CUDA version must match installed CUDA                           │
│                                                                             │
│  # Check PyTorch CUDA version                                              │
│  python -c "import torch; print(torch.version.cuda)"                      │
│                                                                             │
│  COMMON FIXES:                                                             │
│  ─────────────                                                             │
│                                                                             │
│  1. "CUDA not available" padahal ada GPU:                                 │
│     - Reinstall PyTorch dengan CUDA yang benar                            │
│     pip uninstall torch torchvision                                       │
│     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
│                                                                             │
│  2. Driver outdated:                                                       │
│     - Update NVIDIA driver dari nvidia.com                                │
│                                                                             │
│  3. Out of memory:                                                         │
│     - Reduce batch size                                                    │
│     - Close other GPU applications                                        │
│     - Use nvidia-smi to monitor usage                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. FAQ

### Q1: Bagaimana cara meningkatkan akurasi model?

**A:**

1. **Lebih banyak data** - Minimal 100-200 images per class
2. **Data berkualitas** - Annotation yang akurat
3. **Augmentasi** - Rotasi, flip, color jitter
4. **Model lebih besar** - YOLOv8m/l instead of n/s
5. **Fine-tuning** - Train lebih lama dengan patience tinggi
6. **Transfer learning** - Gunakan pretrained weights

### Q2: Kenapa model mendeteksi objek yang salah?

**A:**

1. **Cek annotation** - Mungkin ada label yang salah
2. **Confidence threshold** - Naikkan ke 0.5 atau lebih
3. **Training data** - Pastikan mirip dengan test data
4. **Class confusion** - Beberapa class mungkin mirip

### Q3: Bagaimana deploy ke production?

**A:**

1. **Docker** - Containerize aplikasi
2. **Cloud** - AWS, GCP, Azure, atau Railway
3. **Model optimization** - Export ke ONNX atau TensorRT
4. **Load balancing** - Untuk high traffic
5. **Monitoring** - Setup logging dan alerting

### Q4: Apakah bisa train tanpa GPU?

**A:**
Ya, tapi sangat lambat. Rekomendasi:

- Gunakan Google Colab (free GPU)
- Gunakan model kecil (YOLOv8n)
- Kurangi epoch dan batch size
- Pertimbangkan cloud GPU (AWS, GCP)

### Q5: Format gambar apa yang didukung?

**A:**
YOLO mendukung: JPG, JPEG, PNG, BMP, TIFF, WEBP
Rekomendasi: JPG untuk efficiency, PNG untuk transparency

---

## 10. Quick Reference

### 10.1 Command Cheatsheet

```bash
# Training
python train.py                          # Train dengan default config
python train.py --epochs 200             # Custom epochs

# Detection
python detect.py                         # Realtime webcam
python detect.py --source video.mp4      # Video file

# Web App
streamlit run web_app.py                 # Start web interface
streamlit run web_app.py --server.port 8080

# API
python api.py                            # Start API server
uvicorn api:app --reload --port 8000     # With hot reload

# Dataset
python convert_datasets.py               # Convert raw to YOLO
python split_and_prep.py                 # Split dataset
```

### 10.2 Important Paths

```python
# Model
MODEL_PATH = './models/best_model.pt'

# Dataset
DATA_YAML = './datasets/processed/data.yaml'
TRAIN_DIR = './datasets/processed/train/'
VAL_DIR = './datasets/processed/val/'
TEST_DIR = './datasets/processed/test/'

# Training outputs
RUNS_DIR = './runs/'
LOGS_DIR = './runs/logs/'
```

### 10.3 Key Parameters

```python
# Training
epochs = 120
batch = 20
imgsz = 640
patience = 10
optimizer = 'AdamW'
lr0 = 0.01

# Inference
confidence = 0.25
iou_threshold = 0.7
max_detections = 300
```

---

**Selamat! Anda telah menyelesaikan Modul 09: Tips dan Troubleshooting**

_Lanjut ke: [Modul 10 - Referensi](./10-referensi.md)_
