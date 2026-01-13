# 📖 Modul 02: Arsitektur Project

## Daftar Isi

1. [Overview Struktur Project](#1-overview-struktur-project)
2. [Penjelasan Setiap Direktori](#2-penjelasan-setiap-direktori)
3. [Penjelasan File Utama](#3-penjelasan-file-utama)
4. [Alur Data dalam Project](#4-alur-data-dalam-project)
5. [Konfigurasi Penting](#5-konfigurasi-penting)
6. [Dependencies dan Requirements](#6-dependencies-dan-requirements)
7. [Best Practices](#7-best-practices)
8. [Latihan](#8-latihan)

---

## 1. Overview Struktur Project

### 1.1 Struktur Lengkap

```
Klasifikasi_Sampah/
│
├── 📄 MAIN FILES (Entry Points)
│   ├── train.py              # Script untuk training model
│   ├── detect.py             # Real-time detection via webcam
│   ├── web_app.py            # Web application (Streamlit)
│   ├── api.py                # REST API (FastAPI)
│   ├── convert_datasets.py   # Konversi dataset ke format YOLO
│   └── split_and_prep.py     # Split dataset train/val/test
│
├── 📄 CONFIGURATION FILES
│   ├── data.yaml             # Konfigurasi dataset untuk YOLO
│   ├── requirements.txt      # Python dependencies
│   └── README.md             # Dokumentasi project
│
├── 📁 datasets/              # Semua data training
│   ├── raw/                  # Dataset mentah (belum diproses)
│   │   ├── Garbage Classification (Kaggle)/
│   │   └── garbage-classification-v2/
│   │
│   └── processed/            # Dataset yang sudah diproses
│       ├── all/              # Semua data gabungan
│       ├── train/            # Data training (80%)
│       ├── val/              # Data validation (10%)
│       └── test/             # Data testing (10%)
│
├── 📁 models/                # Model hasil training
│   ├── best_model.pt         # Model terbaik (untuk deployment)
│   └── best_model_backup_*.pt # Backup model sebelumnya
│
├── 📁 runs/                  # Output dari training
│   ├── detect/
│   │   ├── train/            # Hasil training
│   │   │   ├── weights/      # Model weights (.pt files)
│   │   │   ├── args.yaml     # Training arguments
│   │   │   └── results.csv   # Training metrics
│   │   └── val/              # Hasil validation
│   └── logs/                 # Log files
│
├── 📁 utils/                 # Utility modules
│   ├── __init__.py
│   ├── annotation_parsers.py # Parser berbagai format anotasi
│   ├── dataset_stats.py      # Statistik dataset
│   ├── image_utils.py        # Utilitas image processing
│   ├── label_mapper.py       # Mapping label/kelas
│   └── logger.py             # Logging configuration
│
├── 📁 notebooks/             # Jupyter notebooks
│   └── scan_image.ipynb      # Notebook untuk eksperimen
│
├── 📁 docs/                  # Dokumentasi
│   └── modul-pembelajaran/   # Modul yang sedang Anda baca
│
├── 📁 tests/                 # Unit tests
│
└── 📁 specs/                 # Spesifikasi dan planning
    └── 001-automated-pipeline/
```

### 1.2 Diagram Relasi Antar Komponen

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RELASI ANTAR KOMPONEN                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                       │
│  │   datasets/raw  │                                                       │
│  │  (raw data)     │                                                       │
│  └────────┬────────┘                                                       │
│           │                                                                 │
│           │ convert_datasets.py                                            │
│           ▼                                                                 │
│  ┌─────────────────┐                                                       │
│  │datasets/processed│                                                       │
│  │   /all          │                                                       │
│  └────────┬────────┘                                                       │
│           │                                                                 │
│           │ split_and_prep.py                                              │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────┐                               │
│  │         datasets/processed              │                               │
│  │  ┌─────────┬─────────┬─────────┐       │                               │
│  │  │ train/  │  val/   │  test/  │       │                               │
│  │  └────┬────┴────┬────┴────┬────┘       │                               │
│  └───────│─────────│─────────│─────────────┘                               │
│          │         │         │                                              │
│          └────┬────┴────┬────┘                                             │
│               │         │                                                   │
│               │data.yaml│                                                   │
│               ▼         │                                                   │
│  ┌─────────────────┐    │                                                   │
│  │    train.py     │    │                                                   │
│  │  (YOLOv8 API)   │    │                                                   │
│  └────────┬────────┘    │                                                   │
│           │             │                                                   │
│           ▼             │                                                   │
│  ┌─────────────────┐    │                                                   │
│  │  models/        │◄───┘                                                   │
│  │  best_model.pt  │         Evaluation                                    │
│  └────────┬────────┘                                                       │
│           │                                                                 │
│           ├───────────────┬───────────────┬───────────────┐                │
│           │               │               │               │                │
│           ▼               ▼               ▼               ▼                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐         │
│  │  web_app.py │  │   api.py    │  │  detect.py  │  │ notebooks │         │
│  │ (Streamlit) │  │  (FastAPI)  │  │  (OpenCV)   │  │ (Jupyter) │         │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Penjelasan Setiap Direktori

### 2.1 `datasets/` - Data Storage

```
datasets/
├── raw/                           # Dataset mentah (original)
│   ├── Garbage Classification (Kaggle)/
│   │   ├── cardboard/            # Gambar kardus
│   │   ├── glass/                # Gambar kaca
│   │   ├── metal/                # Gambar logam
│   │   ├── paper/                # Gambar kertas
│   │   ├── plastic/              # Gambar plastik
│   │   └── trash/                # Gambar sampah umum
│   │
│   └── garbage-classification-v2/
│       ├── battery/              # Gambar baterai
│       ├── biological/           # Gambar organik
│       ├── cardboard/
│       ├── clothes/              # Gambar pakaian
│       ├── glass/
│       ├── metal/
│       ├── paper/
│       ├── plastic/
│       ├── shoes/                # Gambar sepatu
│       └── trash/
│
└── processed/                    # Dataset yang sudah diproses
    ├── conversion_stats.json     # Statistik konversi
    ├── data.yaml                 # Config untuk YOLO
    │
    ├── all/                      # Semua data (sebelum split)
    │   ├── images/               # File gambar
    │   └── labels/               # File label (.txt)
    │
    ├── train/                    # Data training (80%)
    │   ├── images/
    │   └── labels/
    │
    ├── val/                      # Data validation (10%)
    │   ├── images/
    │   └── labels/
    │
    └── test/                     # Data testing (10%)
        ├── images/
        └── labels/
```

**Penjelasan:**

| Direktori          | Fungsi                                     |
| ------------------ | ------------------------------------------ |
| `raw/`             | Menyimpan dataset original yang didownload |
| `processed/all/`   | Data yang sudah dikonversi ke format YOLO  |
| `processed/train/` | Data untuk training model (80%)            |
| `processed/val/`   | Data untuk validasi selama training (10%)  |
| `processed/test/`  | Data untuk evaluasi akhir (10%)            |

### 2.2 `models/` - Trained Models

```
models/
├── best_model.pt                  # Model terbaik untuk deployment
├── best_model_backup_20251226_095856.pt  # Backup model sebelumnya
└── [other backups]
```

**Format File `.pt`:**

- File PyTorch yang berisi trained weights
- Dapat di-load dengan `YOLO('path/to/model.pt')`
- Termasuk architecture + weights

### 2.3 `runs/` - Training Outputs

```
runs/
├── detect/
│   ├── train/                    # Output dari training
│   │   ├── weights/
│   │   │   ├── best.pt          # Best model (highest mAP)
│   │   │   └── last.pt          # Last checkpoint
│   │   ├── args.yaml            # Training arguments yang digunakan
│   │   ├── results.csv          # Metrics per epoch
│   │   ├── results.png          # Plot metrics
│   │   ├── confusion_matrix.png # Confusion matrix
│   │   ├── F1_curve.png         # F1 score curve
│   │   ├── PR_curve.png         # Precision-Recall curve
│   │   └── ...
│   │
│   └── val/                      # Output dari validation
│       └── ...
│
└── logs/
    └── pipeline.log             # Log file dari semua operasi
```

### 2.4 `utils/` - Utility Modules

```
utils/
├── __init__.py                  # Package initialization
├── annotation_parsers.py        # Parser untuk berbagai format anotasi
├── dataset_stats.py             # Fungsi statistik dataset
├── image_utils.py               # Fungsi pemrosesan gambar
├── label_mapper.py              # Mapping kelas/label
└── logger.py                    # Konfigurasi logging
```

**Fungsi Setiap Module:**

| Module                  | Fungsi Utama                              |
| ----------------------- | ----------------------------------------- |
| `annotation_parsers.py` | Parse COCO, VOC, YOLO, CSV format         |
| `dataset_stats.py`      | Detect format, hitung statistik           |
| `image_utils.py`        | Verify image, hash untuk deduplication    |
| `label_mapper.py`       | Map berbagai nama kelas ke target classes |
| `logger.py`             | Setup console dan file logging            |

---

## 3. Penjelasan File Utama

### 3.1 `train.py` - Training Script

**Fungsi:** Melatih model YOLOv8 dengan dataset yang sudah disiapkan.

```python
# Struktur utama train.py

#!/usr/bin/env python3
"""
YOLOv8 Training Script for Klasifikasi Sampah
"""

import argparse
from ultralytics import YOLO

def get_device(device_arg):
    """Detect best available device (GPU/CPU)"""
    ...

def main():
    # 1. Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8s')
    parser.add_argument('--epochs', default=120)
    parser.add_argument('--batch', default=20)
    parser.add_argument('--imgsz', default=640)
    ...
    args = parser.parse_args()

    # 2. Load model
    model = YOLO(f'{args.model}.pt')

    # 3. Train
    model.train(
        data='./data.yaml',
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        ...
    )

    # 4. Save best model
    shutil.copy2('runs/detect/train/weights/best.pt',
                 'models/best_model.pt')

if __name__ == '__main__':
    main()
```

**Arguments:**

| Argument     | Default | Deskripsi                |
| ------------ | ------- | ------------------------ |
| `--model`    | yolov8s | Ukuran model (n/s/m/l/x) |
| `--epochs`   | 120     | Jumlah epoch training    |
| `--batch`    | 20      | Batch size               |
| `--imgsz`    | 640     | Input image size         |
| `--device`   | auto    | Device (cuda/cpu)        |
| `--patience` | 10      | Early stopping patience  |
| `--dry-run`  | False   | Test 1 epoch saja        |

### 3.2 `detect.py` - Real-time Detection

**Fungsi:** Deteksi sampah secara real-time menggunakan webcam.

```python
# Struktur utama detect.py

#!/usr/bin/env python3
"""
Real-time Waste Detection
"""

import cv2
from ultralytics import YOLO

# Config
MODEL = './models/best_model.pt'
CONF = 0.25

# Colors per class
COLORS = {
    'battery': (0, 0, 255),      # Red
    'biological': (0, 128, 0),    # Green
    ...
}

def load():
    """Load YOLO model"""
    model = YOLO(MODEL)
    return model

def draw(frame, box, idx, model, show_conf):
    """Draw bounding box on frame"""
    ...

def main():
    # 1. Load model
    model = load()

    # 2. Open camera
    cap = cv2.VideoCapture(0)

    # 3. Main loop
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 4. Run inference
        results = model(frame, conf=CONF)

        # 5. Draw results
        for box in results[0].boxes:
            draw(frame, box, ...)

        # 6. Display
        cv2.imshow('Detection', frame)

        # 7. Handle key press
        key = cv2.waitKey(1)
        if key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
```

**Keyboard Controls:**

| Key | Action                    |
| --- | ------------------------- |
| `Q` | Quit aplikasi             |
| `S` | Save screenshot           |
| `C` | Toggle confidence display |

### 3.3 `web_app.py` - Streamlit Web Application

**Fungsi:** Web interface untuk upload dan deteksi gambar.

```python
# Struktur utama web_app.py

#!/usr/bin/env python3
"""
Streamlit Web Application for Waste Classification
"""

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Config
MODEL_PATH = './models/best_model.pt'
DEFAULT_CONF = 0.25

# Class info dengan metadata
CLASS_INFO = {
    'battery': {
        'emoji': '🔋',
        'category': 'B3 (Berbahaya)',
        'bin_color': '🔴 Merah',
        'disposal': 'Tempat khusus limbah B3',
        'recyclable': False,
        'tips': 'Bawa ke drop point khusus baterai'
    },
    ...
}

@st.cache_resource
def load_model():
    """Load model dengan caching"""
    return YOLO(MODEL_PATH)

def main():
    # 1. Setup page config
    st.set_page_config(
        page_title="Klasifikasi Sampah",
        page_icon="♻️",
        layout="wide"
    )

    # 2. Sidebar (settings)
    with st.sidebar:
        confidence = st.slider("Confidence", 0.1, 0.9, 0.25)
        ...

    # 3. Main content
    st.title("♻️ Klasifikasi Sampah")

    # 4. File upload
    uploaded_file = st.file_uploader("Upload gambar...")

    if uploaded_file:
        # 5. Load image
        image = Image.open(uploaded_file)

        # 6. Run detection
        model = load_model()
        results = model(image, conf=confidence)

        # 7. Display results
        annotated = results[0].plot()
        st.image(annotated)

        # 8. Show details
        for det in results[0].boxes:
            ...

if __name__ == "__main__":
    main()
```

### 3.4 `api.py` - FastAPI REST API

**Fungsi:** REST API untuk integrasi dengan sistem lain.

```python
# Struktur utama api.py

#!/usr/bin/env python3
"""
FastAPI REST API for Waste Classification
"""

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO

app = FastAPI(
    title="Klasifikasi Sampah API",
    description="REST API for waste classification",
    version="1.0.0"
)

model = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    model = YOLO('./models/best_model.pt')

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Klasifikasi Sampah API"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/classes")
async def get_classes():
    """Get available classes"""
    return {"classes": CLASS_INFO}

@app.post("/detect")
async def detect_waste(
    file: UploadFile = File(...),
    confidence: float = 0.25
):
    """
    Detect waste in uploaded image

    Returns:
        - detections: list of detected objects
        - summary: recyclable vs non-recyclable count
    """
    image = Image.open(file.file)
    results = model(image, conf=confidence)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": ...,
            "confidence": ...,
            "bbox": ...
        })

    return {
        "success": True,
        "detections": detections,
        "summary": {...}
    }
```

**API Endpoints:**

| Method | Endpoint        | Deskripsi                       |
| ------ | --------------- | ------------------------------- |
| `GET`  | `/`             | Root, info API                  |
| `GET`  | `/health`       | Health check                    |
| `GET`  | `/classes`      | Daftar kelas                    |
| `POST` | `/detect`       | Deteksi sampah dari gambar      |
| `POST` | `/detect/image` | Deteksi, return annotated image |

### 3.5 `convert_datasets.py` - Dataset Converter

**Fungsi:** Mengkonversi berbagai format dataset ke format YOLO.

```python
# Struktur utama convert_datasets.py

#!/usr/bin/env python3
"""
Dataset Conversion Script
"""

from utils.annotation_parsers import (
    parse_coco_json,
    parse_voc_xml,
    parse_yolo_txt,
    create_class_folder_annotation
)
from utils.label_mapper import map_label, TARGET_CLASSES

def convert_coco(dataset_path, out_dir):
    """Convert COCO format to YOLO"""
    ...

def convert_voc(dataset_path, out_dir):
    """Convert Pascal VOC format to YOLO"""
    ...

def convert_class_folders(dataset_path, out_dir):
    """Convert class folder structure to YOLO"""
    ...

def main():
    # 1. Detect dataset format
    format = detect_format(dataset_path)

    # 2. Convert based on format
    if format == 'coco':
        convert_coco(...)
    elif format == 'voc':
        convert_voc(...)
    elif format == 'class_folders':
        convert_class_folders(...)

    # 3. Save statistics
    save_mapping(...)
```

### 3.6 `split_and_prep.py` - Dataset Splitter

**Fungsi:** Membagi dataset menjadi train/val/test dengan stratified sampling.

```python
# Struktur utama split_and_prep.py

#!/usr/bin/env python3
"""
Dataset Split and Preparation Script
"""

from sklearn.model_selection import train_test_split
import yaml

DEFAULT_SPLIT = (0.8, 0.1, 0.1)  # train, val, test

def deduplicate_images(images_dir, labels_dir):
    """Remove duplicate images using content hash"""
    ...

def split_dataset(images, labels_dir, ratios):
    """Stratified split into train/val/test"""
    train, temp = train_test_split(images, train_size=ratios[0], stratify=...)
    val, test = train_test_split(temp, ...)
    return train, val, test

def create_data_yaml(out_path, train_dir, val_dir, test_dir, classes):
    """Generate data.yaml for YOLO"""
    data = {
        'train': str(train_dir),
        'val': str(val_dir),
        'test': str(test_dir),
        'nc': len(classes),
        'names': classes
    }
    out_path.write_text(yaml.dump(data))

def main():
    # 1. Deduplicate
    unique_images = deduplicate_images(...)

    # 2. Split
    train, val, test = split_dataset(unique_images, ...)

    # 3. Copy files
    copy_files(train, ..., 'train')
    copy_files(val, ..., 'val')
    copy_files(test, ..., 'test')

    # 4. Generate data.yaml
    create_data_yaml(...)
```

---

## 4. Alur Data dalam Project

### 4.1 Pipeline Lengkap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: DATA PREPARATION                                                  │
│  ─────────────────────────                                                  │
│                                                                             │
│  [Download Dataset]                                                         │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────┐                                                       │
│  │ datasets/raw/   │  Format: Class folders, COCO, VOC, dll                │
│  └────────┬────────┘                                                       │
│           │                                                                 │
│           │  python convert_datasets.py                                    │
│           │                                                                 │
│           │  - Deteksi format otomatis                                     │
│           │  - Konversi ke YOLO format                                     │
│           │  - Mapping label ke 10 kelas target                            │
│           │  - Validasi gambar                                             │
│           ▼                                                                 │
│  ┌─────────────────┐                                                       │
│  │datasets/processed│                                                       │
│  │     /all        │  Format: YOLO (images/ + labels/)                     │
│  └────────┬────────┘                                                       │
│           │                                                                 │
│           │  python split_and_prep.py                                      │
│           │                                                                 │
│           │  - Deduplikasi gambar                                          │
│           │  - Stratified split (80/10/10)                                 │
│           │  - Generate data.yaml                                          │
│           ▼                                                                 │
│  ┌───────────────────────────────────────┐                                 │
│  │        datasets/processed             │                                 │
│  │  ┌─────────┬─────────┬─────────┐     │                                 │
│  │  │ train/  │  val/   │  test/  │     │  + data.yaml                    │
│  │  │  80%    │  10%    │  10%    │     │                                 │
│  │  └─────────┴─────────┴─────────┘     │                                 │
│  └───────────────────────────────────────┘                                 │
│                                                                             │
│                                                                             │
│  PHASE 2: TRAINING                                                          │
│  ─────────────────                                                          │
│                                                                             │
│  ┌─────────────────┐                                                       │
│  │    data.yaml    │  Config: paths, nc, class names                       │
│  └────────┬────────┘                                                       │
│           │                                                                 │
│           │  python train.py --model yolov8s --epochs 120                  │
│           │                                                                 │
│           │  - Load pretrained weights (COCO)                              │
│           │  - Fine-tune pada dataset kita                                 │
│           │  - Early stopping jika tidak improve                           │
│           │  - Save checkpoints                                            │
│           ▼                                                                 │
│  ┌─────────────────┐       ┌─────────────────┐                            │
│  │ runs/detect/    │       │    models/      │                            │
│  │   train/        │──────▶│ best_model.pt   │                            │
│  │  weights/       │ copy  │                 │                            │
│  └─────────────────┘       └────────┬────────┘                            │
│                                     │                                      │
│                                     │                                      │
│  PHASE 3: DEPLOYMENT                │                                      │
│  ───────────────────                │                                      │
│                                     │                                      │
│           ┌─────────────────────────┼─────────────────────────┐           │
│           │                         │                         │           │
│           ▼                         ▼                         ▼           │
│  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────┐     │
│  │   web_app.py    │       │     api.py      │       │  detect.py  │     │
│  │  (Streamlit)    │       │   (FastAPI)     │       │  (OpenCV)   │     │
│  │                 │       │                 │       │             │     │
│  │  Upload image   │       │  REST endpoint  │       │  Webcam     │     │
│  │  Interactive UI │       │  JSON response  │       │  Real-time  │     │
│  └─────────────────┘       └─────────────────┘       └─────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Format Data YOLO

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FORMAT DATA YOLO                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Struktur Folder:                                                           │
│  ────────────────                                                           │
│                                                                             │
│  train/                                                                     │
│  ├── images/                    # Gambar training                          │
│  │   ├── image001.jpg                                                      │
│  │   ├── image002.jpg                                                      │
│  │   └── ...                                                               │
│  │                                                                         │
│  └── labels/                    # Label untuk setiap gambar                │
│      ├── image001.txt           # Sama nama dengan gambar                  │
│      ├── image002.txt                                                      │
│      └── ...                                                               │
│                                                                             │
│                                                                             │
│  Format Label File (.txt):                                                  │
│  ─────────────────────────                                                  │
│                                                                             │
│  Setiap baris = 1 objek                                                    │
│                                                                             │
│  <class_id> <x_center> <y_center> <width> <height>                         │
│                                                                             │
│  Contoh (battery_105.txt):                                                 │
│  ┌──────────────────────────────────────────────────┐                      │
│  │ 0 0.4921875 0.4921875 0.984375 0.984375          │                      │
│  └──────────────────────────────────────────────────┘                      │
│                                                                             │
│  Penjelasan:                                                               │
│  - class_id = 0 (battery)                                                  │
│  - x_center = 0.4921875 (49.2% dari width gambar)                         │
│  - y_center = 0.4921875 (49.2% dari height gambar)                        │
│  - width = 0.984375 (98.4% dari width gambar)                             │
│  - height = 0.984375 (98.4% dari height gambar)                           │
│                                                                             │
│  ⚠️ PENTING: Semua koordinat NORMALIZED (0-1), bukan pixel absolut!       │
│                                                                             │
│                                                                             │
│  Visualisasi:                                                               │
│  ────────────                                                               │
│                                                                             │
│  ┌────────────────────────────────────────┐                                │
│  │                                        │                                │
│  │    (x_center, y_center)               │                                │
│  │            ●                          │                                │
│  │    ┌───────┼───────┐                  │                                │
│  │    │       │       │ ← height         │                                │
│  │    │       │       │                  │                                │
│  │    └───────────────┘                  │                                │
│  │          width                        │                                │
│  │                                        │                                │
│  └────────────────────────────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Konfigurasi Penting

### 5.1 `data.yaml` - Dataset Configuration

```yaml
# data.yaml - Konfigurasi dataset untuk YOLO

# Path ke dataset (absolut atau relatif)
train: ./datasets/processed/train/images
val: ./datasets/processed/val/images
test: ./datasets/processed/test/images

# Jumlah kelas
nc: 10

# Nama kelas (urutan = class_id)
names:
  0: battery # Baterai (limbah B3)
  1: biological # Sampah organik
  2: cardboard # Kardus
  3: clothes # Pakaian
  4: glass # Kaca
  5: metal # Logam
  6: paper # Kertas
  7: plastic # Plastik
  8: shoes # Sepatu
  9: trash # Sampah umum
```

**Penjelasan:**

- `train`, `val`, `test`: Path ke folder images
- `nc`: Number of classes
- `names`: Mapping class_id ke nama kelas

### 5.2 `requirements.txt` - Dependencies

```text
# Deep Learning (YOLOv8)
ultralytics>=8.0.0

# Image Processing
opencv-python>=4.8.0
Pillow>=10.0.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Label Matching
rapidfuzz>=3.0.0

# Utilities
tqdm>=4.66.0
pyyaml>=6.0.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0

# Web Application
streamlit>=1.28.0

# REST API
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6

# Jupyter (optional)
jupyter>=1.0.0
ipywidgets>=8.0.0
```

### 5.3 Konfigurasi Training Default

```python
# Default training configuration dalam train.py

DEFAULT_CONFIG = {
    'model': 'yolov8s',      # Model size (balanced)
    'epochs': 120,           # Cukup untuk konvergensi
    'batch': 20,             # Sesuaikan dengan GPU memory
    'imgsz': 640,            # Standard YOLO input size
    'patience': 10,          # Early stopping patience
    'optimizer': 'AdamW',    # Optimizer yang stabil
    'seed': 42,              # Reproducibility
}
```

---

## 6. Dependencies dan Requirements

### 6.1 Core Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DEPENDENCIES TREE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ultralytics (YOLOv8)                                                      │
│  ├── torch (PyTorch)                                                       │
│  │   └── torchvision                                                       │
│  ├── opencv-python                                                         │
│  ├── numpy                                                                 │
│  ├── pillow                                                                │
│  ├── pyyaml                                                                │
│  ├── matplotlib                                                            │
│  └── tqdm                                                                  │
│                                                                             │
│  streamlit (Web App)                                                       │
│  ├── pillow                                                                │
│  ├── numpy                                                                 │
│  └── altair (charting)                                                     │
│                                                                             │
│  fastapi (REST API)                                                        │
│  ├── uvicorn                                                               │
│  ├── starlette                                                             │
│  └── python-multipart                                                      │
│                                                                             │
│  scikit-learn (Data Splitting)                                             │
│  └── numpy                                                                 │
│                                                                             │
│  rapidfuzz (Label Matching)                                                │
│  └── (standalone)                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Instalasi dengan CUDA

```bash
# Step 1: Install PyTorch dengan CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Step 2: Install remaining requirements
pip install -r requirements.txt

# Step 3: Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLOv8: OK')"
```

---

## 7. Best Practices

### 7.1 Struktur Project

✅ **DO:**

- Pisahkan kode berdasarkan fungsi (training, inference, api)
- Gunakan utils/ untuk fungsi yang reusable
- Simpan model di folder terpisah (models/)
- Backup model sebelum overwrite
- Gunakan logging untuk tracking

❌ **DON'T:**

- Hardcode paths
- Simpan model weights di repository (gunakan .gitignore)
- Campurkan training dan inference code
- Skip validation step

### 7.2 Data Management

✅ **DO:**

- Simpan raw data terpisah dari processed data
- Dokumentasikan source dataset
- Verifikasi integritas gambar sebelum training
- Deduplikasi untuk menghindari data leakage

❌ **DON'T:**

- Modify raw data
- Split data secara random (gunakan stratified)
- Training tanpa validation set
- Mix train dan test data

### 7.3 Version Control

```
# .gitignore recommendations

# Data (terlalu besar)
datasets/
*.pt
*.pth

# Training outputs
runs/

# Environment
.venv/
__pycache__/

# IDE
.vscode/
.idea/
```

---

## 8. Latihan

### Latihan 1: Eksplorasi Struktur

1. Navigasi ke project directory dan list semua file Python
2. Buka `data.yaml` dan identifikasi:
   - Berapa jumlah kelas?
   - Apa path ke training images?
   - Kelas ID berapa untuk "plastic"?

### Latihan 2: Memahami Dependencies

1. Baca `requirements.txt`
2. Identifikasi library untuk:
   - Deep learning
   - Image processing
   - Web application
   - REST API

### Latihan 3: Analisis Alur Data

Gambarkan flowchart sederhana yang menunjukkan perjalanan data dari:

1. Dataset raw
2. Konversi ke YOLO
3. Split train/val/test
4. Training
5. Deployment

---

**Selamat! Anda telah menyelesaikan Modul 02: Arsitektur Project**

_Lanjut ke: [Modul 03 - Persiapan Dataset](./03-persiapan-dataset.md)_
