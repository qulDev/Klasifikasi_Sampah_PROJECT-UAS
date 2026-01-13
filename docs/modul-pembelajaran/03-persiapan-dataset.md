# 📖 Modul 03: Persiapan Dataset

## Daftar Isi

1. [Pengantar Dataset untuk Object Detection](#1-pengantar-dataset-untuk-object-detection)
2. [Format Dataset yang Didukung](#2-format-dataset-yang-didukung)
3. [Konversi Dataset dengan convert_datasets.py](#3-konversi-dataset-dengan-convert_datasetspy)
4. [Label Mapping](#4-label-mapping)
5. [Split Dataset dengan split_and_prep.py](#5-split-dataset-dengan-split_and_preppy)
6. [Validasi dan Quality Control](#6-validasi-dan-quality-control)
7. [Format YOLO Annotation](#7-format-yolo-annotation)
8. [Contoh Praktis](#8-contoh-praktis)
9. [Troubleshooting](#9-troubleshooting)
10. [Latihan](#10-latihan)

---

## 1. Pengantar Dataset untuk Object Detection

### 1.1 Komponen Dataset

Dataset untuk object detection terdiri dari dua komponen utama:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KOMPONEN DATASET OBJECT DETECTION                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────┐        │
│  │         IMAGES              │    │        ANNOTATIONS          │        │
│  │                             │    │                             │        │
│  │  ┌─────────────────────┐   │    │  ┌─────────────────────┐   │        │
│  │  │                     │   │    │  │ Bounding Box:       │   │        │
│  │  │    📷 image.jpg     │   │    │  │ - Koordinat (x,y,w,h)│   │        │
│  │  │                     │   │    │  │ - Class label       │   │        │
│  │  │  Pixel data         │   │    │  │ - Optional: conf    │   │        │
│  │  │  RGB channels       │   │    │  │                     │   │        │
│  │  └─────────────────────┘   │    │  └─────────────────────┘   │        │
│  │                             │    │                             │        │
│  │  Format:                    │    │  Format:                    │        │
│  │  - JPG, PNG, BMP           │    │  - YOLO (.txt)              │        │
│  │  - Berbagai ukuran         │    │  - COCO (.json)             │        │
│  │                             │    │  - VOC (.xml)               │        │
│  └─────────────────────────────┘    │  - CSV (.csv)               │        │
│                                      └─────────────────────────────┘        │
│                                                                             │
│  HUBUNGAN:                                                                  │
│  image001.jpg ──────────────▶ image001.txt (atau .json/.xml)               │
│  image002.jpg ──────────────▶ image002.txt                                  │
│  ...                                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Dataset dalam Project Ini

Project menggunakan dua sumber dataset:

| Dataset                   | Source | Kelas    | Jumlah Gambar |
| ------------------------- | ------ | -------- | ------------- |
| Garbage Classification    | Kaggle | 6 kelas  | ~2,500        |
| garbage-classification-v2 | Custom | 10 kelas | ~15,000       |

**10 Kelas Target:**

```
0: battery      - Baterai (limbah B3)
1: biological   - Sampah organik
2: cardboard    - Kardus
3: clothes      - Pakaian
4: glass        - Kaca
5: metal        - Logam
6: paper        - Kertas
7: plastic      - Plastik
8: shoes        - Sepatu
9: trash        - Sampah umum
```

---

## 2. Format Dataset yang Didukung

### 2.1 YOLO Format (Native)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FORMAT YOLO                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Struktur Folder:                                                           │
│  ────────────────                                                           │
│  dataset/                                                                   │
│  ├── images/                                                               │
│  │   ├── img001.jpg                                                        │
│  │   └── img002.jpg                                                        │
│  └── labels/                                                               │
│      ├── img001.txt                                                        │
│      └── img002.txt                                                        │
│                                                                             │
│  Format Label (.txt):                                                       │
│  ────────────────────                                                       │
│  <class_id> <x_center> <y_center> <width> <height>                         │
│                                                                             │
│  Contoh img001.txt:                                                         │
│  ┌──────────────────────────────────────────────┐                          │
│  │ 7 0.52 0.48 0.25 0.30                        │  ← plastic, normalized   │
│  │ 5 0.15 0.70 0.10 0.15                        │  ← metal                 │
│  └──────────────────────────────────────────────┘                          │
│                                                                             │
│  Keterangan:                                                               │
│  - class_id: Integer dimulai dari 0                                        │
│  - x_center, y_center: Center dari bbox (0-1)                              │
│  - width, height: Dimensi bbox (0-1)                                       │
│  - Semua nilai NORMALIZED (relatif terhadap ukuran gambar)                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 COCO Format (Microsoft)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FORMAT COCO                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Struktur:                                                                  │
│  ─────────                                                                  │
│  dataset/                                                                   │
│  ├── images/                                                               │
│  │   ├── img001.jpg                                                        │
│  │   └── img002.jpg                                                        │
│  └── annotations/                                                          │
│      └── instances_train.json                                              │
│                                                                             │
│  Format JSON:                                                               │
│  ─────────────                                                              │
│  {                                                                          │
│    "images": [                                                             │
│      {"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480}    │
│    ],                                                                      │
│    "annotations": [                                                        │
│      {                                                                     │
│        "id": 1,                                                            │
│        "image_id": 1,                                                      │
│        "category_id": 7,                                                   │
│        "bbox": [100, 150, 200, 180],  ← [x, y, width, height] ABSOLUTE    │
│        "area": 36000                                                       │
│      }                                                                     │
│    ],                                                                      │
│    "categories": [                                                         │
│      {"id": 7, "name": "plastic"}                                         │
│    ]                                                                       │
│  }                                                                          │
│                                                                             │
│  Keterangan:                                                               │
│  - bbox format: [x_top_left, y_top_left, width, height]                   │
│  - Koordinat ABSOLUTE (pixel values)                                       │
│  - Semua anotasi dalam 1 file JSON                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Pascal VOC Format

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FORMAT PASCAL VOC                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Struktur:                                                                  │
│  ─────────                                                                  │
│  dataset/                                                                   │
│  ├── JPEGImages/                                                           │
│  │   ├── img001.jpg                                                        │
│  │   └── img002.jpg                                                        │
│  └── Annotations/                                                          │
│      ├── img001.xml                                                        │
│      └── img002.xml                                                        │
│                                                                             │
│  Format XML:                                                                │
│  ───────────                                                                │
│  <annotation>                                                               │
│    <folder>images</folder>                                                  │
│    <filename>img001.jpg</filename>                                         │
│    <size>                                                                   │
│      <width>640</width>                                                    │
│      <height>480</height>                                                  │
│      <depth>3</depth>                                                      │
│    </size>                                                                  │
│    <object>                                                                 │
│      <name>plastic</name>                                                  │
│      <bndbox>                                                              │
│        <xmin>100</xmin>    ← Top-left x                                    │
│        <ymin>150</ymin>    ← Top-left y                                    │
│        <xmax>300</xmax>    ← Bottom-right x                                │
│        <ymax>330</ymax>    ← Bottom-right y                                │
│      </bndbox>                                                             │
│    </object>                                                                │
│  </annotation>                                                              │
│                                                                             │
│  Keterangan:                                                               │
│  - bbox format: [xmin, ymin, xmax, ymax]                                   │
│  - Koordinat ABSOLUTE                                                      │
│  - 1 file XML per gambar                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Class Folder Format (Classification)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FORMAT CLASS FOLDER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Struktur:                                                                  │
│  ─────────                                                                  │
│  dataset/                                                                   │
│  ├── battery/                                                              │
│  │   ├── bat001.jpg                                                        │
│  │   └── bat002.jpg                                                        │
│  ├── plastic/                                                              │
│  │   ├── pla001.jpg                                                        │
│  │   └── pla002.jpg                                                        │
│  ├── metal/                                                                │
│  │   └── met001.jpg                                                        │
│  └── ...                                                                   │
│                                                                             │
│  Keterangan:                                                               │
│  - Nama folder = nama kelas                                                │
│  - Setiap gambar dalam folder = instance kelas tersebut                    │
│  - TIDAK ADA bounding box (full image = single object)                     │
│  - Format ini untuk IMAGE CLASSIFICATION (bukan detection)                 │
│                                                                             │
│  Konversi ke YOLO:                                                          │
│  - Bbox = seluruh gambar: [0.5, 0.5, 1.0, 1.0]                             │
│  - Asumsi: 1 objek per gambar, objek memenuhi gambar                       │
│                                                                             │
│  📦 Dataset dalam project ini menggunakan format ini!                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.5 Perbandingan Format

| Aspek              | YOLO         | COCO            | VOC      | Class Folder |
| ------------------ | ------------ | --------------- | -------- | ------------ |
| **File per image** | 1 .txt       | 1 .json (semua) | 1 .xml   | None         |
| **Koordinat**      | Normalized   | Absolute        | Absolute | N/A          |
| **Bbox format**    | center, w, h | x, y, w, h      | xyxy     | Full image   |
| **Multi-object**   | ✓            | ✓               | ✓        | ✗            |
| **Human readable** | ✓            | Partially       | ✓        | ✓            |
| **File size**      | Kecil        | Besar           | Sedang   | None         |

---

## 3. Konversi Dataset dengan convert_datasets.py

### 3.1 Overview Script

`convert_datasets.py` adalah script utama untuk mengkonversi berbagai format dataset ke format YOLO.

```python
# Lokasi: ./convert_datasets.py

#!/usr/bin/env python3
"""
Dataset Conversion Script

Auto-detect and convert multiple dataset formats to YOLO format.

Simple usage:
    python convert_datasets.py
"""
```

### 3.2 Arsitektur Konversi

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ARSITEKTUR KONVERSI DATASET                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT                    PROSES                     OUTPUT                 │
│  ─────                    ──────                     ──────                 │
│                                                                             │
│  ┌─────────────┐    ┌──────────────────────┐    ┌─────────────────┐        │
│  │ COCO JSON   │    │                      │    │                 │        │
│  │ (.json)     │───▶│  parse_coco_json()   │───▶│                 │        │
│  └─────────────┘    └──────────────────────┘    │                 │        │
│                                                  │                 │        │
│  ┌─────────────┐    ┌──────────────────────┐    │   YOLO Format   │        │
│  │ Pascal VOC  │    │                      │    │                 │        │
│  │ (.xml)      │───▶│  parse_voc_xml()     │───▶│  images/        │        │
│  └─────────────┘    └──────────────────────┘    │  labels/        │        │
│                                                  │                 │        │
│  ┌─────────────┐    ┌──────────────────────┐    │  Normalized     │        │
│  │ Class Folder│    │ create_class_folder_ │    │  coordinates    │        │
│  │ (dirs)      │───▶│    annotation()      │───▶│                 │        │
│  └─────────────┘    └──────────────────────┘    │                 │        │
│                                                  │                 │        │
│  ┌─────────────┐    ┌──────────────────────┐    │                 │        │
│  │ CSV         │    │                      │    │                 │        │
│  │ (.csv)      │───▶│ parse_csv_annotations│───▶│                 │        │
│  └─────────────┘    └──────────────────────┘    └─────────────────┘        │
│                                                                             │
│                                                                             │
│  LABEL MAPPING                                                              │
│  ─────────────                                                              │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │                                                                │        │
│  │  Source Label    map_label()     Target Label                 │        │
│  │  ─────────────   ─────────────   ────────────                 │        │
│  │  "Plastik"    ──────────▶       "plastic" (ID: 7)            │        │
│  │  "METAL"      ──────────▶       "metal" (ID: 5)              │        │
│  │  "organic"    ──────────▶       "biological" (ID: 1)         │        │
│  │  "shoe"       ──────────▶       "shoes" (ID: 8)              │        │
│  │                                                                │        │
│  └────────────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Penjelasan Kode Line-by-Line

#### Import dan Konfigurasi

```python
#!/usr/bin/env python3
"""
Dataset Conversion Script
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

# Import utility functions
from utils.annotation_parsers import (
    parse_coco_json,      # Parser untuk COCO format
    parse_voc_xml,        # Parser untuk VOC format
    parse_yolo_txt,       # Parser untuk YOLO format (existing)
    parse_csv_annotations,# Parser untuk CSV format
    create_class_folder_annotation,  # Untuk class folder format
    BoundingBox,          # Class untuk bounding box
)
from utils.dataset_stats import detect_dataset_format  # Auto-detect format
from utils.image_utils import verify_image  # Validasi gambar
from utils.label_mapper import TARGET_CLASSES, map_label, MANUAL_CLASS_MAPPINGS
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Default paths
DEFAULT_SRC = './datasets/raw'        # Sumber dataset
DEFAULT_DST = './datasets/processed/all'  # Destinasi
DEFAULT_CLASSES = TARGET_CLASSES       # 10 kelas target
```

#### Fungsi Utama

```python
def copy_image_and_label(img_path: Path, annotations, out_dir: Path) -> int:
    """
    Copy image dan create YOLO label file.

    Args:
        img_path: Path ke file gambar sumber
        annotations: List of (class_id, BoundingBox) tuples
        out_dir: Direktori output

    Returns:
        Jumlah anotasi yang berhasil disimpan
    """
    # Step 1: Verify image is valid
    is_valid, error = verify_image(img_path)
    if not is_valid:
        logger.warning(f"Invalid: {img_path.name} - {error}")
        return 0

    # Step 2: Copy image ke destination
    dest_img = out_dir / 'images' / img_path.name
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, dest_img)

    # Step 3: Create label file
    if annotations:
        dest_lbl = out_dir / 'labels' / f'{img_path.stem}.txt'
        dest_lbl.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_lbl, 'w') as f:
            for class_id, bbox in annotations:
                # Format: class_id x_center y_center width height
                f.write(bbox.to_yolo_line(class_id) + '\n')

        return len(annotations)
    return 0
```

#### Fungsi Konversi per Format

```python
def convert_class_folders(dataset_path: Path, out_dir: Path,
                          class_map: Dict, dry_run: bool) -> Tuple[int, int]:
    """
    Convert class folder structure ke YOLO format.

    Dataset format:
    dataset/
    ├── plastic/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── metal/
    │   └── img3.jpg
    └── ...

    Setiap gambar dalam folder = 1 instance dari kelas tersebut.
    Bounding box = seluruh gambar (full image detection).
    """
    logger.info(f"Converting class folders: {dataset_path.name}")

    # Get all class directories
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    # Filter out non-class directories (TRAIN, TEST, etc)
    valid_class_dirs = []
    for d in class_dirs:
        if d.name.upper() in ['TRAIN', 'TEST', 'VAL', 'VALIDATION']:
            # Recursively add subdirectories
            valid_class_dirs.extend([sd for sd in d.iterdir() if sd.is_dir()])
        else:
            valid_class_dirs.append(d)

    if dry_run:
        # Count total images without actually converting
        total = sum(
            len(list(d.glob('*.jpg')) + list(d.glob('*.png')))
            for d in valid_class_dirs
        )
        logger.info(f"[DRY RUN] {total} images from {len(valid_class_dirs)} classes")
        return total, total

    num_img, num_ann = 0, 0

    for class_dir in tqdm(valid_class_dirs, desc="Classes"):
        class_name = class_dir.name.lower()

        # Map source class ke target class
        target_class, method, confidence = map_label(class_name)

        if target_class is None:
            logger.warning(f"Unknown class: {class_name}")
            continue

        # Get class ID
        class_id = TARGET_CLASSES.index(target_class)

        # Process all images in this class folder
        image_files = list(class_dir.glob('*.jpg')) + \
                     list(class_dir.glob('*.png')) + \
                     list(class_dir.glob('*.jpeg'))

        for img_path in image_files:
            # Create full-image bounding box
            # Karena class folder = 1 objek per gambar, bbox = full image
            bbox = create_class_folder_annotation()  # Returns [0.5, 0.5, 1.0, 1.0]
            annotations = [(class_id, bbox)]

            # Copy image and create label
            num_ann += copy_image_and_label(img_path, annotations, out_dir)
            num_img += 1

    logger.info(f"Converted {num_img} images, {num_ann} annotations")
    return num_img, num_ann
```

### 3.4 Cara Penggunaan

```bash
# Basic usage (dengan semua default)
python convert_datasets.py

# Specify source dan destination
python convert_datasets.py --src ./my_dataset --dst ./output

# Dry run (preview tanpa actual conversion)
python convert_datasets.py --dry-run

# Verbose output
python convert_datasets.py --verbose
```

### 3.5 Output Statistik

Setelah konversi, file `conversion_stats.json` akan dibuat:

```json
{
  "target_classes": [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
  ],
  "statistics": {
    "total_images": 15234,
    "total_annotations": 15234,
    "per_class": {
      "battery": 1250,
      "biological": 1850,
      "cardboard": 1523,
      "clothes": 1420,
      "glass": 1380,
      "metal": 1425,
      "paper": 1520,
      "plastic": 1680,
      "shoes": 1186,
      "trash": 2000
    },
    "sources_converted": [
      "Garbage Classification (Kaggle)",
      "garbage-classification-v2"
    ]
  }
}
```

---

## 4. Label Mapping

### 4.1 Masalah Heterogenitas Label

Dataset dari berbagai sumber sering memiliki nama kelas yang berbeda:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MASALAH HETEROGENITAS LABEL                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Dataset A             Dataset B              Dataset C                     │
│  ─────────             ─────────              ─────────                     │
│  "plastic"             "Plastik"              "PLASTIC_BOTTLE"              │
│  "metal"               "Metal Cans"           "aluminium"                   │
│  "organic"             "biological"           "food_waste"                  │
│  "cardboard"           "Karton"               "paper_board"                 │
│                                                                             │
│                         ↓                                                   │
│                                                                             │
│            Semua harus dipetakan ke TARGET CLASSES:                         │
│            ────────────────────────────────────────                         │
│            [battery, biological, cardboard, clothes,                        │
│             glass, metal, paper, plastic, shoes, trash]                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Label Mapper (utils/label_mapper.py)

```python
# utils/label_mapper.py

# Target classes (standard output)
TARGET_CLASSES = [
    "battery",     # 0
    "biological",  # 1
    "cardboard",   # 2
    "clothes",     # 3
    "glass",       # 4
    "metal",       # 5
    "paper",       # 6
    "plastic",     # 7
    "shoes",       # 8
    "trash",       # 9
]

# Keyword-based mapping
MATERIAL_KEYWORDS = {
    "battery": ["battery", "batteries", "cell", "lithium", "alkaline"],
    "biological": ["biological", "organic", "food", "compost", "vegetable"],
    "cardboard": ["cardboard", "carton", "box", "packaging", "corrugated"],
    "clothes": ["clothes", "clothing", "textile", "fabric", "shirt"],
    "glass": ["glass", "jar", "bottle", "window", "mirror"],
    "metal": ["metal", "aluminum", "steel", "can", "tin", "foil"],
    "paper": ["paper", "newspaper", "magazine", "book", "document"],
    "plastic": ["plastic", "pet", "hdpe", "pvc", "bottle", "wrapper"],
    "shoes": ["shoes", "shoe", "footwear", "sneaker", "boot"],
    "trash": ["trash", "garbage", "waste", "rubbish", "mixed"],
}

# Manual exact mappings
MANUAL_CLASS_MAPPINGS = {
    "battery": "battery",
    "biological": "biological",
    "organic": "biological",      # alias
    "food": "biological",         # alias
    "cardboard": "cardboard",
    "clothes": "clothes",
    "textile": "clothes",         # alias
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "shoes": "shoes",
    "footwear": "shoes",          # alias
    "trash": "trash",
    "garbage": "trash",           # alias
    "other": "trash",             # catchall
}


def map_label(source_label: str) -> Tuple[str, str, float]:
    """
    Map a source label to target class using multiple strategies.

    Strategies (in order):
    1. Exact match
    2. Manual mapping
    3. Keyword matching
    4. Fuzzy matching (rapidfuzz)

    Returns:
        (target_class, mapping_method, confidence)
    """
    normalized = source_label.lower().strip()

    # Strategy 1: Exact match
    if normalized in TARGET_CLASSES:
        return normalized, "exact", 1.0

    # Strategy 2: Manual mapping
    if normalized in MANUAL_CLASS_MAPPINGS:
        return MANUAL_CLASS_MAPPINGS[normalized], "manual", 1.0

    # Strategy 3: Keyword matching
    for target, keywords in MATERIAL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized:
                return target, "keyword", 0.9

    # Strategy 4: Fuzzy matching
    from rapidfuzz import fuzz, process
    best_match, score, _ = process.extractOne(
        normalized,
        TARGET_CLASSES,
        scorer=fuzz.ratio
    )
    if score >= 80:
        return best_match, "fuzzy", score / 100

    # No match found
    return None, "none", 0.0
```

### 4.3 Alur Label Mapping

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ALUR LABEL MAPPING                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: "Plastic_Bottle"                                                    │
│                │                                                            │
│                ▼                                                            │
│  ┌──────────────────────────┐                                              │
│  │   normalize_label()      │                                              │
│  │   "plastic_bottle"       │                                              │
│  │   → "plastic bottle"     │                                              │
│  └───────────┬──────────────┘                                              │
│              │                                                              │
│              ▼                                                              │
│  ┌──────────────────────────┐                                              │
│  │ 1. Exact Match?          │  "plastic bottle" in TARGET_CLASSES?         │
│  │    NO → continue         │                                              │
│  └───────────┬──────────────┘                                              │
│              │                                                              │
│              ▼                                                              │
│  ┌──────────────────────────┐                                              │
│  │ 2. Manual Mapping?       │  "plastic bottle" in MANUAL_MAPPINGS?        │
│  │    NO → continue         │                                              │
│  └───────────┬──────────────┘                                              │
│              │                                                              │
│              ▼                                                              │
│  ┌──────────────────────────┐                                              │
│  │ 3. Keyword Match?        │  "plastic" keyword found!                    │
│  │    YES → return          │  → return ("plastic", "keyword", 0.9)        │
│  └───────────┬──────────────┘                                              │
│              │                                                              │
│              ▼                                                              │
│  Output: ("plastic", "keyword", 0.9)                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Split Dataset dengan split_and_prep.py

### 5.1 Overview

`split_and_prep.py` membagi dataset menjadi train/val/test dengan:

- **Deduplication**: Menghapus gambar duplikat
- **Stratified Split**: Menjaga proporsi kelas di setiap split
- **data.yaml Generation**: Membuat config file untuk YOLO

### 5.2 Penjelasan Kode

```python
# split_and_prep.py

#!/usr/bin/env python3
"""
Dataset Split and Preparation Script
"""

from sklearn.model_selection import train_test_split
import yaml

DEFAULT_SPLIT = (0.8, 0.1, 0.1)  # 80% train, 10% val, 10% test
DEFAULT_CLASSES = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]


def deduplicate_images(images_dir: Path, labels_dir: Path) -> Tuple[List[Path], int]:
    """
    Menghapus gambar duplikat berdasarkan content hash.

    Mengapa penting?
    - Mencegah data leakage (gambar sama di train & test)
    - Mengurangi overfitting
    - Menghemat storage
    """
    logger.info("Deduplicating images...")

    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    seen_hashes = {}
    unique_images = []
    dup_count = 0

    for img_path in tqdm(image_files, desc="Hashing images"):
        # Verify image valid
        is_valid, error = verify_image(img_path)
        if not is_valid:
            logger.warning(f"Invalid: {img_path.name} - {error}")
            continue

        # Compute content hash
        img_hash = hash_image(img_path)  # SHA-256 hash of file content

        if img_hash in seen_hashes:
            dup_count += 1
            # Gambar duplikat ditemukan, skip
        else:
            seen_hashes[img_hash] = img_path
            unique_images.append(img_path)

    logger.info(f"Found {len(unique_images)} unique, {dup_count} duplicates")
    return unique_images, dup_count


def split_dataset(
    images: List[Path],
    labels_dir: Path,
    ratios: Tuple[float, float, float],  # (train, val, test)
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Stratified split: menjaga proporsi kelas di setiap split.

    Contoh:
    Jika dataset punya 1000 plastic, 500 metal
    Train (80%): 800 plastic, 400 metal
    Val (10%):   100 plastic, 50 metal
    Test (10%):  100 plastic, 50 metal
    """
    logger.info(f"Splitting with ratios {ratios}...")

    # Step 1: Get class labels for stratification
    valid_images = []
    image_classes = []

    for img in tqdm(images, desc="Reading labels"):
        class_ids = get_labels(img, labels_dir)  # Get class IDs from label file
        if class_ids:
            valid_images.append(img)
            image_classes.append(class_ids[0])  # Use first class for stratification
        else:
            logger.warning(f"No labels: {img.name}")

    if not valid_images:
        raise ValueError("No valid images with labels!")

    # Step 2: First split - train vs (val+test)
    train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
        valid_images,
        image_classes,
        train_size=ratios[0],      # 0.8 untuk train
        stratify=image_classes,    # Stratified sampling
        random_state=seed          # Reproducibility
    )

    # Step 3: Second split - val vs test
    val_size = ratios[1] / (ratios[1] + ratios[2])  # 0.1 / 0.2 = 0.5
    val_imgs, test_imgs, _, _ = train_test_split(
        temp_imgs,
        temp_lbls,
        train_size=val_size,
        stratify=temp_lbls,
        random_state=seed
    )

    logger.info(f"Split: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    return train_imgs, val_imgs, test_imgs


def create_data_yaml(out_path: Path, train_dir: Path, val_dir: Path,
                     test_dir: Path, classes: List[str]):
    """
    Generate data.yaml untuk YOLO training.

    File ini memberitahu YOLO:
    - Di mana data training/validation/test
    - Berapa jumlah kelas
    - Nama-nama kelas
    """
    data = {
        'train': str(train_dir.resolve()),  # Absolute path
        'val': str(val_dir.resolve()),
        'test': str(test_dir.resolve()),
        'nc': len(classes),                  # Number of classes
        'names': classes                     # Class names
    }

    out_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    logger.info(f"Generated: {out_path}")
```

### 5.3 Visualisasi Split Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SPLIT PROCESS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  datasets/processed/all/  (Input: 15,234 images)                           │
│  ├── images/                                                               │
│  └── labels/                                                               │
│                                                                             │
│           │                                                                 │
│           │ 1. Deduplicate (remove ~500 duplicates)                        │
│           ▼                                                                 │
│                                                                             │
│  14,734 unique images                                                      │
│                                                                             │
│           │                                                                 │
│           │ 2. Stratified Split (80/10/10)                                 │
│           ▼                                                                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  TRAIN (80%)          VAL (10%)           TEST (10%)               │   │
│  │  11,787 images        1,473 images        1,474 images             │   │
│  │                                                                     │   │
│  │  Class Distribution (Proportional):                                 │   │
│  │  ┌─────────────┬─────────────┬─────────────┬─────────────────┐     │   │
│  │  │ Class       │ Train       │ Val         │ Test            │     │   │
│  │  ├─────────────┼─────────────┼─────────────┼─────────────────┤     │   │
│  │  │ battery     │ 1,000       │ 125         │ 125             │     │   │
│  │  │ biological  │ 1,480       │ 185         │ 185             │     │   │
│  │  │ cardboard   │ 1,218       │ 152         │ 153             │     │   │
│  │  │ ...         │ ...         │ ...         │ ...             │     │   │
│  │  └─────────────┴─────────────┴─────────────┴─────────────────┘     │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│           │                                                                 │
│           │ 3. Copy to final directories                                   │
│           ▼                                                                 │
│                                                                             │
│  datasets/processed/                                                        │
│  ├── train/                                                                │
│  │   ├── images/ (11,787 files)                                           │
│  │   └── labels/ (11,787 files)                                           │
│  ├── val/                                                                  │
│  │   ├── images/ (1,473 files)                                            │
│  │   └── labels/ (1,473 files)                                            │
│  ├── test/                                                                 │
│  │   ├── images/ (1,474 files)                                            │
│  │   └── labels/ (1,474 files)                                            │
│  └── data.yaml                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Cara Penggunaan

```bash
# Basic usage (default 80/10/10 split)
python split_and_prep.py

# Custom split ratio
python split_and_prep.py --split 0.7 0.15 0.15

# Custom source dan output
python split_and_prep.py --src ./my_data --out ./processed

# Different random seed
python split_and_prep.py --seed 123
```

---

## 6. Validasi dan Quality Control

### 6.1 Image Validation (utils/image_utils.py)

```python
def verify_image(image_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Verify bahwa gambar valid dan dapat dibaca.

    Checks:
    1. File exists
    2. File is readable
    3. Valid image format (JPEG, PNG, BMP)
    4. Valid dimensions (width > 0, height > 0)
    5. Not corrupted
    """
    if not image_path.exists():
        return False, f"File does not exist: {image_path}"

    if not image_path.is_file():
        return False, f"Path is not a file: {image_path}"

    try:
        with Image.open(image_path) as img:
            img.verify()  # Check for corruption

        # Re-open untuk check dimensi
        with Image.open(image_path) as img:
            width, height = img.size
            if width <= 0 or height <= 0:
                return False, f"Invalid dimensions: {width}x{height}"

            if img.format not in ['JPEG', 'PNG', 'BMP', 'JPG']:
                return False, f"Unsupported format: {img.format}"

        return True, None

    except Exception as e:
        return False, f"Failed to open image: {str(e)}"
```

### 6.2 Deduplication (hash-based)

```python
def hash_image(image_path: Path) -> str:
    """
    Compute SHA-256 hash dari content gambar.

    Digunakan untuk mendeteksi duplikat:
    - Gambar yang persis sama → hash sama
    - Gambar berbeda → hash berbeda (dengan probabilitas sangat tinggi)
    """
    sha256_hash = hashlib.sha256()

    with open(image_path, "rb") as f:
        # Read in chunks (memory efficient untuk file besar)
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()
```

---

## 7. Format YOLO Annotation

### 7.1 Struktur File Label

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FORMAT ANNOTATION YOLO                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  File: image001.txt                                                         │
│  ───────────────────                                                        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 7 0.456 0.523 0.234 0.312                                            │  │
│  │ 5 0.712 0.234 0.156 0.189                                            │  │
│  │ 9 0.234 0.867 0.345 0.098                                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Format setiap baris:                                                       │
│  <class_id> <x_center> <y_center> <width> <height>                         │
│                                                                             │
│  Penjelasan baris 1: "7 0.456 0.523 0.234 0.312"                           │
│  ─────────────────────────────────────────────────                         │
│  • class_id = 7 → plastic                                                  │
│  • x_center = 0.456 → center di 45.6% width gambar                        │
│  • y_center = 0.523 → center di 52.3% height gambar                       │
│  • width = 0.234 → bbox width = 23.4% dari gambar                         │
│  • height = 0.312 → bbox height = 31.2% dari gambar                       │
│                                                                             │
│                                                                             │
│  Visualisasi pada gambar 640x480:                                          │
│  ─────────────────────────────────                                         │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │  (0,0)                                                640      │        │
│  │    ┌───────────────────────────────────────────────────────┐   │        │
│  │    │                                                       │   │        │
│  │    │                 ┌─────────────┐                      │   │        │
│  │    │                 │  plastic    │  center: (292, 251)  │   │        │
│  │    │                 │  ID: 7      │  size: 150x150       │   │        │
│  │    │                 └─────────────┘                      │   │        │
│  │    │                                                       │   │        │
│  │    │       ┌──────┐                                       │   │        │
│  │    │       │metal │                                       │   │        │
│  │    │       │ ID:5 │                                       │   │        │
│  │    │       └──────┘                         ┌─────┐       │   │        │
│  │    │                                        │trash│       │   │        │
│  │    │                                        │ID:9 │       │   │        │
│  │ 480│                                        └─────┘       │   │        │
│  │    └───────────────────────────────────────────────────────┘   │        │
│  └────────────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Konversi Koordinat

```python
# utils/annotation_parsers.py

class BoundingBox:
    """Normalized bounding box in YOLO format."""

    def __init__(self, x_center: float, y_center: float,
                 width: float, height: float):
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    @classmethod
    def from_coco(cls, x: float, y: float, w: float, h: float,
                  img_width: int, img_height: int) -> 'BoundingBox':
        """
        Convert from COCO format (absolute top-left xywh).

        COCO: [x_topleft, y_topleft, width, height] (pixels)
        YOLO: [x_center, y_center, width, height] (normalized 0-1)
        """
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        return cls(x_center, y_center, width, height)

    @classmethod
    def from_voc(cls, xmin: float, ymin: float, xmax: float, ymax: float,
                 img_width: int, img_height: int) -> 'BoundingBox':
        """
        Convert from Pascal VOC format (absolute xyxy).

        VOC: [xmin, ymin, xmax, ymax] (pixels)
        YOLO: [x_center, y_center, width, height] (normalized 0-1)
        """
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        return cls(x_center, y_center, width, height)

    def to_yolo_line(self, class_id: int) -> str:
        """Convert to YOLO annotation line."""
        return f"{class_id} {self.x_center:.6f} {self.y_center:.6f} " \
               f"{self.width:.6f} {self.height:.6f}"
```

---

## 8. Contoh Praktis

### 8.1 Full Pipeline Execution

```bash
# Step 1: Download/prepare raw dataset
# Letakkan dataset di ./datasets/raw/

# Step 2: Convert ke YOLO format
python convert_datasets.py
# Output: ./datasets/processed/all/

# Step 3: Split dataset
python split_and_prep.py
# Output:
#   ./datasets/processed/train/
#   ./datasets/processed/val/
#   ./datasets/processed/test/
#   ./data.yaml

# Step 4: Verify
ls datasets/processed/train/images | wc -l  # Count training images
cat data.yaml  # Check configuration
```

### 8.2 Custom Dataset

Jika Anda memiliki dataset sendiri dengan format class folder:

```
my_dataset/
├── botol_plastik/
│   ├── img1.jpg
│   └── img2.jpg
├── kaleng/
│   └── img3.jpg
└── kertas/
    └── img4.jpg
```

Langkah-langkah:

```python
# 1. Tambahkan mapping di utils/label_mapper.py
MANUAL_CLASS_MAPPINGS.update({
    "botol_plastik": "plastic",
    "kaleng": "metal",
    "kertas": "paper",
})

# 2. Jalankan konversi
# python convert_datasets.py --src ./my_dataset
```

---

## 9. Troubleshooting

### 9.1 Masalah Umum

| Masalah                 | Penyebab             | Solusi                             |
| ----------------------- | -------------------- | ---------------------------------- |
| "Unknown class" warning | Label tidak dikenali | Tambahkan ke MANUAL_CLASS_MAPPINGS |
| "Invalid image"         | Gambar corrupt       | Hapus atau ganti gambar            |
| Banyak duplikat         | Dataset overlap      | Normal, akan di-deduplicate        |
| Split tidak balance     | Kelas minority       | Pertimbangkan data augmentation    |

### 9.2 Validasi Manual

```python
# Script untuk validasi dataset

from pathlib import Path
import random

# Check random samples
images_dir = Path('./datasets/processed/train/images')
labels_dir = Path('./datasets/processed/train/labels')

samples = random.sample(list(images_dir.glob('*.jpg')), 5)

for img in samples:
    label = labels_dir / f'{img.stem}.txt'
    print(f"\nImage: {img.name}")
    if label.exists():
        print(f"Label: {label.read_text()}")
    else:
        print("WARNING: No label file!")
```

---

## 10. Latihan

### Latihan 1: Format Konversi

Diberikan bounding box dalam format COCO:

- Image size: 640 x 480
- COCO bbox: [100, 120, 200, 150] (x, y, w, h absolute)

Convert ke format YOLO!

### Latihan 2: Label Mapping

Bagaimana Anda akan menambahkan mapping untuk dataset dengan kelas:

- "botol_kaca"
- "newspaper"
- "baju_bekas"

### Latihan 3: Custom Split

Jelaskan mengapa stratified split lebih baik daripada random split untuk dataset klasifikasi!

---

**Selamat! Anda telah menyelesaikan Modul 03: Persiapan Dataset**

_Lanjut ke: [Modul 04 - Training Model](./04-training-model.md)_
