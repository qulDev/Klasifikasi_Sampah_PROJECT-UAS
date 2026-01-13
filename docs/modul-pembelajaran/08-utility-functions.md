# 📖 Modul 08: Utility Functions

## Daftar Isi

1. [Overview Utils Module](#1-overview-utils-module)
2. [Annotation Parsers](#2-annotation-parsers)
3. [Label Mapper](#3-label-mapper)
4. [Image Utils](#4-image-utils)
5. [Logger](#5-logger)
6. [Dataset Stats](#6-dataset-stats)
7. [Best Practices](#7-best-practices)
8. [Extending Utils](#8-extending-utils)
9. [Testing Utils](#9-testing-utils)
10. [Latihan](#10-latihan)

---

## 1. Overview Utils Module

### 1.1 Struktur Direktori

```
utils/
├── __init__.py           # Package initialization
├── annotation_parsers.py # Parse berbagai format annotation
├── label_mapper.py       # Map label names to IDs
├── image_utils.py        # Image validation & processing
├── dataset_stats.py      # Dataset analysis tools
└── logger.py             # Logging configuration
```

### 1.2 Design Philosophy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       UTILITY MODULE DESIGN                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PRINSIP UTAMA:                                                            │
│  ──────────────                                                            │
│                                                                             │
│  1. SINGLE RESPONSIBILITY                                                  │
│     Setiap modul punya satu fokus spesifik                                │
│                                                                             │
│     ┌─────────────────┐                                                    │
│     │ annotation_     │ → Parse annotation files                          │
│     │ parsers.py      │                                                   │
│     └─────────────────┘                                                    │
│                                                                             │
│     ┌─────────────────┐                                                    │
│     │ label_mapper.py │ → Map label names ↔ IDs                           │
│     └─────────────────┘                                                    │
│                                                                             │
│     ┌─────────────────┐                                                    │
│     │ image_utils.py  │ → Image validation & hashing                      │
│     └─────────────────┘                                                    │
│                                                                             │
│  2. REUSABILITY                                                            │
│     Functions yang bisa dipakai di multiple scripts                       │
│                                                                             │
│     convert_datasets.py ──┬──→ annotation_parsers.py                      │
│                           │                                                │
│     split_and_prep.py ────┴──→ label_mapper.py                           │
│                           │                                                │
│     train.py ─────────────┴──→ logger.py                                 │
│                                                                             │
│  3. TESTABILITY                                                            │
│     Pure functions yang mudah di-unit test                                │
│                                                                             │
│     Input → Function → Output  (tanpa side effects)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Annotation Parsers

### 2.1 File: annotation_parsers.py

Module ini berisi parser untuk berbagai format annotation dalam object detection.

### 2.2 BoundingBox Class

```python
"""
annotation_parsers.py
====================
Parse berbagai format annotation object detection.

Supported formats:
- YOLO: <class_id> <x_center> <y_center> <width> <height> (normalized)
- COCO: {"bbox": [x, y, width, height], "category_id": int}
- Pascal VOC: XML format with <bndbox><xmin>...</xmax>...</bndbox>
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import xml.etree.ElementTree as ET
import json
from pathlib import Path


@dataclass
class BoundingBox:
    """
    Representasi bounding box yang format-agnostic.

    Menyimpan dalam format normalized (0-1) untuk konsistensi.

    Attributes:
        x_center: Center X coordinate (normalized 0-1)
        y_center: Center Y coordinate (normalized 0-1)
        width: Box width (normalized 0-1)
        height: Box height (normalized 0-1)
        class_id: Class ID (integer)
        class_name: Optional class name (string)
        confidence: Optional confidence score (float 0-1)
    """
    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int
    class_name: str = ""
    confidence: float = 1.0
```

### 2.3 Format Conversion Methods

```python
    def to_yolo(self) -> str:
        """
        Convert ke YOLO format string.

        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        Semua values normalized (0-1)

        Returns:
            String dalam format YOLO

        Example:
            >>> box = BoundingBox(0.5, 0.5, 0.3, 0.4, 2)
            >>> box.to_yolo()
            '2 0.500000 0.500000 0.300000 0.400000'
        """
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def to_xyxy(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Convert ke pixel coordinates [x1, y1, x2, y2].

        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Tuple of (x1, y1, x2, y2) in pixel coordinates

        Example:
            >>> box = BoundingBox(0.5, 0.5, 0.2, 0.3, 0)
            >>> box.to_xyxy(640, 480)
            (256, 168, 384, 312)
        """
        x1 = int((self.x_center - self.width / 2) * img_width)
        y1 = int((self.y_center - self.height / 2) * img_height)
        x2 = int((self.x_center + self.width / 2) * img_width)
        y2 = int((self.y_center + self.height / 2) * img_height)
        return (x1, y1, x2, y2)

    def to_xywh(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Convert ke pixel coordinates [x, y, width, height].
        (COCO format - top-left origin)

        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Tuple of (x, y, w, h) in pixel coordinates
        """
        x = int((self.x_center - self.width / 2) * img_width)
        y = int((self.y_center - self.height / 2) * img_height)
        w = int(self.width * img_width)
        h = int(self.height * img_height)
        return (x, y, w, h)
```

### 2.4 Class Methods untuk Parsing

```python
    @classmethod
    def from_yolo(cls, line: str, class_names: Dict[int, str] = None) -> 'BoundingBox':
        """
        Parse dari YOLO format.

        Args:
            line: String format "<class_id> <x_center> <y_center> <width> <height>"
            class_names: Optional mapping dari class_id ke class_name

        Returns:
            BoundingBox instance

        Example:
            >>> BoundingBox.from_yolo("2 0.5 0.5 0.3 0.4")
            BoundingBox(x_center=0.5, y_center=0.5, width=0.3, height=0.4, class_id=2)
        """
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        class_name = ""
        if class_names and class_id in class_names:
            class_name = class_names[class_id]

        return cls(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            class_id=class_id,
            class_name=class_name
        )

    @classmethod
    def from_coco(cls, annotation: Dict, img_width: int, img_height: int,
                  category_map: Dict[int, str] = None) -> 'BoundingBox':
        """
        Parse dari COCO annotation format.

        COCO bbox format: [x, y, width, height] (top-left, pixels)

        Args:
            annotation: COCO annotation dict dengan "bbox" dan "category_id"
            img_width: Image width untuk normalisasi
            img_height: Image height untuk normalisasi
            category_map: Optional mapping category_id ke name

        Returns:
            BoundingBox instance
        """
        bbox = annotation['bbox']  # [x, y, w, h]
        category_id = annotation['category_id']

        # Convert to center format and normalize
        x, y, w, h = bbox
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height

        class_name = category_map.get(category_id, "") if category_map else ""

        return cls(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            class_id=category_id,
            class_name=class_name
        )

    @classmethod
    def from_voc(cls, xml_element: ET.Element, img_width: int, img_height: int,
                 class_to_id: Dict[str, int] = None) -> 'BoundingBox':
        """
        Parse dari Pascal VOC XML element.

        VOC format:
        <object>
            <name>class_name</name>
            <bndbox>
                <xmin>100</xmin>
                <ymin>150</ymin>
                <xmax>300</xmax>
                <ymax>400</ymax>
            </bndbox>
        </object>

        Args:
            xml_element: XML <object> element
            img_width: Image width untuk normalisasi
            img_height: Image height untuk normalisasi
            class_to_id: Mapping class name ke class ID

        Returns:
            BoundingBox instance
        """
        class_name = xml_element.find('name').text
        bndbox = xml_element.find('bndbox')

        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Convert to center format and normalize
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        class_id = class_to_id.get(class_name, 0) if class_to_id else 0

        return cls(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            class_id=class_id,
            class_name=class_name
        )
```

### 2.5 File Parser Functions

```python
def parse_yolo_file(filepath: Path, class_names: Dict[int, str] = None) -> List[BoundingBox]:
    """
    Parse YOLO annotation file.

    Args:
        filepath: Path ke .txt file
        class_names: Optional class ID to name mapping

    Returns:
        List of BoundingBox objects
    """
    boxes = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                boxes.append(BoundingBox.from_yolo(line, class_names))
    return boxes


def parse_coco_file(filepath: Path) -> Dict:
    """
    Parse COCO JSON annotation file.

    Returns:
        Dict dengan 'images', 'annotations', 'categories'
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def parse_voc_file(filepath: Path, class_to_id: Dict[str, int] = None) -> List[BoundingBox]:
    """
    Parse Pascal VOC XML annotation file.

    Args:
        filepath: Path ke .xml file
        class_to_id: Class name to ID mapping

    Returns:
        List of BoundingBox objects
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    # Get image dimensions
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    boxes = []
    for obj in root.findall('object'):
        box = BoundingBox.from_voc(obj, img_width, img_height, class_to_id)
        boxes.append(box)

    return boxes
```

### 2.6 Format Conversion Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FORMAT CONVERSION FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT FORMATS                    INTERMEDIATE                OUTPUT       │
│  ─────────────                    ────────────                ──────       │
│                                                                             │
│  ┌─────────────────┐                                                       │
│  │    COCO JSON    │                                                       │
│  │                 │                                                       │
│  │ {"bbox": [...], │                                                       │
│  │  "category_id"} │──┐                                                    │
│  └─────────────────┘  │           ┌─────────────────┐                      │
│                       │           │                 │                      │
│  ┌─────────────────┐  │           │  BoundingBox    │   ┌───────────────┐ │
│  │   PASCAL VOC    │  ├──────────→│                 │──→│   YOLO TXT    │ │
│  │                 │  │           │  x_center       │   │               │ │
│  │  <xmin>100</..> │  │           │  y_center       │   │ "0 0.5 0.5   │ │
│  │  <ymin>150</..> │──┤           │  width          │   │  0.3 0.4"    │ │
│  └─────────────────┘  │           │  height         │   │               │ │
│                       │           │  class_id       │   └───────────────┘ │
│  ┌─────────────────┐  │           │                 │                      │
│  │  Class Folders  │  │           └─────────────────┘                      │
│  │                 │──┘                                                    │
│  │  plastic/       │     Normalized format (0-1)                          │
│  │  metal/         │     → Consistent representation                      │
│  │  ...            │     → Easy conversion to any format                  │
│  └─────────────────┘                                                       │
│                                                                             │
│  Kenapa intermediate format?                                               │
│  ─────────────────────────────                                             │
│  1. Single source of truth                                                │
│  2. Easier to validate                                                    │
│  3. Format-agnostic processing                                            │
│  4. Simpler testing                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Label Mapper

### 3.1 File: label_mapper.py

Module untuk mapping label names ke IDs dengan fuzzy matching.

### 3.2 Core Implementation

```python
"""
label_mapper.py
==============
Map berbagai variasi label names ke standard class IDs.

Menggunakan fuzzy matching untuk handle:
- Typos: "platic" → "plastic"
- Case differences: "PLASTIC" → "plastic"
- Variations: "plastics" → "plastic"
"""

from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz, process

# Standard class names (10 kelas)
STANDARD_CLASSES = [
    'battery',      # 0
    'biological',   # 1
    'cardboard',    # 2
    'clothes',      # 3
    'glass',        # 4
    'metal',        # 5
    'paper',        # 6
    'plastic',      # 7
    'shoes',        # 8
    'trash',        # 9
]

# Class name to ID mapping
CLASS_TO_ID = {name: idx for idx, name in enumerate(STANDARD_CLASSES)}

# ID to class name mapping
ID_TO_CLASS = {idx: name for idx, name in enumerate(STANDARD_CLASSES)}

# Known aliases (common variations)
ALIASES = {
    # Battery variations
    'batteries': 'battery',
    'baterai': 'battery',
    'cell': 'battery',

    # Biological variations
    'organic': 'biological',
    'organik': 'biological',
    'bio': 'biological',
    'food': 'biological',
    'food waste': 'biological',

    # Cardboard variations
    'karton': 'cardboard',
    'carton': 'cardboard',
    'box': 'cardboard',
    'boxes': 'cardboard',

    # Clothes variations
    'clothing': 'clothes',
    'fabric': 'clothes',
    'textile': 'clothes',
    'pakaian': 'clothes',

    # Glass variations
    'kaca': 'glass',
    'bottle': 'glass',
    'bottles': 'glass',

    # Metal variations
    'aluminium': 'metal',
    'aluminum': 'metal',
    'steel': 'metal',
    'iron': 'metal',
    'can': 'metal',
    'cans': 'metal',
    'tin': 'metal',
    'logam': 'metal',

    # Paper variations
    'kertas': 'paper',
    'newspaper': 'paper',
    'magazine': 'paper',

    # Plastic variations
    'plastik': 'plastic',
    'plastics': 'plastic',
    'pet': 'plastic',
    'hdpe': 'plastic',
    'ldpe': 'plastic',

    # Shoes variations
    'shoe': 'shoes',
    'footwear': 'shoes',
    'sepatu': 'shoes',

    # Trash variations
    'garbage': 'trash',
    'waste': 'trash',
    'rubbish': 'trash',
    'sampah': 'trash',
    'residue': 'trash',
    'residu': 'trash',
}
```

### 3.3 LabelMapper Class

```python
class LabelMapper:
    """
    Map label names ke standard class IDs dengan fuzzy matching.

    Attributes:
        classes: List of standard class names
        class_to_id: Dict mapping class name to ID
        id_to_class: Dict mapping ID to class name
        aliases: Dict of known aliases
        threshold: Minimum fuzzy match score (0-100)
    """

    def __init__(
        self,
        classes: List[str] = None,
        aliases: Dict[str, str] = None,
        threshold: int = 80
    ):
        """
        Initialize mapper.

        Args:
            classes: Custom class list (default: STANDARD_CLASSES)
            aliases: Additional aliases to include
            threshold: Minimum fuzzy match score (0-100)
        """
        self.classes = classes or STANDARD_CLASSES
        self.class_to_id = {name: idx for idx, name in enumerate(self.classes)}
        self.id_to_class = {idx: name for idx, name in enumerate(self.classes)}

        # Combine default aliases with custom
        self.aliases = ALIASES.copy()
        if aliases:
            self.aliases.update(aliases)

        self.threshold = threshold

        # Cache for fuzzy matches
        self._cache: Dict[str, Optional[str]] = {}

    def map_label(self, label: str) -> Tuple[Optional[int], str, int]:
        """
        Map label string ke class ID.

        Args:
            label: Input label string

        Returns:
            Tuple of (class_id or None, mapped_name, confidence_score)

        Example:
            >>> mapper = LabelMapper()
            >>> mapper.map_label("PLASTIC")
            (7, 'plastic', 100)

            >>> mapper.map_label("platic")  # typo
            (7, 'plastic', 91)

            >>> mapper.map_label("unknown_thing")
            (None, '', 45)
        """
        # Normalize input
        label_lower = label.lower().strip()

        # Check cache
        if label_lower in self._cache:
            cached = self._cache[label_lower]
            if cached:
                return (
                    self.class_to_id[cached],
                    cached,
                    100 if cached == label_lower else 90
                )
            return (None, '', 0)

        # 1. Exact match
        if label_lower in self.class_to_id:
            self._cache[label_lower] = label_lower
            return (self.class_to_id[label_lower], label_lower, 100)

        # 2. Check aliases
        if label_lower in self.aliases:
            mapped = self.aliases[label_lower]
            self._cache[label_lower] = mapped
            return (self.class_to_id[mapped], mapped, 100)

        # 3. Fuzzy match
        result = process.extractOne(
            label_lower,
            self.classes,
            scorer=fuzz.ratio
        )

        if result and result[1] >= self.threshold:
            matched_name, score, _ = result
            self._cache[label_lower] = matched_name
            return (self.class_to_id[matched_name], matched_name, score)

        # No match found
        self._cache[label_lower] = None
        return (None, '', result[1] if result else 0)

    def get_class_name(self, class_id: int) -> str:
        """Get class name from ID."""
        return self.id_to_class.get(class_id, "unknown")

    def get_class_id(self, class_name: str) -> Optional[int]:
        """Get class ID from name."""
        return self.class_to_id.get(class_name.lower())
```

### 3.4 Fuzzy Matching Explained

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FUZZY MATCHING                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEM: Label variations in different datasets                           │
│  ────────────────────────────────────────────────                          │
│                                                                             │
│  Dataset A: "plastic"                                                      │
│  Dataset B: "Plastic"     ← Case difference                               │
│  Dataset C: "plastics"    ← Plural form                                   │
│  Dataset D: "platic"      ← Typo                                          │
│  Dataset E: "plastik"     ← Indonesian                                    │
│                                                                             │
│  SOLUTION: Fuzzy string matching                                           │
│  ─────────────────────────────────                                         │
│                                                                             │
│  Levenshtein Distance (edit distance):                                     │
│  "plastic" vs "platic"                                                     │
│     p l a s t i c                                                          │
│     p l   a t i c   ← Missing 's'                                         │
│     Edit distance = 1 (1 insertion needed)                                │
│     Similarity = 1 - (1/7) = 85.7%                                        │
│                                                                             │
│  RapidFuzz Library:                                                        │
│  ──────────────────                                                        │
│                                                                             │
│  from rapidfuzz import fuzz                                                │
│                                                                             │
│  fuzz.ratio("plastic", "platic")    # 92 (high similarity)               │
│  fuzz.ratio("plastic", "metal")     # 33 (low similarity)                │
│  fuzz.ratio("plastic", "plastik")   # 85 (similar)                       │
│                                                                             │
│  Threshold Setting:                                                        │
│  ─────────────────                                                         │
│  threshold = 80 → Accept if score >= 80                                   │
│                                                                             │
│  "platic"  → plastic (92)  ✓ Accept                                       │
│  "plastik" → plastic (85)  ✓ Accept                                       │
│  "metal"   → plastic (33)  ✗ Reject                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Image Utils

### 4.1 File: image_utils.py

Utility functions untuk image validation dan processing.

### 4.2 Implementation

```python
"""
image_utils.py
=============
Image validation dan processing utilities.

Functions:
- validate_image: Check if file is valid image
- get_image_hash: SHA-256 hash untuk deduplication
- get_image_size: Get width x height
- find_images: Find all images in directory
"""

from pathlib import Path
from typing import Tuple, List, Optional, Set
import hashlib
from PIL import Image
import os

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def validate_image(filepath: Path) -> Tuple[bool, str]:
    """
    Validate apakah file adalah gambar yang valid.

    Checks:
    1. File exists
    2. Extension is supported
    3. Can be opened by PIL
    4. Has valid dimensions

    Args:
        filepath: Path ke file gambar

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> validate_image(Path("test.jpg"))
        (True, "")

        >>> validate_image(Path("corrupt.jpg"))
        (False, "Cannot open image: corrupt file")
    """
    # Check exists
    if not filepath.exists():
        return (False, f"File not found: {filepath}")

    # Check extension
    ext = filepath.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return (False, f"Unsupported extension: {ext}")

    # Try to open
    try:
        with Image.open(filepath) as img:
            img.verify()  # Verify integrity

        # Reopen to get dimensions (verify() closes the file)
        with Image.open(filepath) as img:
            width, height = img.size

            if width <= 0 or height <= 0:
                return (False, f"Invalid dimensions: {width}x{height}")

            if width < 32 or height < 32:
                return (False, f"Image too small: {width}x{height} (min 32x32)")

    except Exception as e:
        return (False, f"Cannot open image: {str(e)}")

    return (True, "")


def get_image_hash(filepath: Path, algorithm: str = 'sha256') -> str:
    """
    Calculate hash dari image file untuk deduplication.

    Args:
        filepath: Path ke image file
        algorithm: Hash algorithm ('sha256', 'md5', etc.)

    Returns:
        Hex string hash

    Example:
        >>> get_image_hash(Path("test.jpg"))
        'a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a'
    """
    hasher = hashlib.new(algorithm)

    with open(filepath, 'rb') as f:
        # Read in chunks untuk handle large files
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_image_size(filepath: Path) -> Tuple[int, int]:
    """
    Get image dimensions without loading entire image.

    Args:
        filepath: Path ke image file

    Returns:
        Tuple of (width, height)
    """
    with Image.open(filepath) as img:
        return img.size


def find_images(
    directory: Path,
    recursive: bool = True,
    extensions: Set[str] = None
) -> List[Path]:
    """
    Find semua image files dalam directory.

    Args:
        directory: Directory untuk di-scan
        recursive: Include subdirectories
        extensions: Custom extensions (default: SUPPORTED_EXTENSIONS)

    Returns:
        List of image file paths
    """
    extensions = extensions or SUPPORTED_EXTENSIONS
    images = []

    if recursive:
        for ext in extensions:
            images.extend(directory.rglob(f"*{ext}"))
            images.extend(directory.rglob(f"*{ext.upper()}"))
    else:
        for ext in extensions:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(images)


def find_duplicates(directory: Path) -> Dict[str, List[Path]]:
    """
    Find duplicate images berdasarkan file hash.

    Args:
        directory: Directory untuk di-scan

    Returns:
        Dict mapping hash ke list of duplicate paths
    """
    hash_to_files: Dict[str, List[Path]] = {}

    for img_path in find_images(directory):
        try:
            img_hash = get_image_hash(img_path)

            if img_hash not in hash_to_files:
                hash_to_files[img_hash] = []

            hash_to_files[img_hash].append(img_path)

        except Exception as e:
            print(f"Warning: Cannot hash {img_path}: {e}")

    # Filter to only duplicates
    duplicates = {
        hash_val: paths
        for hash_val, paths in hash_to_files.items()
        if len(paths) > 1
    }

    return duplicates
```

### 4.3 Deduplication Explained

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       IMAGE DEDUPLICATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEM: Duplicate images in dataset                                      │
│  ──────────────────────────────────────                                    │
│                                                                             │
│  raw/dataset1/plastic/bottle1.jpg  ─┐                                      │
│  raw/dataset2/plastic/IMG_001.jpg  ─┼─ Same image, different names!       │
│  raw/dataset3/plastik/botol.jpg    ─┘                                      │
│                                                                             │
│  Impact:                                                                   │
│  - Data leakage (same image in train & test)                              │
│  - Biased model (overfit to duplicates)                                   │
│  - Inflated dataset size                                                   │
│                                                                             │
│  SOLUTION: Content-based hashing                                           │
│  ────────────────────────────────                                          │
│                                                                             │
│  File content → SHA-256 → Unique hash                                      │
│                                                                             │
│  bottle1.jpg  → a7ffc6f8bf1ed76651c14756a...                              │
│  IMG_001.jpg  → a7ffc6f8bf1ed76651c14756a...  ← Same hash!               │
│  botol.jpg    → a7ffc6f8bf1ed76651c14756a...  ← Same hash!               │
│  bottle2.jpg  → b5c0b187fe309af0f4d35982a...  ← Different                 │
│                                                                             │
│  Same hash = Same content = Duplicate!                                     │
│                                                                             │
│  Algorithm: SHA-256                                                        │
│  ─────────────────                                                         │
│  - Cryptographic hash function                                             │
│  - 256-bit output (64 hex characters)                                     │
│  - Collision-resistant (virtually no false positives)                     │
│  - Fast computation                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Logger

### 5.1 File: logger.py

Centralized logging configuration untuk semua scripts.

### 5.2 Implementation

```python
"""
logger.py
========
Logging configuration untuk project.

Features:
- Console output dengan colors
- File logging (rotating)
- Configurable log levels
- Structured format
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Log directory
LOG_DIR = Path('./runs/logs')


class ColorFormatter(logging.Formatter):
    """
    Custom formatter dengan warna untuk console output.

    Colors:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red bold
    """

    # ANSI color codes
    COLORS = {
        logging.DEBUG: '\033[36m',    # Cyan
        logging.INFO: '\033[32m',     # Green
        logging.WARNING: '\033[33m',  # Yellow
        logging.ERROR: '\033[31m',    # Red
        logging.CRITICAL: '\033[31;1m',  # Red bold
    }
    RESET = '\033[0m'

    def format(self, record):
        # Get color for level
        color = self.COLORS.get(record.levelno, '')

        # Format message
        message = super().format(record)

        # Add color
        if color:
            message = f"{color}{message}{self.RESET}"

        return message


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger dengan console dan file handlers.

    Args:
        name: Logger name (biasanya __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path untuk file logging
        console: Enable console output

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Starting process...")
        2024-01-15 10:30:00 - __main__ - INFO - Starting process...
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColorFormatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Create directory if needed
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


def get_default_logger(name: str = "waste_classifier") -> logging.Logger:
    """
    Get logger dengan default configuration.

    Creates log file in runs/logs/ dengan timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = LOG_DIR / f"{name}_{timestamp}.log"

    return setup_logger(name, log_file=log_file)
```

### 5.3 Usage Example

```python
# Di script lain
from utils.logger import setup_logger, get_default_logger

# Option 1: Custom setup
logger = setup_logger(
    __name__,
    level=logging.DEBUG,
    log_file=Path('./logs/train.log')
)

# Option 2: Default setup
logger = get_default_logger()

# Usage
logger.debug("Debug message")       # Cyan
logger.info("Info message")         # Green
logger.warning("Warning message")   # Yellow
logger.error("Error message")       # Red
logger.critical("Critical!")        # Red bold
```

---

## 6. Dataset Stats

### 6.1 File: dataset_stats.py

Analisis dan statistik dataset.

### 6.2 Implementation

```python
"""
dataset_stats.py
===============
Analyze dataset statistics.

Metrics:
- Class distribution
- Image dimensions
- Annotation statistics
- Data quality checks
"""

from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import json

from .image_utils import find_images, get_image_size
from .annotation_parsers import parse_yolo_file


def analyze_yolo_dataset(data_yaml_path: Path) -> Dict:
    """
    Analyze YOLO format dataset.

    Args:
        data_yaml_path: Path ke data.yaml

    Returns:
        Dict dengan statistics
    """
    import yaml

    with open(data_yaml_path) as f:
        config = yaml.safe_load(f)

    base_dir = data_yaml_path.parent
    stats = {
        'num_classes': config.get('nc', 0),
        'class_names': config.get('names', []),
        'splits': {}
    }

    for split in ['train', 'val', 'test']:
        split_path = base_dir / split / 'images'
        if not split_path.exists():
            continue

        images = find_images(split_path)
        labels_dir = base_dir / split / 'labels'

        # Count annotations per class
        class_counts = Counter()
        total_boxes = 0
        images_with_labels = 0

        for img_path in images:
            label_path = labels_dir / f"{img_path.stem}.txt"

            if label_path.exists():
                images_with_labels += 1
                boxes = parse_yolo_file(label_path)
                total_boxes += len(boxes)

                for box in boxes:
                    class_counts[box.class_id] += 1

        stats['splits'][split] = {
            'total_images': len(images),
            'images_with_labels': images_with_labels,
            'total_annotations': total_boxes,
            'avg_annotations_per_image': total_boxes / len(images) if images else 0,
            'class_distribution': dict(class_counts)
        }

    return stats


def print_dataset_stats(stats: Dict):
    """
    Pretty print dataset statistics.
    """
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    print(f"\nNumber of Classes: {stats['num_classes']}")
    print(f"Class Names: {', '.join(stats['class_names'])}")

    for split_name, split_stats in stats['splits'].items():
        print(f"\n{split_name.upper()} Split:")
        print("-"*40)
        print(f"  Total Images: {split_stats['total_images']}")
        print(f"  Images with Labels: {split_stats['images_with_labels']}")
        print(f"  Total Annotations: {split_stats['total_annotations']}")
        print(f"  Avg Annotations/Image: {split_stats['avg_annotations_per_image']:.2f}")

        print("\n  Class Distribution:")
        for class_id, count in sorted(split_stats['class_distribution'].items()):
            class_name = stats['class_names'][class_id] if class_id < len(stats['class_names']) else 'unknown'
            print(f"    {class_id}: {class_name}: {count}")

    print("\n" + "="*60)
```

---

## 7. Best Practices

### 7.1 Writing Good Utilities

```python
# ✅ GOOD: Pure function, no side effects
def normalize_bbox(x, y, w, h, img_w, img_h):
    """Normalize coordinates to 0-1 range."""
    return (
        x / img_w,
        y / img_h,
        w / img_w,
        h / img_h
    )

# ❌ BAD: Side effects, modifies global state
global_result = None
def normalize_bbox(x, y, w, h):
    global global_result
    global_result = (x / 640, y / 480, w / 640, h / 480)  # Hardcoded!
```

### 7.2 Error Handling

```python
# ✅ GOOD: Descriptive errors
def parse_annotation(filepath: Path) -> List[BoundingBox]:
    if not filepath.exists():
        raise FileNotFoundError(f"Annotation file not found: {filepath}")

    if filepath.suffix != '.txt':
        raise ValueError(f"Expected .txt file, got: {filepath.suffix}")

    # ... parsing logic

# ❌ BAD: Silent failures
def parse_annotation(filepath):
    try:
        with open(filepath) as f:
            return f.read()
    except:
        return None  # Silently returns None!
```

### 7.3 Documentation

```python
def convert_coco_to_yolo(
    coco_file: Path,
    output_dir: Path,
    class_mapping: Dict[int, int] = None
) -> Tuple[int, int]:
    """
    Convert COCO format annotations to YOLO format.

    Args:
        coco_file: Path to COCO JSON file
        output_dir: Directory for YOLO .txt files
        class_mapping: Optional mapping from COCO category_id to YOLO class_id
            If None, uses category_id as-is.

    Returns:
        Tuple of (successful_conversions, failed_conversions)

    Raises:
        FileNotFoundError: If coco_file doesn't exist
        json.JSONDecodeError: If coco_file is invalid JSON

    Example:
        >>> convert_coco_to_yolo(
        ...     Path("annotations.json"),
        ...     Path("labels/"),
        ...     class_mapping={1: 0, 2: 1, 3: 2}
        ... )
        (1000, 5)

    Note:
        COCO bbox format [x, y, w, h] is converted to YOLO [x_center, y_center, w, h]
        with values normalized to 0-1 range.
    """
    pass
```

---

## 8. Extending Utils

### 8.1 Adding New Utility

```python
# utils/augmentation.py
"""
Augmentation utilities untuk training.
"""

from PIL import Image
import numpy as np
from typing import Tuple


def random_flip(image: Image.Image, prob: float = 0.5) -> Image.Image:
    """Randomly flip image horizontally."""
    if np.random.random() < prob:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def random_rotate(image: Image.Image, max_angle: float = 15) -> Image.Image:
    """Randomly rotate image by small angle."""
    angle = np.random.uniform(-max_angle, max_angle)
    return image.rotate(angle, expand=True)


def color_jitter(
    image: Image.Image,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2
) -> Image.Image:
    """Apply random color adjustments."""
    from PIL import ImageEnhance

    # Brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1 + np.random.uniform(-brightness, brightness))

    # Contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1 + np.random.uniform(-contrast, contrast))

    # Saturation
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1 + np.random.uniform(-saturation, saturation))

    return image
```

### 8.2 Update **init**.py

```python
# utils/__init__.py
"""
Utility modules untuk Klasifikasi Sampah project.

Modules:
- annotation_parsers: Parse YOLO, COCO, VOC formats
- label_mapper: Map label names to IDs
- image_utils: Image validation dan processing
- logger: Logging configuration
- dataset_stats: Dataset analysis
- augmentation: Image augmentation (new!)
"""

from .annotation_parsers import BoundingBox, parse_yolo_file, parse_coco_file, parse_voc_file
from .label_mapper import LabelMapper, STANDARD_CLASSES, CLASS_TO_ID, ID_TO_CLASS
from .image_utils import validate_image, get_image_hash, find_images, find_duplicates
from .logger import setup_logger, get_default_logger
from .dataset_stats import analyze_yolo_dataset, print_dataset_stats

# Version
__version__ = '1.0.0'

# Public API
__all__ = [
    # annotation_parsers
    'BoundingBox',
    'parse_yolo_file',
    'parse_coco_file',
    'parse_voc_file',

    # label_mapper
    'LabelMapper',
    'STANDARD_CLASSES',
    'CLASS_TO_ID',
    'ID_TO_CLASS',

    # image_utils
    'validate_image',
    'get_image_hash',
    'find_images',
    'find_duplicates',

    # logger
    'setup_logger',
    'get_default_logger',

    # dataset_stats
    'analyze_yolo_dataset',
    'print_dataset_stats',
]
```

---

## 9. Testing Utils

### 9.1 Unit Tests

```python
# tests/test_annotation_parsers.py
import pytest
from pathlib import Path
from utils.annotation_parsers import BoundingBox


class TestBoundingBox:

    def test_from_yolo_basic(self):
        """Test parsing basic YOLO format."""
        box = BoundingBox.from_yolo("2 0.5 0.5 0.3 0.4")

        assert box.class_id == 2
        assert box.x_center == 0.5
        assert box.y_center == 0.5
        assert box.width == 0.3
        assert box.height == 0.4

    def test_to_yolo(self):
        """Test conversion to YOLO format."""
        box = BoundingBox(0.5, 0.5, 0.3, 0.4, 2)
        yolo_str = box.to_yolo()

        assert yolo_str == "2 0.500000 0.500000 0.300000 0.400000"

    def test_to_xyxy(self):
        """Test conversion to pixel coordinates."""
        box = BoundingBox(0.5, 0.5, 0.2, 0.3, 0)
        x1, y1, x2, y2 = box.to_xyxy(640, 480)

        assert x1 == 256  # (0.5 - 0.1) * 640
        assert y1 == 168  # (0.5 - 0.15) * 480
        assert x2 == 384  # (0.5 + 0.1) * 640
        assert y2 == 312  # (0.5 + 0.15) * 480

    def test_roundtrip(self):
        """Test YOLO format roundtrip."""
        original = "5 0.123456 0.654321 0.111111 0.222222"
        box = BoundingBox.from_yolo(original)
        result = box.to_yolo()

        # Parse both dan compare
        box2 = BoundingBox.from_yolo(result)

        assert box.class_id == box2.class_id
        assert abs(box.x_center - box2.x_center) < 1e-5
        assert abs(box.y_center - box2.y_center) < 1e-5
```

### 9.2 Running Tests

```bash
# Run all utils tests
pytest tests/test_*.py -v

# Run specific test file
pytest tests/test_annotation_parsers.py -v

# Run with coverage
pytest tests/ --cov=utils --cov-report=html
```

---

## 10. Latihan

### Latihan 1: New Parser

Implementasi parser untuk format baru:

```python
def from_detectron2(cls, annotation: Dict) -> 'BoundingBox':
    """Parse Detectron2 format annotation."""
    # Detectron2 format:
    # {"bbox": [x1, y1, x2, y2], "bbox_mode": BoxMode.XYXY_ABS, "category_id": int}
    pass
```

### Latihan 2: Image Augmentation

Tambahkan augmentation ke image_utils.py:

```python
def apply_augmentation(
    image: Image.Image,
    flip_prob: float = 0.5,
    rotate_range: Tuple[int, int] = (-15, 15),
    brightness_range: Tuple[float, float] = (0.8, 1.2)
) -> Image.Image:
    """Apply random augmentations to image."""
    pass
```

### Latihan 3: Batch Statistics

Buat utility untuk batch analysis:

```python
def compare_datasets(dataset_paths: List[Path]) -> Dict:
    """
    Compare statistics across multiple datasets.

    Returns class distribution differences, size statistics, etc.
    """
    pass
```

---

**Selamat! Anda telah menyelesaikan Modul 08: Utility Functions**

_Lanjut ke: [Modul 09 - Tips dan Troubleshooting](./09-tips-troubleshooting.md)_
