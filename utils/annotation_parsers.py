"""
Multi-format annotation parsing and conversion to YOLO format.

Supports:
- COCO JSON (Microsoft COCO format)
- Pascal VOC XML (PASCAL Visual Object Classes)
- YOLO TXT (existing YOLO annotations)
- Class Folders (folder-per-class for classification)
- CSV (custom bbox annotations)
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .logger import setup_logger

logger = setup_logger(__name__)


class BoundingBox:
    """Normalized bounding box in YOLO format."""
    
    def __init__(self, x_center: float, y_center: float, width: float, height: float):
        """
        Initialize normalized bounding box.

        Args:
            x_center: Normalized x-coordinate of box center [0, 1]
            y_center: Normalized y-coordinate of box center [0, 1]
            width: Normalized box width [0, 1]
            height: Normalized box height [0, 1]
        """
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    @classmethod
    def from_coco(cls, x: float, y: float, w: float, h: float, img_width: int, img_height: int) -> 'BoundingBox':
        """
        Create bounding box from COCO format (absolute xywh).

        Args:
            x: Top-left x coordinate (absolute)
            y: Top-left y coordinate (absolute)
            w: Box width (absolute)
            h: Box height (absolute)
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            BoundingBox instance with normalized coordinates
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
        Create bounding box from Pascal VOC format (absolute xyxy).

        Args:
            xmin: Minimum x coordinate (absolute)
            ymin: Minimum y coordinate (absolute)
            xmax: Maximum x coordinate (absolute)
            ymax: Maximum y coordinate (absolute)
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            BoundingBox instance with normalized coordinates
        """
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        return cls(x_center, y_center, width, height)

    def is_valid(self) -> bool:
        """
        Validate bounding box coordinates.

        Returns:
            True if all coordinates are in [0, 1] and box is within image bounds
        """
        if not (0 <= self.x_center <= 1 and 0 <= self.y_center <= 1):
            return False
        if not (0 < self.width <= 1 and 0 < self.height <= 1):
            return False
        
        # Check if box edges are within image bounds
        left = self.x_center - self.width / 2
        right = self.x_center + self.width / 2
        top = self.y_center - self.height / 2
        bottom = self.y_center + self.height / 2
        
        return (0 <= left <= 1 and 0 <= right <= 1 and 
                0 <= top <= 1 and 0 <= bottom <= 1)

    def to_yolo_line(self, class_id: int) -> str:
        """
        Convert to YOLO format annotation line.

        Args:
            class_id: Zero-indexed class ID

        Returns:
            YOLO format string: "<class_id> <x_center> <y_center> <width> <height>"
        """
        return f"{class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


def parse_coco_json(json_path: Path, class_mapping: Dict[str, int]) -> Dict[str, List[BoundingBox]]:
    """
    Parse COCO JSON annotations and convert to YOLO format.

    Args:
        json_path: Path to COCO JSON file (annotations.json or instances_*.json)
        class_mapping: Dictionary mapping class_name -> class_id

    Returns:
        Dictionary mapping image_filename -> list of (class_id, BoundingBox) tuples

    Example:
        >>> class_map = {"plastic": 0, "metal": 1}
        >>> annotations = parse_coco_json(Path("annotations.json"), class_map)
    """
    logger.info(f"Parsing COCO JSON: {json_path}")

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Build lookup tables
    images_dict = {img['id']: img for img in coco_data.get('images', [])}
    categories_dict = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}

    # Parse annotations
    image_annotations = {}

    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox']  # [x, y, width, height]

        if image_id not in images_dict:
            logger.warning(f"Image ID {image_id} not found in images list")
            continue

        image_info = images_dict[image_id]
        image_filename = image_info['file_name']
        img_width = image_info['width']
        img_height = image_info['height']

        # Get class name and map to target class
        class_name = categories_dict.get(category_id, 'unknown')
        
        if class_name not in class_mapping:
            logger.warning(f"Class '{class_name}' not in mapping, skipping annotation")
            continue

        class_id = class_mapping[class_name]

        # Convert COCO bbox to YOLO format
        x, y, w, h = bbox
        bbox_yolo = BoundingBox.from_coco(x, y, w, h, img_width, img_height)

        if not bbox_yolo.is_valid():
            logger.warning(f"Invalid bbox in {image_filename}: {bbox}")
            continue

        if image_filename not in image_annotations:
            image_annotations[image_filename] = []

        image_annotations[image_filename].append((class_id, bbox_yolo))

    logger.info(f"Parsed {len(image_annotations)} images with {sum(len(v) for v in image_annotations.values())} annotations")
    return image_annotations


def parse_voc_xml(xml_path: Path, class_mapping: Dict[str, int]) -> Tuple[str, int, int, List[Tuple[int, BoundingBox]]]:
    """
    Parse single Pascal VOC XML annotation file.

    Args:
        xml_path: Path to VOC XML file
        class_mapping: Dictionary mapping class_name -> class_id

    Returns:
        Tuple of (image_filename, image_width, image_height, annotations)
        - annotations: List of (class_id, BoundingBox) tuples

    Example:
        >>> class_map = {"plastic": 0, "metal": 1}
        >>> filename, w, h, annots = parse_voc_xml(Path("img001.xml"), class_map)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract image info
    filename = root.find('filename').text
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    # Parse objects
    annotations = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text

        if class_name not in class_mapping:
            logger.warning(f"Class '{class_name}' not in mapping, skipping annotation in {xml_path}")
            continue

        class_id = class_mapping[class_name]

        # Parse bounding box
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        bbox_yolo = BoundingBox.from_voc(xmin, ymin, xmax, ymax, img_width, img_height)

        if not bbox_yolo.is_valid():
            logger.warning(f"Invalid bbox in {xml_path}: {xmin},{ymin},{xmax},{ymax}")
            continue

        annotations.append((class_id, bbox_yolo))

    return filename, img_width, img_height, annotations


def parse_yolo_txt(txt_path: Path) -> List[Tuple[int, BoundingBox]]:
    """
    Parse existing YOLO format annotation file.

    Args:
        txt_path: Path to YOLO .txt file

    Returns:
        List of (class_id, BoundingBox) tuples

    Example:
        >>> annots = parse_yolo_txt(Path("img001.txt"))
        >>> print(f"Found {len(annots)} annotations")
    """
    annotations = []

    with open(txt_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                logger.warning(f"Invalid YOLO annotation in {txt_path} line {line_num}: {line}")
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                bbox = BoundingBox(x_center, y_center, width, height)

                if not bbox.is_valid():
                    logger.warning(f"Invalid bbox in {txt_path} line {line_num}: {line}")
                    continue

                annotations.append((class_id, bbox))

            except ValueError as e:
                logger.warning(f"Failed to parse {txt_path} line {line_num}: {e}")
                continue

    return annotations


def parse_csv_annotations(csv_path: Path, class_mapping: Dict[str, int]) -> Dict[str, List[Tuple[int, BoundingBox]]]:
    """
    Parse CSV bbox annotations and convert to YOLO format.

    Expected CSV format:
        filename, xmin, ymin, xmax, ymax, class, width, height

    Args:
        csv_path: Path to CSV file
        class_mapping: Dictionary mapping class_name -> class_id

    Returns:
        Dictionary mapping image_filename -> list of (class_id, BoundingBox) tuples

    Example:
        >>> class_map = {"plastic": 0, "metal": 1}
        >>> annotations = parse_csv_annotations(Path("annotations.csv"), class_map)
    """
    logger.info(f"Parsing CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize column names (case-insensitive)
    df.columns = [col.lower() for col in df.columns]

    required_columns = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    image_annotations = {}

    for _, row in df.iterrows():
        filename = row['filename']
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        class_name = row['class']

        # Get image dimensions
        if 'width' in df.columns and 'height' in df.columns:
            img_width = row['width']
            img_height = row['height']
        else:
            logger.warning(f"Image dimensions not in CSV, assuming 640x640 for {filename}")
            img_width = 640
            img_height = 640

        if class_name not in class_mapping:
            logger.warning(f"Class '{class_name}' not in mapping, skipping annotation")
            continue

        class_id = class_mapping[class_name]

        bbox_yolo = BoundingBox.from_voc(xmin, ymin, xmax, ymax, img_width, img_height)

        if not bbox_yolo.is_valid():
            logger.warning(f"Invalid bbox in {filename}: {xmin},{ymin},{xmax},{ymax}")
            continue

        if filename not in image_annotations:
            image_annotations[filename] = []

        image_annotations[filename].append((class_id, bbox_yolo))

    logger.info(f"Parsed {len(image_annotations)} images with {sum(len(v) for v in image_annotations.values())} annotations")
    return image_annotations


def create_class_folder_annotation(class_name: str, class_mapping: Dict[str, int]) -> Optional[Tuple[int, BoundingBox]]:
    """
    Create full-image annotation for class folder dataset.

    For classification datasets, create a bounding box covering the entire image.

    Args:
        class_name: Name of the class (folder name)
        class_mapping: Dictionary mapping class_name -> class_id

    Returns:
        Tuple of (class_id, BoundingBox) representing full image, or None if class not in mapping

    Example:
        >>> class_map = {"plastic": 0}
        >>> annot = create_class_folder_annotation("plastic", class_map)
    """
    if class_name not in class_mapping:
        return None

    class_id = class_mapping[class_name]

    # Create bounding box covering entire image (center at 0.5, 0.5, full size)
    bbox = BoundingBox(x_center=0.5, y_center=0.5, width=1.0, height=1.0)

    return (class_id, bbox)
