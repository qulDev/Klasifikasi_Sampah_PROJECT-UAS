"""
Dataset statistics and reporting utilities.

Provides functions to calculate class distributions, count images,
and detect dataset formats.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from .image_utils import verify_image


def count_images(directory: Path) -> int:
    """
    Count valid image files in a directory.

    Args:
        directory: Path to directory containing images

    Returns:
        Number of valid image files

    Example:
        >>> count = count_images(Path("./datasets/raw/TACO/images"))
        >>> print(f"Found {count} images")
    """
    if not directory.exists():
        return 0

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    count = 0

    for file_path in directory.rglob('*'):
        if file_path.suffix.lower() in valid_extensions:
            is_valid, _ = verify_image(file_path)
            if is_valid:
                count += 1

    return count


def get_class_distribution(annotations: List[Tuple[str, str]]) -> Dict[str, int]:
    """
    Calculate class distribution from annotations.

    Args:
        annotations: List of (image_id, class_name) tuples

    Returns:
        Dictionary mapping class_name -> count

    Example:
        >>> annots = [("img1", "plastic"), ("img2", "metal"), ("img3", "plastic")]
        >>> dist = get_class_distribution(annots)
        >>> print(dist)
        {'plastic': 2, 'metal': 1}
    """
    class_names = [class_name for _, class_name in annotations]
    return dict(Counter(class_names))


def detect_coco_format(dataset_path: Path) -> bool:
    """
    Detect if dataset is in COCO JSON format.

    Args:
        dataset_path: Path to dataset root directory

    Returns:
        True if COCO format detected, False otherwise

    Detection pattern:
        - File named 'annotations.json' or matching 'instances_*.json'
        - Contains 'images' and 'annotations' keys
    """
    # Check for common COCO annotation file names
    coco_files = [
        dataset_path / 'annotations.json',
        dataset_path / 'instances_train.json',
        dataset_path / 'instances_val.json',
    ]

    for coco_file in coco_files:
        if coco_file.exists():
            return True

    # Check for any annotations.json in subdirectories
    for json_file in dataset_path.rglob('annotations.json'):
        return True

    for json_file in dataset_path.rglob('instances_*.json'):
        return True

    return False


def detect_voc_format(dataset_path: Path) -> bool:
    """
    Detect if dataset is in Pascal VOC XML format.

    Args:
        dataset_path: Path to dataset root directory

    Returns:
        True if VOC format detected, False otherwise

    Detection pattern:
        - 'Annotations' folder containing .xml files
        - 'JPEGImages' or 'images' folder containing images
    """
    annotations_dir = dataset_path / 'Annotations'
    
    if not annotations_dir.exists():
        return False

    # Check for XML files in Annotations directory
    xml_files = list(annotations_dir.glob('*.xml'))
    if not xml_files:
        return False

    # Check for images directory
    images_dir = dataset_path / 'JPEGImages'
    if not images_dir.exists():
        images_dir = dataset_path / 'images'
        if not images_dir.exists():
            return False

    return True


def detect_yolo_format(dataset_path: Path) -> bool:
    """
    Detect if dataset is in YOLO format.

    Args:
        dataset_path: Path to dataset root directory

    Returns:
        True if YOLO format detected, False otherwise

    Detection pattern:
        - 'images' and 'labels' folders exist
        - .txt files in labels folder with matching image names
    """
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'

    if not (images_dir.exists() and labels_dir.exists()):
        return False

    # Check for matching .txt files
    txt_files = list(labels_dir.glob('*.txt'))
    if not txt_files:
        return False

    return True


def detect_class_folders(dataset_path: Path) -> bool:
    """
    Detect if dataset uses class-based folder structure.

    Args:
        dataset_path: Path to dataset root directory

    Returns:
        True if class folders detected, False otherwise

    Detection pattern:
        - Multiple subdirectories
        - Each subdirectory contains only images (no annotations)
    """
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if len(subdirs) < 2:
        return False

    # Check if subdirectories contain only images
    for subdir in subdirs:
        # Skip common annotation directories
        if subdir.name in ['Annotations', 'labels', 'images', 'JPEGImages']:
            return False

        # Check for image files
        image_files = list(subdir.glob('*.jpg')) + list(subdir.glob('*.png'))
        if not image_files:
            return False

        # Check for annotation files (should not exist in class folders)
        annotation_files = list(subdir.glob('*.xml')) + list(subdir.glob('*.txt')) + list(subdir.glob('*.json'))
        if annotation_files:
            return False

    return True


def detect_csv_format(dataset_path: Path) -> bool:
    """
    Detect if dataset uses CSV annotations.

    Args:
        dataset_path: Path to dataset root directory

    Returns:
        True if CSV format detected, False otherwise

    Detection pattern:
        - .csv file exists
        - Contains columns: filename, xmin, ymin, xmax, ymax, class
    """
    csv_files = list(dataset_path.glob('*.csv'))
    
    if not csv_files:
        return False

    # Check first CSV file for expected columns
    import pandas as pd
    try:
        df = pd.read_csv(csv_files[0], nrows=1)
        required_columns = {'filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class'}
        
        # Check if all required columns are present (case-insensitive)
        df_columns = set(col.lower() for col in df.columns)
        
        return required_columns.issubset(df_columns)
    except Exception:
        return False


def detect_dataset_format(dataset_path: Path) -> str:
    """
    Auto-detect dataset format using heuristics.

    Args:
        dataset_path: Path to dataset root directory

    Returns:
        Format string: 'coco', 'voc', 'yolo', 'class_folders', 'csv', or 'unknown'

    Example:
        >>> format_type = detect_dataset_format(Path("./datasets/raw/TACO"))
        >>> print(f"Detected format: {format_type}")
        Detected format: coco
    """
    if detect_coco_format(dataset_path):
        return 'coco'
    
    if detect_voc_format(dataset_path):
        return 'voc'
    
    if detect_yolo_format(dataset_path):
        return 'yolo'
    
    if detect_csv_format(dataset_path):
        return 'csv'
    
    if detect_class_folders(dataset_path):
        return 'class_folders'
    
    return 'unknown'
