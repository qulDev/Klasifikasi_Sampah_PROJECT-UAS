#!/usr/bin/env python3
"""
Dataset Conversion Script for Klasifikasi Sampah

Automatically detect and convert multiple dataset formats to YOLO format:
- COCO JSON (Microsoft COCO format)
- Pascal VOC XML  
- Existing YOLO TXT
- Class-based folder structure
- CSV annotations

Usage:
    python convert_datasets.py --src ./datasets/raw --dst ./datasets/processed --dry-run
    python convert_datasets.py --src ./datasets/raw --dst ./datasets/processed
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from utils.annotation_parsers import (
    parse_coco_json,
    parse_voc_xml,
    parse_yolo_txt,
    parse_csv_annotations,
    create_class_folder_annotation,
)
from utils.dataset_stats import detect_dataset_format, count_images
from utils.image_utils import verify_image, hash_image, get_image_info
from utils.label_mapper import map_label, TARGET_CLASSES
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_manual_mappings(mapping_file: Path) -> Dict[str, str]:
    """
    Load manual label mappings from JSON file.

    Args:
        mapping_file: Path to mapping JSON file

    Returns:
        Dictionary mapping source_label -> target_label
    """
    if not mapping_file.exists():
        return {}

    with open(mapping_file, 'r') as f:
        data = json.load(f)
        return data.get('manual_mappings', {})


def save_label_mapping(mapping_file: Path, mappings: Dict, statistics: Dict):
    """
    Save label mapping results to JSON file.

    Args:
        mapping_file: Path to output JSON file
        mappings: Dictionary of source -> (target, method, confidence)
        statistics: Statistics about mapping process
    """
    mapping_file.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "mappings": {
            source: {
                "target_class": target,
                "mapping_method": method,
                "confidence": conf,
                "manual_override": method == "manual"
            }
            for source, (target, method, conf) in mappings.items()
        },
        "target_classes": TARGET_CLASSES,
        "statistics": statistics
    }

    with open(mapping_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Label mapping saved to {mapping_file}")


def convert_dataset_coco(
    dataset_path: Path,
    output_dir: Path,
    class_mapping: Dict[str, int],
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Convert COCO format dataset to YOLO.

    Args:
        dataset_path: Path to dataset root directory
        output_dir: Path to output directory
        class_mapping: Dictionary mapping class_name -> class_id
        dry_run: If True, only report what would be done

    Returns:
        Tuple of (num_images_converted, num_annotations_converted)
    """
    logger.info(f"Converting COCO dataset: {dataset_path}")

    # Find COCO JSON file
    json_files = list(dataset_path.rglob('annotations.json')) + \
                 list(dataset_path.rglob('instances_*.json'))

    if not json_files:
        logger.warning(f"No COCO JSON found in {dataset_path}")
        return 0, 0

    json_path = json_files[0]
    logger.info(f"Using COCO JSON: {json_path}")

    # Parse annotations
    image_annotations = parse_coco_json(json_path, class_mapping)

    if dry_run:
        num_annots = sum(len(annots) for annots in image_annotations.values())
        logger.info(f"[DRY RUN] Would convert {len(image_annotations)} images, {num_annots} annotations")
        return len(image_annotations), num_annots

    # Find images directory
    images_dir = dataset_path / 'images'
    if not images_dir.exists():
        images_dir = dataset_path

    # Copy images and create YOLO labels
    num_images = 0
    num_annotations = 0

    for image_filename, annotations in tqdm(image_annotations.items(), desc="Converting COCO"):
        # Find image file
        image_path = images_dir / image_filename
        if not image_path.exists():
            # Try searching subdirectories
            found_images = list(dataset_path.rglob(image_filename))
            if found_images:
                image_path = found_images[0]
            else:
                logger.warning(f"Image not found: {image_filename}")
                continue

        # Verify image
        is_valid, error = verify_image(image_path)
        if not is_valid:
            logger.warning(f"Invalid image {image_filename}: {error}")
            continue

        # Copy image
        dest_image = output_dir / 'images' / image_filename
        dest_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dest_image)

        # Create YOLO label file
        label_filename = image_path.stem + '.txt'
        dest_label = output_dir / 'labels' / label_filename
        dest_label.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_label, 'w') as f:
            for class_id, bbox in annotations:
                f.write(bbox.to_yolo_line(class_id) + '\n')
                num_annotations += 1

        num_images += 1

    logger.info(f"Converted {num_images} images, {num_annotations} annotations")
    return num_images, num_annotations


def convert_dataset_voc(
    dataset_path: Path,
    output_dir: Path,
    class_mapping: Dict[str, int],
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Convert Pascal VOC format dataset to YOLO.

    Args:
        dataset_path: Path to dataset root directory
        output_dir: Path to output directory
        class_mapping: Dictionary mapping class_name -> class_id
        dry_run: If True, only report what would be done

    Returns:
        Tuple of (num_images_converted, num_annotations_converted)
    """
    logger.info(f"Converting VOC dataset: {dataset_path}")

    annotations_dir = dataset_path / 'Annotations'
    xml_files = list(annotations_dir.glob('*.xml'))

    if dry_run:
        logger.info(f"[DRY RUN] Would convert {len(xml_files)} images from VOC format")
        return len(xml_files), 0

    # Find images directory
    images_dir = dataset_path / 'JPEGImages'
    if not images_dir.exists():
        images_dir = dataset_path / 'images'

    num_images = 0
    num_annotations = 0

    for xml_path in tqdm(xml_files, desc="Converting VOC"):
        try:
            filename, img_width, img_height, annotations = parse_voc_xml(xml_path, class_mapping)

            # Find image file
            image_path = images_dir / filename
            if not image_path.exists():
                logger.warning(f"Image not found: {filename}")
                continue

            # Verify image
            is_valid, error = verify_image(image_path)
            if not is_valid:
                logger.warning(f"Invalid image {filename}: {error}")
                continue

            # Copy image
            dest_image = output_dir / 'images' / filename
            dest_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, dest_image)

            # Create YOLO label file
            label_filename = image_path.stem + '.txt'
            dest_label = output_dir / 'labels' / label_filename
            dest_label.parent.mkdir(parents=True, exist_ok=True)

            with open(dest_label, 'w') as f:
                for class_id, bbox in annotations:
                    f.write(bbox.to_yolo_line(class_id) + '\n')
                    num_annotations += 1

            num_images += 1

        except Exception as e:
            logger.error(f"Failed to process {xml_path}: {e}")
            continue

    logger.info(f"Converted {num_images} images, {num_annotations} annotations")
    return num_images, num_annotations


def convert_dataset_yolo(
    dataset_path: Path,
    output_dir: Path,
    class_mapping: Dict[str, int],
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Copy existing YOLO format dataset (validate and reorganize).

    Args:
        dataset_path: Path to dataset root directory
        output_dir: Path to output directory
        class_mapping: Dictionary mapping class_name -> class_id
        dry_run: If True, only report what would be done

    Returns:
        Tuple of (num_images_converted, num_annotations_converted)
    """
    logger.info(f"Converting YOLO dataset: {dataset_path}")

    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'

    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

    if dry_run:
        logger.info(f"[DRY RUN] Would copy {len(image_files)} images from YOLO format")
        return len(image_files), 0

    num_images = 0
    num_annotations = 0

    for image_path in tqdm(image_files, desc="Converting YOLO"):
        # Verify image
        is_valid, error = verify_image(image_path)
        if not is_valid:
            logger.warning(f"Invalid image {image_path.name}: {error}")
            continue

        # Copy image
        dest_image = output_dir / 'images' / image_path.name
        dest_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dest_image)

        # Copy label if exists
        label_path = labels_dir / (image_path.stem + '.txt')
        if label_path.exists():
            annotations = parse_yolo_txt(label_path)

            dest_label = output_dir / 'labels' / label_path.name
            dest_label.parent.mkdir(parents=True, exist_ok=True)

            with open(dest_label, 'w') as f:
                for class_id, bbox in annotations:
                    f.write(bbox.to_yolo_line(class_id) + '\n')
                    num_annotations += 1

        num_images += 1

    logger.info(f"Converted {num_images} images, {num_annotations} annotations")
    return num_images, num_annotations


def convert_dataset_class_folders(
    dataset_path: Path,
    output_dir: Path,
    class_mapping: Dict[str, int],
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Convert class folder structure to YOLO format.

    Creates full-image annotations for classification datasets.

    Args:
        dataset_path: Path to dataset root directory
        output_dir: Path to output directory
        class_mapping: Dictionary mapping class_name -> class_id
        dry_run: If True, only report what would be done

    Returns:
        Tuple of (num_images_converted, num_annotations_converted)
    """
    logger.info(f"Converting class folders dataset: {dataset_path}")

    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    if dry_run:
        total_images = sum(len(list(d.glob('*.jpg')) + list(d.glob('*.png'))) for d in class_dirs)
        logger.info(f"[DRY RUN] Would convert {total_images} images from {len(class_dirs)} class folders")
        return total_images, total_images

    num_images = 0
    num_annotations = 0

    for class_dir in tqdm(class_dirs, desc="Converting class folders"):
        class_name = class_dir.name

        # Create full-image annotation
        annotation = create_class_folder_annotation(class_name, class_mapping)
        if annotation is None:
            logger.warning(f"Class '{class_name}' not in mapping, skipping folder")
            continue

        class_id, bbox = annotation

        # Process all images in class folder
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))

        for image_path in image_files:
            # Verify image
            is_valid, error = verify_image(image_path)
            if not is_valid:
                logger.warning(f"Invalid image {image_path.name}: {error}")
                continue

            # Copy image
            dest_image = output_dir / 'images' / image_path.name
            dest_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, dest_image)

            # Create YOLO label file
            label_filename = image_path.stem + '.txt'
            dest_label = output_dir / 'labels' / label_filename
            dest_label.parent.mkdir(parents=True, exist_ok=True)

            with open(dest_label, 'w') as f:
                f.write(bbox.to_yolo_line(class_id) + '\n')
                num_annotations += 1

            num_images += 1

    logger.info(f"Converted {num_images} images, {num_annotations} annotations")
    return num_images, num_annotations


def convert_dataset_csv(
    dataset_path: Path,
    output_dir: Path,
    class_mapping: Dict[str, int],
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Convert CSV annotations to YOLO format.

    Args:
        dataset_path: Path to dataset root directory
        output_dir: Path to output directory
        class_mapping: Dictionary mapping class_name -> class_id
        dry_run: If True, only report what would be done

    Returns:
        Tuple of (num_images_converted, num_annotations_converted)
    """
    logger.info(f"Converting CSV dataset: {dataset_path}")

    csv_files = list(dataset_path.glob('*.csv'))
    if not csv_files:
        logger.warning(f"No CSV files found in {dataset_path}")
        return 0, 0

    csv_path = csv_files[0]
    image_annotations = parse_csv_annotations(csv_path, class_mapping)

    if dry_run:
        num_annots = sum(len(annots) for annots in image_annotations.values())
        logger.info(f"[DRY RUN] Would convert {len(image_annotations)} images, {num_annots} annotations")
        return len(image_annotations), num_annots

    # Find images directory
    images_dir = dataset_path / 'images'
    if not images_dir.exists():
        images_dir = dataset_path

    num_images = 0
    num_annotations = 0

    for image_filename, annotations in tqdm(image_annotations.items(), desc="Converting CSV"):
        # Find image file
        image_path = images_dir / image_filename
        if not image_path.exists():
            logger.warning(f"Image not found: {image_filename}")
            continue

        # Verify image
        is_valid, error = verify_image(image_path)
        if not is_valid:
            logger.warning(f"Invalid image {image_filename}: {error}")
            continue

        # Copy image
        dest_image = output_dir / 'images' / image_filename
        dest_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dest_image)

        # Create YOLO label file
        label_filename = Path(image_filename).stem + '.txt'
        dest_label = output_dir / 'labels' / label_filename
        dest_label.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_label, 'w') as f:
            for class_id, bbox in annotations:
                f.write(bbox.to_yolo_line(class_id) + '\n')
                num_annotations += 1

        num_images += 1

    logger.info(f"Converted {num_images} images, {num_annotations} annotations")
    return num_images, num_annotations


def main():
    parser = argparse.ArgumentParser(
        description="Convert multiple dataset formats to YOLO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be converted
  python convert_datasets.py --src ./datasets/raw --dst ./datasets/processed/all --dry-run

  # Convert all datasets
  python convert_datasets.py --src ./datasets/raw --dst ./datasets/processed/all

  # Convert with custom class list
  python convert_datasets.py --src ./datasets/raw --dst ./datasets/processed/all --classes plastic metal glass

  # Use manual mapping file
  python convert_datasets.py --src ./datasets/raw --dst ./datasets/processed/all --mapping-file custom_map.json
        """
    )

    parser.add_argument('--src', type=Path, required=True,
                        help='Source directory containing raw datasets')
    parser.add_argument('--dst', type=Path, required=True,
                        help='Destination directory for converted YOLO dataset')
    parser.add_argument('--classes', nargs='*', default=TARGET_CLASSES,
                        help=f'Target class names (default: {TARGET_CLASSES})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only report what would be done, without converting')
    parser.add_argument('--mapping-file', type=Path, default=Path('./datasets/label_map.json'),
                        help='Path to save label mapping JSON (default: ./datasets/label_map.json)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging (DEBUG level)')

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Klasifikasi Sampah - Dataset Conversion")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be modified")

    # Load manual mappings
    manual_mappings = load_manual_mappings(args.mapping_file) if args.mapping_file.exists() else {}

    # Build class mapping (class_name -> class_id)
    target_classes = args.classes
    class_name_to_id = {name: idx for idx, name in enumerate(target_classes)}

    # Track all label mappings
    all_mappings = {}
    mapping_stats = defaultdict(int)

    # Scan source directory for datasets
    dataset_dirs = [d for d in args.src.iterdir() if d.is_dir()]
    logger.info(f"Found {len(dataset_dirs)} potential datasets in {args.src}")

    total_images = 0
    total_annotations = 0

    for dataset_dir in dataset_dirs:
        logger.info("")
        logger.info(f"Processing: {dataset_dir.name}")
        logger.info("-" * 60)

        # Detect format
        format_type = detect_dataset_format(dataset_dir)
        logger.info(f"Detected format: {format_type}")

        if format_type == 'unknown':
            logger.warning(f"Unknown format, skipping {dataset_dir.name}")
            continue

        # Convert based on format
        try:
            if format_type == 'coco':
                num_img, num_ann = convert_dataset_coco(dataset_dir, args.dst, class_name_to_id, args.dry_run)
            elif format_type == 'voc':
                num_img, num_ann = convert_dataset_voc(dataset_dir, args.dst, class_name_to_id, args.dry_run)
            elif format_type == 'yolo':
                num_img, num_ann = convert_dataset_yolo(dataset_dir, args.dst, class_name_to_id, args.dry_run)
            elif format_type == 'class_folders':
                num_img, num_ann = convert_dataset_class_folders(dataset_dir, args.dst, class_name_to_id, args.dry_run)
            elif format_type == 'csv':
                num_img, num_ann = convert_dataset_csv(dataset_dir, args.dst, class_name_to_id, args.dry_run)
            else:
                logger.warning(f"Format '{format_type}' not yet implemented")
                continue

            total_images += num_img
            total_annotations += num_ann

        except Exception as e:
            logger.error(f"Failed to convert {dataset_dir.name}: {e}", exc_info=True)
            continue

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total images: {total_images}")
    logger.info(f"Total annotations: {total_annotations}")
    logger.info(f"Output directory: {args.dst}")

    if not args.dry_run:
        # Save label mapping
        mapping_statistics = {
            "total_target_classes": len(target_classes),
            "total_images_converted": total_images,
            "total_annotations_converted": total_annotations,
        }

        save_label_mapping(args.mapping_file, all_mappings, mapping_statistics)
        logger.info(f"Label mapping saved to: {args.mapping_file}")

    logger.info("=" * 60)


if __name__ == '__main__':
    main()
