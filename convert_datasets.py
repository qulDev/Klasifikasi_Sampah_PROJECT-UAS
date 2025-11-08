#!/usr/bin/env python3
"""
Dataset Conversion Script

Auto-detect and convert multiple dataset formats to YOLO format.

Simple usage:
    python convert_datasets.py
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

from utils.annotation_parsers import (
    parse_coco_json,
    parse_voc_xml,
    parse_yolo_txt,
    parse_csv_annotations,
    create_class_folder_annotation,
)
from utils.dataset_stats import detect_dataset_format
from utils.image_utils import verify_image
from utils.label_mapper import TARGET_CLASSES
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Default configuration
DEFAULT_SRC = './datasets/raw'
DEFAULT_DST = './datasets/processed/all'
DEFAULT_CLASSES = TARGET_CLASSES


def save_mapping(mapping_file: Path, stats: Dict):
    """Save conversion statistics to JSON."""
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    mapping_file.write_text(json.dumps({
        "target_classes": TARGET_CLASSES,
        "statistics": stats
    }, indent=2))
    logger.info(f"Saved mapping: {mapping_file}")


def copy_image_and_label(img_path: Path, annotations, out_dir: Path) -> int:
    """Copy image and create YOLO label. Returns number of annotations."""
    # Verify image
    is_valid, error = verify_image(img_path)
    if not is_valid:
        logger.warning(f"Invalid: {img_path.name} - {error}")
        return 0
    
    # Copy image
    dest_img = out_dir / 'images' / img_path.name
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, dest_img)
    
    # Create label
    if annotations:
        dest_lbl = out_dir / 'labels' / f'{img_path.stem}.txt'
        dest_lbl.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_lbl, 'w') as f:
            for class_id, bbox in annotations:
                f.write(bbox.to_yolo_line(class_id) + '\n')
        
        return len(annotations)
    return 0


def find_image(filename: str, search_dir: Path) -> Path:
    """Find image file in directory or subdirectories."""
    # Try direct path
    img_path = search_dir / filename
    if img_path.exists():
        return img_path
    
    # Try images subdirectory
    img_path = search_dir / 'images' / filename
    if img_path.exists():
        return img_path
    
    # Search subdirectories
    found = list(search_dir.rglob(filename))
    return found[0] if found else None


def convert_coco(dataset_path: Path, out_dir: Path, class_map: Dict, dry_run: bool) -> Tuple[int, int]:
    """Convert COCO format to YOLO."""
    logger.info(f"Converting COCO: {dataset_path.name}")
    
    # Find JSON
    json_files = list(dataset_path.rglob('annotations.json')) + \
                 list(dataset_path.rglob('instances_*.json'))
    
    if not json_files:
        logger.warning("No COCO JSON found")
        return 0, 0
    
    # Parse annotations
    image_annotations = parse_coco_json(json_files[0], class_map)
    
    if dry_run:
        num_ann = sum(len(a) for a in image_annotations.values())
        logger.info(f"[DRY RUN] {len(image_annotations)} images, {num_ann} annotations")
        return len(image_annotations), num_ann
    
    # Convert
    num_img, num_ann = 0, 0
    for filename, annotations in tqdm(image_annotations.items(), desc="COCO"):
        img_path = find_image(filename, dataset_path)
        if img_path:
            num_ann += copy_image_and_label(img_path, annotations, out_dir)
            num_img += 1
    
    logger.info(f"Converted {num_img} images, {num_ann} annotations")
    return num_img, num_ann


def convert_voc(dataset_path: Path, out_dir: Path, class_map: Dict, dry_run: bool) -> Tuple[int, int]:
    """Convert Pascal VOC format to YOLO."""
    logger.info(f"Converting VOC: {dataset_path.name}")
    
    xml_files = list((dataset_path / 'Annotations').glob('*.xml'))
    
    if dry_run:
        logger.info(f"[DRY RUN] {len(xml_files)} XML files")
        return len(xml_files), 0
    
    # Find images directory
    img_dir = dataset_path / 'JPEGImages'
    if not img_dir.exists():
        img_dir = dataset_path / 'images'
    
    num_img, num_ann = 0, 0
    for xml_path in tqdm(xml_files, desc="VOC"):
        try:
            filename, _, _, annotations = parse_voc_xml(xml_path, class_map)
            img_path = img_dir / filename
            
            if img_path.exists():
                num_ann += copy_image_and_label(img_path, annotations, out_dir)
                num_img += 1
        except Exception as e:
            logger.error(f"Failed {xml_path.name}: {e}")
    
    logger.info(f"Converted {num_img} images, {num_ann} annotations")
    return num_img, num_ann


def convert_yolo(dataset_path: Path, out_dir: Path, class_map: Dict, dry_run: bool) -> Tuple[int, int]:
    """Copy existing YOLO format."""
    logger.info(f"Converting YOLO: {dataset_path.name}")
    
    img_files = list((dataset_path / 'images').glob('*.jpg')) + \
                list((dataset_path / 'images').glob('*.png'))
    
    if dry_run:
        logger.info(f"[DRY RUN] {len(img_files)} images")
        return len(img_files), 0
    
    lbl_dir = dataset_path / 'labels'
    num_img, num_ann = 0, 0
    
    for img_path in tqdm(img_files, desc="YOLO"):
        lbl_path = lbl_dir / f'{img_path.stem}.txt'
        annotations = parse_yolo_txt(lbl_path) if lbl_path.exists() else []
        
        num_ann += copy_image_and_label(img_path, annotations, out_dir)
        num_img += 1
    
    logger.info(f"Converted {num_img} images, {num_ann} annotations")
    return num_img, num_ann


def convert_class_folders(dataset_path: Path, out_dir: Path, class_map: Dict, dry_run: bool) -> Tuple[int, int]:
    """Convert class folder structure to YOLO."""
    logger.info(f"Converting class folders: {dataset_path.name}")
    
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if dry_run:
        total = sum(len(list(d.glob('*.jpg')) + list(d.glob('*.png'))) for d in class_dirs)
        logger.info(f"[DRY RUN] {total} images from {len(class_dirs)} classes")
        return total, total
    
    num_img, num_ann = 0, 0
    for class_dir in tqdm(class_dirs, desc="Class folders"):
        annotation = create_class_folder_annotation(class_dir.name, class_map)
        if not annotation:
            logger.warning(f"Unknown class: {class_dir.name}")
            continue
        
        img_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        for img_path in img_files:
            num_ann += copy_image_and_label(img_path, [annotation], out_dir)
            num_img += 1
    
    logger.info(f"Converted {num_img} images, {num_ann} annotations")
    return num_img, num_ann


def convert_csv(dataset_path: Path, out_dir: Path, class_map: Dict, dry_run: bool) -> Tuple[int, int]:
    """Convert CSV annotations to YOLO."""
    logger.info(f"Converting CSV: {dataset_path.name}")
    
    csv_files = list(dataset_path.glob('*.csv'))
    if not csv_files:
        logger.warning("No CSV files found")
        return 0, 0
    
    image_annotations = parse_csv_annotations(csv_files[0], class_map)
    
    if dry_run:
        num_ann = sum(len(a) for a in image_annotations.values())
        logger.info(f"[DRY RUN] {len(image_annotations)} images, {num_ann} annotations")
        return len(image_annotations), num_ann
    
    num_img, num_ann = 0, 0
    for filename, annotations in tqdm(image_annotations.items(), desc="CSV"):
        img_path = find_image(filename, dataset_path)
        if img_path:
            num_ann += copy_image_and_label(img_path, annotations, out_dir)
            num_img += 1
    
    logger.info(f"Converted {num_img} images, {num_ann} annotations")
    return num_img, num_ann


def main():
    parser = argparse.ArgumentParser(
        description="Convert datasets to YOLO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Simple usage:
  python convert_datasets.py

Custom paths:
  python convert_datasets.py --src ./datasets/raw --dst ./datasets/processed/all
  python convert_datasets.py --dry-run  # Preview only
        """
    )

    parser.add_argument('--src', type=Path, default=Path(DEFAULT_SRC),
                        help=f'Source directory (default: {DEFAULT_SRC})')
    parser.add_argument('--dst', type=Path, default=Path(DEFAULT_DST),
                        help=f'Destination directory (default: {DEFAULT_DST})')
    parser.add_argument('--classes', nargs='*', default=DEFAULT_CLASSES,
                        help=f'Target classes (default: {len(DEFAULT_CLASSES)} classes)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview only, no conversion')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Dataset Conversion")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN - No files will be modified")

    # Build class mapping
    class_map = {name: idx for idx, name in enumerate(args.classes)}

    # Converter dispatch
    converters = {
        'coco': convert_coco,
        'voc': convert_voc,
        'yolo': convert_yolo,
        'class_folders': convert_class_folders,
        'csv': convert_csv
    }

    # Scan datasets
    dataset_dirs = [d for d in args.src.iterdir() if d.is_dir()]
    logger.info(f"Found {len(dataset_dirs)} datasets in {args.src}")

    total_img, total_ann = 0, 0

    for dataset_dir in dataset_dirs:
        logger.info("")
        logger.info(f"Processing: {dataset_dir.name}")
        logger.info("-" * 60)

        # Detect format
        fmt = detect_dataset_format(dataset_dir)
        logger.info(f"Format: {fmt}")

        if fmt == 'unknown':
            logger.warning("Unknown format, skipping")
            continue

        # Convert
        try:
            converter = converters.get(fmt)
            if converter:
                num_img, num_ann = converter(dataset_dir, args.dst, class_map, args.dry_run)
                total_img += num_img
                total_ann += num_ann
            else:
                logger.warning(f"Converter for '{fmt}' not implemented")

        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=args.verbose)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Images: {total_img}")
    logger.info(f"Annotations: {total_ann}")
    logger.info(f"Output: {args.dst}")

    if not args.dry_run:
        # Save stats
        stats = {
            "total_classes": len(args.classes),
            "total_images": total_img,
            "total_annotations": total_ann
        }
        save_mapping(args.dst.parent / 'conversion_stats.json', stats)

    logger.info("=" * 60)


if __name__ == '__main__':
    main()
