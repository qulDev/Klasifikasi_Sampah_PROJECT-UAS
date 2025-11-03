#!/usr/bin/env python3
"""
Dataset Split and Preparation Script for Klasifikasi Sampah

Merge converted datasets, deduplicate images, and create stratified train/val/test splits.

Usage:
    python split_and_prep.py --src ./datasets/processed/all --out ./datasets/processed --split 0.8 0.1 0.1
    python split_and_prep.py --src ./datasets/processed/all --out ./datasets/processed --split 0.7 0.15 0.15 --seed 123
"""

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.image_utils import verify_image, hash_image
from utils.logger import setup_logger

logger = setup_logger(__name__)


def deduplicate_images(images_dir: Path, labels_dir: Path) -> Tuple[List[Path], Dict[str, str]]:
    """
    Deduplicate images based on content hash.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing corresponding labels

    Returns:
        Tuple of (unique_images, duplicates_map)
        - unique_images: List of unique image paths
        - duplicates_map: Dictionary mapping duplicate_hash -> original_path
    """
    logger.info("Deduplicating images...")

    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    seen_hashes = {}
    unique_images = []
    duplicates = {}

    for img_path in tqdm(image_files, desc="Hashing images"):
        # Verify image first
        is_valid, error = verify_image(img_path)
        if not is_valid:
            logger.warning(f"Invalid image {img_path.name}: {error}")
            continue

        # Compute hash
        img_hash = hash_image(img_path)

        if img_hash in seen_hashes:
            duplicates[img_hash] = seen_hashes[img_hash]
            logger.debug(f"Duplicate: {img_path.name} == {seen_hashes[img_hash].name}")
        else:
            seen_hashes[img_hash] = img_path
            unique_images.append(img_path)

    logger.info(f"Found {len(unique_images)} unique images, {len(duplicates)} duplicates")
    return unique_images, duplicates


def get_image_labels(image_path: Path, labels_dir: Path) -> List[int]:
    """
    Get class IDs from image label file.

    Args:
        image_path: Path to image file
        labels_dir: Directory containing label files

    Returns:
        List of class IDs in the image
    """
    label_path = labels_dir / (image_path.stem + '.txt')

    if not label_path.exists():
        return []

    class_ids = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_ids.append(int(parts[0]))

    return class_ids


def stratified_split(
    images: List[Path],
    labels_dir: Path,
    ratios: Tuple[float, float, float],
    random_seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Perform stratified split based on class distribution.

    Args:
        images: List of image paths
        labels_dir: Directory containing label files
        ratios: Tuple of (train_ratio, val_ratio, test_ratio)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_images, val_images, test_images)
    """
    logger.info(f"Performing stratified split with ratios {ratios}...")

    # Get class labels for each image (use first class if multiple)
    image_classes = []
    valid_images = []

    for img_path in tqdm(images, desc="Reading labels"):
        class_ids = get_image_labels(img_path, labels_dir)
        if class_ids:
            image_classes.append(class_ids[0])  # Use first class for stratification
            valid_images.append(img_path)
        else:
            logger.warning(f"No labels found for {img_path.name}, excluding from split")

    if not valid_images:
        raise ValueError("No valid images with labels found!")

    logger.info(f"Splitting {len(valid_images)} images...")

    # First split: train vs (val + test)
    train_ratio = ratios[0]
    temp_ratio = ratios[1] + ratios[2]

    train_images, temp_images, train_labels, temp_labels = train_test_split(
        valid_images,
        image_classes,
        train_size=train_ratio,
        stratify=image_classes,
        random_state=random_seed
    )

    # Second split: val vs test
    val_ratio = ratios[1] / temp_ratio
    val_images, test_images, _, _ = train_test_split(
        temp_images,
        temp_labels,
        train_size=val_ratio,
        stratify=temp_labels,
        random_state=random_seed
    )

    logger.info(f"Split complete: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

    return train_images, val_images, test_images


def copy_split(
    images: List[Path],
    src_images_dir: Path,
    src_labels_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    split_name: str
):
    """
    Copy images and labels to split directory.

    Args:
        images: List of image paths to copy
        src_images_dir: Source images directory
        src_labels_dir: Source labels directory
        dst_images_dir: Destination images directory
        dst_labels_dir: Destination labels directory
        split_name: Name of split (train/val/test) for logging
    """
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(images, desc=f"Copying {split_name}"):
        # Copy image
        dest_img = dst_images_dir / img_path.name
        shutil.copy2(img_path, dest_img)

        # Copy label
        label_path = src_labels_dir / (img_path.stem + '.txt')
        if label_path.exists():
            dest_label = dst_labels_dir / label_path.name
            shutil.copy2(label_path, dest_label)


def get_class_distribution(images: List[Path], labels_dir: Path) -> Dict[int, int]:
    """
    Calculate class distribution for a set of images.

    Args:
        images: List of image paths
        labels_dir: Directory containing label files

    Returns:
        Dictionary mapping class_id -> count
    """
    all_classes = []

    for img_path in images:
        class_ids = get_image_labels(img_path, labels_dir)
        all_classes.extend(class_ids)

    return dict(Counter(all_classes))


def generate_data_yaml(
    output_path: Path,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    class_names: List[str]
):
    """
    Generate data.yaml configuration file for YOLO training.

    Args:
        output_path: Path to save data.yaml
        train_dir: Path to training images directory
        val_dir: Path to validation images directory
        test_dir: Path to test images directory
        class_names: List of class names
    """
    data = {
        'train': str(train_dir.resolve()),
        'val': str(val_dir.resolve()),
        'test': str(test_dir.resolve()),
        'nc': len(class_names),
        'names': class_names
    }

    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Generated data.yaml: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test with stratification and deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 80/10/10 split
  python split_and_prep.py --src ./datasets/processed/all --out ./datasets/processed --split 0.8 0.1 0.1

  # Custom split with seed
  python split_and_prep.py --src ./datasets/processed/all --out ./datasets/processed --split 0.7 0.15 0.15 --seed 123

  # Generate data.yaml
  python split_and_prep.py --src ./datasets/processed/all --out ./datasets/processed --split 0.8 0.1 0.1 --data-yaml ./data.yaml
        """
    )

    parser.add_argument('--src', type=Path, required=True,
                        help='Source directory containing merged YOLO dataset (images/ and labels/)')
    parser.add_argument('--out', type=Path, required=True,
                        help='Output directory for train/val/test splits')
    parser.add_argument('--split', nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='Split ratios (must sum to 1.0). Default: 0.8 0.1 0.1')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no-stratify', action='store_true',
                        help='Disable stratified sampling (not recommended)')
    parser.add_argument('--data-yaml', type=Path, default=None,
                        help='Path to generate data.yaml file (default: <out>/data.yaml)')
    parser.add_argument('--classes', nargs='*', 
                        default=['plastic', 'metal', 'glass', 'paper', 'cardboard', 'other'],
                        help='Class names in order of class IDs')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Klasifikasi Sampah - Dataset Split & Preparation")
    logger.info("=" * 60)

    # Validate split ratios
    ratios = tuple(args.split)
    if abs(sum(ratios) - 1.0) > 0.001:
        logger.error(f"Split ratios must sum to 1.0, got {sum(ratios)}")
        return 1

    if any(r < 0 for r in ratios):
        logger.error("Split ratios must be non-negative")
        return 1

    # Check source directories
    src_images_dir = args.src / 'images'
    src_labels_dir = args.src / 'labels'

    if not src_images_dir.exists():
        logger.error(f"Source images directory not found: {src_images_dir}")
        return 1

    if not src_labels_dir.exists():
        logger.error(f"Source labels directory not found: {src_labels_dir}")
        return 1

    # Deduplicate images
    unique_images, duplicates = deduplicate_images(src_images_dir, src_labels_dir)

    if not unique_images:
        logger.error("No valid images found after deduplication!")
        return 1

    # Perform split
    if args.no_stratify:
        logger.warning("Stratification disabled - class distribution may be imbalanced")
        # Simple random split (not implemented - keeping it simple)
        raise NotImplementedError("Non-stratified split not yet implemented. Remove --no-stratify flag.")

    train_images, val_images, test_images = stratified_split(
        unique_images,
        src_labels_dir,
        ratios,
        args.seed
    )

    # Get class distributions
    train_dist = get_class_distribution(train_images, src_labels_dir)
    val_dist = get_class_distribution(val_images, src_labels_dir)
    test_dist = get_class_distribution(test_images, src_labels_dir)

    # Log distributions
    logger.info("")
    logger.info("Class Distribution:")
    logger.info(f"  Train: {train_dist}")
    logger.info(f"  Val:   {val_dist}")
    logger.info(f"  Test:  {test_dist}")

    # Copy files to splits
    logger.info("")
    logger.info("Copying files to split directories...")

    copy_split(
        train_images,
        src_images_dir,
        src_labels_dir,
        args.out / 'train' / 'images',
        args.out / 'train' / 'labels',
        'train'
    )

    copy_split(
        val_images,
        src_images_dir,
        src_labels_dir,
        args.out / 'val' / 'images',
        args.out / 'val' / 'labels',
        'val'
    )

    copy_split(
        test_images,
        src_images_dir,
        src_labels_dir,
        args.out / 'test' / 'images',
        args.out / 'test' / 'labels',
        'test'
    )

    # Generate data.yaml
    if args.data_yaml is None:
        args.data_yaml = args.out / 'data.yaml'

    generate_data_yaml(
        args.data_yaml,
        args.out / 'train' / 'images',
        args.out / 'val' / 'images',
        args.out / 'test' / 'images',
        args.classes
    )

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SPLIT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total unique images: {len(unique_images)}")
    logger.info(f"Duplicates removed: {len(duplicates)}")
    logger.info(f"Train: {len(train_images)} images ({ratios[0]*100:.1f}%)")
    logger.info(f"Val:   {len(val_images)} images ({ratios[1]*100:.1f}%)")
    logger.info(f"Test:  {len(test_images)} images ({ratios[2]*100:.1f}%)")
    logger.info(f"Data YAML: {args.data_yaml}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
