#!/usr/bin/env python3
"""
Dataset Split and Preparation Script

Split dataset into train/val/test with deduplication and stratified sampling.

Simple usage:
    python split_and_prep.py
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.image_utils import verify_image, hash_image
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Default configuration
DEFAULT_SRC = './datasets/processed/all'
DEFAULT_OUT = './datasets/processed'
DEFAULT_SPLIT = (0.8, 0.1, 0.1)  # train, val, test
DEFAULT_CLASSES = ['plastic', 'metal', 'glass', 'paper', 'cardboard', 'other']


def deduplicate_images(images_dir: Path, labels_dir: Path) -> Tuple[List[Path], int]:
    """Deduplicate images based on content hash."""
    logger.info("Deduplicating images...")
    
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    seen_hashes = {}
    unique_images = []
    dup_count = 0

    for img_path in tqdm(image_files, desc="Hashing images"):
        is_valid, error = verify_image(img_path)
        if not is_valid:
            logger.warning(f"Invalid: {img_path.name} - {error}")
            continue

        img_hash = hash_image(img_path)
        if img_hash in seen_hashes:
            dup_count += 1
        else:
            seen_hashes[img_hash] = img_path
            unique_images.append(img_path)

    logger.info(f"Found {len(unique_images)} unique, {dup_count} duplicates")
    return unique_images, dup_count


def get_labels(image_path: Path, labels_dir: Path) -> List[int]:
    """Get class IDs from label file."""
    label_path = labels_dir / f'{image_path.stem}.txt'
    if not label_path.exists():
        return []
    
    return [int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip()]


def split_dataset(
    images: List[Path],
    labels_dir: Path,
    ratios: Tuple[float, float, float],
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Stratified split into train/val/test."""
    logger.info(f"Splitting with ratios {ratios}...")
    
    # Get labels for stratification
    valid_images = []
    image_classes = []
    
    for img in tqdm(images, desc="Reading labels"):
        class_ids = get_labels(img, labels_dir)
        if class_ids:
            valid_images.append(img)
            image_classes.append(class_ids[0])  # Use first class
        else:
            logger.warning(f"No labels: {img.name}")
    
    if not valid_images:
        raise ValueError("No valid images with labels!")
    
    # Split: train vs (val+test)
    train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
        valid_images, image_classes,
        train_size=ratios[0],
        stratify=image_classes,
        random_state=seed
    )
    
    # Split: val vs test
    val_size = ratios[1] / (ratios[1] + ratios[2])
    val_imgs, test_imgs, _, _ = train_test_split(
        temp_imgs, temp_lbls,
        train_size=val_size,
        stratify=temp_lbls,
        random_state=seed
    )
    
    logger.info(f"Split: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    return train_imgs, val_imgs, test_imgs


def copy_files(images: List[Path], src_img_dir: Path, src_lbl_dir: Path, 
               dst_img_dir: Path, dst_lbl_dir: Path, name: str):
    """Copy images and labels to destination."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    for img in tqdm(images, desc=f"Copying {name}"):
        shutil.copy2(img, dst_img_dir / img.name)
        
        lbl = src_lbl_dir / f'{img.stem}.txt'
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl_dir / lbl.name)


def get_distribution(images: List[Path], labels_dir: Path) -> Dict[int, int]:
    """Calculate class distribution."""
    all_classes = []
    for img in images:
        all_classes.extend(get_labels(img, labels_dir))
    return dict(Counter(all_classes))


def create_data_yaml(out_path: Path, train_dir: Path, val_dir: Path, 
                     test_dir: Path, classes: List[str]):
    """Generate data.yaml for YOLO."""
    data = {
        'train': str(train_dir.resolve()),
        'val': str(val_dir.resolve()),
        'test': str(test_dir.resolve()),
        'nc': len(classes),
        'names': classes
    }
    
    out_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    logger.info(f"Generated: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Simple usage:
  python split_and_prep.py

Custom split:
  python split_and_prep.py --split 0.7 0.15 0.15
        """
    )

    parser.add_argument('--src', type=Path, default=Path(DEFAULT_SRC),
                        help=f'Source directory (default: {DEFAULT_SRC})')
    parser.add_argument('--out', type=Path, default=Path(DEFAULT_OUT),
                        help=f'Output directory (default: {DEFAULT_OUT})')
    parser.add_argument('--split', nargs=3, type=float, default=list(DEFAULT_SPLIT),
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help=f'Split ratios (default: {DEFAULT_SPLIT[0]} {DEFAULT_SPLIT[1]} {DEFAULT_SPLIT[2]})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--classes', nargs='*', default=DEFAULT_CLASSES,
                        help='Class names')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Dataset Split & Preparation")
    logger.info("=" * 60)

    # Validate ratios
    ratios = tuple(args.split)
    if abs(sum(ratios) - 1.0) > 0.001:
        logger.error(f"Split ratios must sum to 1.0, got {sum(ratios)}")
        return 1

    # Check source
    src_img_dir = args.src / 'images'
    src_lbl_dir = args.src / 'labels'

    if not src_img_dir.exists():
        logger.error(f"Images not found: {src_img_dir}")
        return 1
    if not src_lbl_dir.exists():
        logger.error(f"Labels not found: {src_lbl_dir}")
        return 1

    # Deduplicate
    unique_images, dup_count = deduplicate_images(src_img_dir, src_lbl_dir)
    if not unique_images:
        logger.error("No valid images found!")
        return 1

    # Split
    train_imgs, val_imgs, test_imgs = split_dataset(
        unique_images, src_lbl_dir, ratios, args.seed
    )

    # Show distributions
    train_dist = get_distribution(train_imgs, src_lbl_dir)
    val_dist = get_distribution(val_imgs, src_lbl_dir)
    test_dist = get_distribution(test_imgs, src_lbl_dir)

    logger.info("")
    logger.info("Class Distribution:")
    logger.info(f"  Train: {train_dist}")
    logger.info(f"  Val:   {val_dist}")
    logger.info(f"  Test:  {test_dist}")

    # Copy files
    logger.info("")
    copy_files(train_imgs, src_img_dir, src_lbl_dir,
               args.out / 'train' / 'images', args.out / 'train' / 'labels', 'train')
    copy_files(val_imgs, src_img_dir, src_lbl_dir,
               args.out / 'val' / 'images', args.out / 'val' / 'labels', 'val')
    copy_files(test_imgs, src_img_dir, src_lbl_dir,
               args.out / 'test' / 'images', args.out / 'test' / 'labels', 'test')

    # Generate data.yaml
    data_yaml = args.out / 'data.yaml'
    create_data_yaml(
        data_yaml,
        args.out / 'train' / 'images',
        args.out / 'val' / 'images',
        args.out / 'test' / 'images',
        args.classes
    )

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Unique images: {len(unique_images)}")
    logger.info(f"Duplicates removed: {dup_count}")
    logger.info(f"Train: {len(train_imgs)} ({ratios[0]*100:.0f}%)")
    logger.info(f"Val:   {len(val_imgs)} ({ratios[1]*100:.0f}%)")
    logger.info(f"Test:  {len(test_imgs)} ({ratios[2]*100:.0f}%)")
    logger.info(f"Config: {data_yaml}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
