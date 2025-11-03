"""
Image utility functions for the waste classification pipeline.

Provides image verification, hashing for deduplication, and validation.
"""

import hashlib
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image


def verify_image(image_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Verify that an image file is valid and readable.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if image is valid, False otherwise
        - error_message: None if valid, error description if invalid

    Example:
        >>> is_valid, error = verify_image(Path("image.jpg"))
        >>> if not is_valid:
        >>>     print(f"Invalid image: {error}")
    """
    if not image_path.exists():
        return False, f"File does not exist: {image_path}"

    if not image_path.is_file():
        return False, f"Path is not a file: {image_path}"

    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify image integrity
        
        # Re-open to check format and dimensions (verify() closes the file)
        with Image.open(image_path) as img:
            width, height = img.size
            if width <= 0 or height <= 0:
                return False, f"Invalid dimensions: {width}x{height}"
            
            if img.format not in ['JPEG', 'PNG', 'BMP', 'JPG']:
                return False, f"Unsupported format: {img.format}"
        
        return True, None

    except Exception as e:
        return False, f"Failed to open image: {str(e)}"


def hash_image(image_path: Path) -> str:
    """
    Compute SHA-256 hash of image file content for deduplication.

    Args:
        image_path: Path to image file

    Returns:
        Hexadecimal SHA-256 hash string

    Example:
        >>> hash1 = hash_image(Path("img1.jpg"))
        >>> hash2 = hash_image(Path("img2.jpg"))
        >>> if hash1 == hash2:
        >>>     print("Duplicate images detected")
    """
    sha256_hash = hashlib.sha256()
    
    with open(image_path, "rb") as f:
        # Read in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def get_image_info(image_path: Path) -> dict:
    """
    Extract metadata from an image file.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary with keys: filename, width, height, format, size_bytes

    Raises:
        ValueError: If image is invalid or cannot be opened

    Example:
        >>> info = get_image_info(Path("test.jpg"))
        >>> print(f"Image: {info['width']}x{info['height']} {info['format']}")
    """
    is_valid, error = verify_image(image_path)
    if not is_valid:
        raise ValueError(f"Invalid image: {error}")

    with Image.open(image_path) as img:
        return {
            "filename": image_path.name,
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "size_bytes": image_path.stat().st_size,
        }
