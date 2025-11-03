"""
Klasifikasi Sampah - Utility Package

This package contains helper modules for the waste classification pipeline:
- logger: Logging configuration
- image_utils: Image verification and hashing
- label_mapper: Label standardization and mapping
- dataset_stats: Dataset statistics and reporting
- annotation_parsers: Multi-format annotation parsing and conversion
"""

__version__ = "1.0.0"
__author__ = "Klasifikasi Sampah Team"

# Import commonly used utilities
from .logger import setup_logger

__all__ = [
    "setup_logger",
]
